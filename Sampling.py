"""
Generates result-section figures for one patient: loads a trained
checkpoint, picks a slice (middle by default), runs full
reverse-diffusion sampling, and saves MRI / LD-PET / Ground Truth /
Generated / Absolute Error as separate, clean (no axes/titles) PNG
images -- suitable for composing into a paper figure.

Usage:
    python sample_for_figures.py --patient_dir /path/to/PATIENT_ID \
                                  --checkpoint /path/to/diffusion_final_..._fold0.pth

IMPORTANT: adjust CHECKPOINT_DIR/AE_CHECKPOINT_PATH defaults and the
`from train_diffusion_MRI import sample_batch` line below to match
your actual training script's filename.
"""

import argparse
import random
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler

from dataset_MRI import SliceDataset
from metrics import compute_all_metrics
from models.PET_KL_Autoencoder import KLAutoencoder
from models.diffusion_model_MRI import ConditionalUNet
from models.Condition_Encoder import Condition_Encoder

# Reuses the EXACT sampling loop AND fold-splitting logic from the
# training script, so the test-set membership check below is
# guaranteed to match how folds were actually built.
from train_diffusion_MRI import sample_batch, make_kfold_splits  # <-- adjust to your actual training script filename

# ============================================================
# Must match your training script's dataset root / fold settings
# exactly, or the recomputed splits won't match what was trained.
# ============================================================
DATASET_ROOT = Path("/home/pedrocarreiro/Desktop/Dataset_Normalized3/")
N_FOLDS = 5
VAL_RATIO = 0.15
SPLIT_SEED = 42

# Must match the training script's external-dataset settings exactly,
# for the same reason as above.
EXTERNAL_TEST_DATASET_ROOT = Path("/home/pedrocarreiro/Desktop/Dataset_2/")
EXTERNAL_TRAIN_RATIO = 0.7
EXTERNAL_SPLIT_SEED = 42


def verify_patient_in_fold_test_set(patient_dir, fold_idx):
    """
    Recomputes the same deterministic k-fold split training used, and
    checks that `patient_dir` is actually in fold `fold_idx`'s held-out
    test set -- not its train or val set. Raises loudly if not, since
    generating a figure from a train/val patient would silently show
    memorization instead of genuine generalization.
    """
    all_exams = sorted([d for d in DATASET_ROOT.iterdir() if d.is_dir()])
    splits = make_kfold_splits(all_exams, N_FOLDS, val_ratio=VAL_RATIO, seed=SPLIT_SEED)
    train_exams, val_exams, test_exams = splits[fold_idx]

    patient_name = patient_dir.name
    test_names = {d.name for d in test_exams}
    train_names = {d.name for d in train_exams}
    val_names = {d.name for d in val_exams}

    if patient_name in test_names:
        print(f"[OK] {patient_name} is in fold {fold_idx}'s held-out TEST set.")
    elif patient_name in train_names:
        raise ValueError(
            f"{patient_name} was in fold {fold_idx}'s TRAINING set -- this model has "
            f"already seen this patient. Pick a different patient from the test set, "
            f"or use a different fold's checkpoint."
        )
    elif patient_name in val_names:
        raise ValueError(
            f"{patient_name} was in fold {fold_idx}'s VALIDATION set -- used for "
            f"checkpoint selection during training. Pick a patient from the test set instead."
        )
    else:
        print(f"[WARNING] {patient_name} was not found in fold {fold_idx}'s split at all -- "
              f"this is expected if it's an EXTERNAL-dataset patient (not part of "
              f"{DATASET_ROOT}). Skipping the internal train/val/test check.")


def parse_train_on_external_from_filename(checkpoint_path):
    """
    Your run_name includes an "exttrain{0|1}" tag even though the
    checkpoint's saved config dict doesn't store train_on_external
    directly. Parse it from the filename instead -- no checkpoint
    patching or retraining needed.

    Returns True/False if the tag is found, or None if this
    checkpoint predates the tag (older run_name format).
    """
    match = re.search(r"exttrain(\d)", str(checkpoint_path))
    if match is None:
        return None
    return bool(int(match.group(1)))


def get_external_split(external_dataset_root, external_train_ratio, seed=EXTERNAL_SPLIT_SEED):
    """
    Reproduces the EXACT external train/test split used during
    training -- deterministic given the same directory listing, ratio,
    and seed. No retraining needed; this recomputes the same
    random.Random(seed).shuffle(...) call training already made.
    """
    external_exams = sorted([d for d in external_dataset_root.iterdir() if d.is_dir()])
    external_exams_shuffled = external_exams.copy()
    random.Random(seed).shuffle(external_exams_shuffled)

    n_external_train = int(len(external_exams_shuffled) * external_train_ratio)
    external_train_exams = external_exams_shuffled[:n_external_train]
    external_test_exams = external_exams_shuffled[n_external_train:]
    return external_train_exams, external_test_exams


def verify_patient_in_external_split(patient_dir, external_train_ratio=EXTERNAL_TRAIN_RATIO):
    """
    Checks whether `patient_dir` (an external-dataset patient) falls
    in the external TRAIN portion (already seen by every fold's model,
    if TRAIN_ON_EXTERNAL was used) or the external TEST portion
    (genuinely held out). Raises loudly on a train-set patient, same
    reasoning as verify_patient_in_fold_test_set.
    """
    external_train_exams, external_test_exams = get_external_split(
        EXTERNAL_TEST_DATASET_ROOT, external_train_ratio, EXTERNAL_SPLIT_SEED
    )

    patient_name = patient_dir.name
    train_names = {d.name for d in external_train_exams}
    test_names = {d.name for d in external_test_exams}

    if patient_name in test_names:
        print(f"[OK] {patient_name} is in the held-out EXTERNAL TEST set.")
    elif patient_name in train_names:
        raise ValueError(
            f"{patient_name} was used in EXTERNAL TRAINING data (every fold's model "
            f"has already seen this patient). Pick a different external patient from "
            f"the held-out test portion instead."
        )
    else:
        print(f"[WARNING] {patient_name} was not found in the external dataset "
              f"({EXTERNAL_TEST_DATASET_ROOT}) at all.")

# ============================================================
# DEFAULTS -- override via CLI args below, or edit directly
# ============================================================
DEFAULT_AE_CHECKPOINT_PATH = Path(
    "/home/pedrocarreiro/Desktop/Latent_Diffusion_Model/checkpoints/kl_autoencoder_best.pth"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/pedrocarreiro/Desktop/Latent_Diffusion_Model/result_figures"
)

TARGET_SIZE = (256, 256)
NUM_INFERENCE_STEPS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_diffusion_model(checkpoint_path, device):
    """
    Reconstructs the exact model architecture from the config stored
    inside the checkpoint itself (use_mri/use_ldpet/use_dose_cond/
    target_dose) -- self-describing, so this always matches whatever
    that checkpoint was actually trained with.
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = dict(checkpoint["config"])  # copy, so we can safely add to it
    if "fold" in checkpoint:
        config["fold"] = checkpoint["fold"]  # stored top-level in the checkpoint, not inside "config"

    unet = ConditionalUNet(
        latent_dim=32, time_dim=256,
        use_mri=config["use_mri"], use_ldpet=config["use_ldpet"],
        use_dose_cond=config["use_dose_cond"],
    ).to(device)
    unet.load_state_dict(checkpoint["model_state_dict"])
    unet.eval()

    mri_encoder = None
    if config["use_mri"]:
        mri_encoder = Condition_Encoder(in_channels=1).to(device)
        mri_encoder.load_state_dict(checkpoint["mri_encoder_state_dict"])
        mri_encoder.eval()

    ldpet_encoder = None
    if config["use_ldpet"]:
        ldpet_encoder = Condition_Encoder(in_channels=1).to(device)
        ldpet_encoder.load_state_dict(checkpoint["ldpet_encoder_state_dict"])
        ldpet_encoder.eval()

    latent_std = checkpoint["latent_std"]
    return unet, mri_encoder, ldpet_encoder, latent_std, config


def save_clean_image(array, save_path, cmap="hot", vmin=None, vmax=None):
    """Saves one 2D array as a clean image: no axes, no border, no title."""
    fig = plt.figure(figsize=(4, 4), frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def find_slice_sample(patient_dir, target_dose, target_size, slice_z=None):
    """
    Builds a SliceDataset for just this one patient -- reusing the
    exact same loading/reorientation/crop logic used in
    training/evaluation -- and returns (dataset, sample_index) for
    the requested slice, or the middle slice if slice_z is None.
    """
    dataset = SliceDataset([patient_dir], target_size=target_size, dose_levels=[target_dose])

    if len(dataset) == 0:
        raise ValueError(f"No usable slices found for {patient_dir} at dose {target_dose}")

    z_values = sorted(set(s["z"] for s in dataset.samples))

    if slice_z is None:
        target_z = z_values[len(z_values) // 2]
    else:
        if slice_z not in z_values:
            raise ValueError(f"Slice z={slice_z} not available for this patient/dose "
                              f"(available range: {z_values[0]}-{z_values[-1]})")
        target_z = slice_z

    for idx, sample in enumerate(dataset.samples):
        if sample["z"] == target_z:
            return dataset, idx, target_z

    raise ValueError("Could not locate a sample at the requested slice.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_dir", type=str, required=True,
                         help="Path to the patient's exam directory (containing PET/MRI_trimmed/PET_LD).")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a diffusion_final_*.pth checkpoint.")
    parser.add_argument("--ae_checkpoint", type=str, default=str(DEFAULT_AE_CHECKPOINT_PATH))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--slice_z", type=int, default=None,
                         help="Specific slice index to sample. Default: auto-pick the middle slice.")
    args = parser.parse_args()

    patient_dir = Path(args.patient_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")

    # ---- Load autoencoder ----
    autoencoder = KLAutoencoder().to(DEVICE)
    ae_checkpoint = torch.load(args.ae_checkpoint, map_location=DEVICE, weights_only=False)
    autoencoder.load_state_dict(ae_checkpoint["model_state_dict"])
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    # ---- Load diffusion model (config read from checkpoint itself) ----
    unet, mri_encoder, ldpet_encoder, latent_std, config = load_diffusion_model(args.checkpoint, DEVICE)
    print(f"Loaded model: use_mri={config['use_mri']}, use_ldpet={config['use_ldpet']}, "
          f"use_dose_cond={config['use_dose_cond']}, target_dose={config['target_dose']}, "
          f"fold={config.get('fold', 'unknown')}")

    if "fold" in config:
        try:
            patient_dir.relative_to(DATASET_ROOT)
            is_external = False
        except ValueError:
            is_external = True

        if is_external:
            train_on_external = parse_train_on_external_from_filename(args.checkpoint)
            if train_on_external is None:
                train_on_external = config.get("train_on_external", None)

            if train_on_external:
                verify_patient_in_external_split(patient_dir)
            elif train_on_external is False:
                print(f"[OK] This checkpoint's run_name indicates TRAIN_ON_EXTERNAL=False -- "
                      f"the full external set was held-out test, so {patient_dir.name} is fine to use.")
            else:
                print(f"[WARNING] Could not determine TRAIN_ON_EXTERNAL for this checkpoint "
                      f"(no 'exttrain' tag in filename, no 'train_on_external' in config) -- "
                      f"cannot verify {patient_dir.name}'s test-set membership automatically.")
        else:
            verify_patient_in_fold_test_set(patient_dir, config["fold"])
    else:
        print("[WARNING] Checkpoint config has no 'fold' entry -- cannot verify test-set membership.")

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.01,
        beta_schedule="squaredcos_cap_v2", prediction_type="epsilon",
        clip_sample=False, clip_sample_range=3,
    )

    # ---- Pick the slice ----
    dataset, sample_idx, z = find_slice_sample(
        patient_dir, config["target_dose"], TARGET_SIZE, slice_z=args.slice_z
    )
    pet, mri, ld_pet, dose_level = dataset[sample_idx]
    print(f"Patient: {patient_dir.name}, slice z={z}, dose={int(dose_level.item())}%")

    # ---- Sample ----
    generated_pet = sample_batch(
        unet, mri_encoder, ldpet_encoder, autoencoder, noise_scheduler,
        mri.unsqueeze(0), ld_pet.unsqueeze(0), dose_level.unsqueeze(0),
        latent_std, DEVICE, num_inference_steps=NUM_INFERENCE_STEPS,
    )

    mri_np = mri[0].numpy()
    ld_np = ld_pet[0].numpy()
    gt_np = pet[0].numpy()
    gen_np = generated_pet[0, 0].cpu().numpy()
    error_np = np.abs(gen_np - gt_np)

    # ---- Metrics for this sample (useful for the figure caption) ----
    sample_metrics = compute_all_metrics(gen_np, gt_np)
    print("Per-sample metrics:")
    for k, v in sample_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ---- Save separate, clean images ----
    # Ground truth and generated share the same color scale (based on
    # ground truth's range) so they're visually comparable side by
    # side in the paper.
    gt_vmin, gt_vmax = float(gt_np.min()), float(gt_np.max())

    # MRI/LD-PET/ground-truth are identical regardless of which model
    # produced the reconstruction, so they're saved once, unqualified.
    # Generated/error DO depend on the model, so they're tagged with
    # its config to avoid different models silently overwriting each
    # other's output for the same patient/slice.
    model_tag = f"mri{int(config['use_mri'])}_ldpet{int(config['use_ldpet'])}"
    shared_prefix = output_dir / f"{patient_dir.name}_z{z}"
    model_prefix = output_dir / f"{patient_dir.name}_z{z}_{model_tag}"

    save_clean_image(mri_np, f"{shared_prefix}_mri.png", cmap="gray")
    save_clean_image(ld_np, f"{shared_prefix}_ldpet.png", cmap="hot", vmin=gt_vmin, vmax=gt_vmax)
    save_clean_image(gt_np, f"{shared_prefix}_groundtruth.png", cmap="hot", vmin=gt_vmin, vmax=gt_vmax)
    save_clean_image(gen_np, f"{model_prefix}_generated.png", cmap="hot", vmin=gt_vmin, vmax=gt_vmax)
    save_clean_image(error_np, f"{model_prefix}_error.png", cmap="inferno")

    print(f"\nSaved images to {output_dir} (shared prefix '{patient_dir.name}_z{z}_*', "
          f"model-specific prefix '..._{model_tag}_*')")