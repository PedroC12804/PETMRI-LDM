import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
import lpips
import gc
import numpy as np

from dataset_MRI import SliceDataset
from metrics import compute_all_metrics, aggregate, compute_fid_from_features, get_fid_features
from models.PET_KL_Autoencoder import KLAutoencoder
from models.diffusion_model_MRI import ConditionalUNet
from models.Condition_Encoder import Condition_Encoder



# ============================================================
#CONFIG
# ============================================================
USE_MRI = True
USE_LDPET = True
USE_EXTERNAL_TEST = True
RUN_TEST_EVAL = True
USE_DOSE_COND = False
TRAIN_ON_ALL_DOSES = False
TARGET_DOSE = 1
TRAIN_ON_EXTERNAL = True
EXTERNAL_TRAIN_RATIO = 0.7

N_FOLDS = 5
SKIP_TRAINING = False
TEST_METRICS_MAX_SAMPLES = None  # None = full test set (thorough, slow)
METRIC_SAMPLE_SIZE = 8

run_name = (
    f"mri{int(USE_MRI)}_ldpet{int(USE_LDPET)}_"
    f"target{TARGET_DOSE}_ftarget{TRAIN_ON_EXTERNAL}"
)

PROJECT_ROOT = Path("/home/pedrocarreiro/Desktop/Latent_Diffusion_Model/")
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
DIFFUSION_BEST_PATH = CHECKPOINT_DIR / f"diffusion_best_{run_name}.pth"
DIFFUSION_FINAL_PATH = CHECKPOINT_DIR / f"diffusion_final_{run_name}.pth"

DATASET_ROOT = Path("/home/pedrocarreiro/Desktop/Dataset_Normalized3/")
EXTERNAL_TEST_DATASET_ROOT = Path("/home/pedrocarreiro/Desktop/Dataset_2/")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# DIFFUSION CONFIG
# ============================================
DIFFUSION_CONFIG = {
    "num_train_timesteps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.01,
    "beta_schedule": "squaredcos_cap_v2",
    "prediction_type": "epsilon",
}

# ============================================
# AUTOENCODER CONFIG
# ============================================
AE_CONFIG = {
    "latent_dim": 32,
    "target_size": (256, 256),
    "input_channels": 1,
    "batch_size": 4,
}

# ============================================
# TRAINING CONFIG
# ============================================
TRAIN_CONFIG = {
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_epochs": 100,
    "num_workers": 0,
    "log_every": 20,
    "save_every": 40,
    "warmup_steps": 500,
}
# ============================================
# ABLATION CONFIG
# ============================================
ABLATION_CONFIG = {
    "use_mri": USE_MRI,
    "use_ldpet": USE_LDPET,
    "run_name": run_name,
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def make_kfold_splits(exams, n_folds=5, val_ratio=0.15, seed=42):
    """
    Splits `exams` into `n_folds` folds using sklearn's KFold. Returns
    a list of (train_exams, val_exams, test_exams) tuples, one per
    fold -- fold i's test set is fold i itself; the remaining
    n_folds-1 folds are further split into train/val.
    """
    exams = list(exams)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    splits = []

    for remaining_idx, test_idx in kf.split(exams):
        remaining = [exams[i] for i in remaining_idx]
        test_exams = [exams[i] for i in test_idx]

        random.Random(seed).shuffle(remaining)  # reproducible train/val split within this fold
        n_val = int(len(remaining) * val_ratio)
        val_exams = remaining[:n_val]
        train_exams = remaining[n_val:]

        splits.append((train_exams, val_exams, test_exams))

    return splits

@torch.no_grad()
def compute_latent_std(loader, autoencoder, device):

    latents = []

    for pet, _,_,_ in tqdm(loader, desc="Computing latent std"):

        pet = pet.to(device)

        mu, logvar = autoencoder.encoder(pet)

        latents.append(mu.cpu())

    latents = torch.cat(latents)

    std = latents.std()

    print(f"Latent std: {std:.4f}")

    return std.to(device)

def build_fixed_viz_set(dataset, n_samples=5, seed=42):
    """
    Picks a fixed, reproducible set of `n_samples` indices from
    `dataset` for visualization. Since val/test are now filtered to
    TARGET_DOSE only, there's no multi-dose row structure anymore --
    just a handful of representative reconstructions at the one dose
    level this run is about.
    """
    n_samples = min(n_samples, len(dataset))
    indices = random.Random(seed).sample(range(len(dataset)), n_samples)
    return indices

@torch.no_grad()
def sample_batch(unet, mri_encoder, ldpet_encoder, autoencoder, scheduler,
                  mri, ld_pet, dose_level, latent_std, device,
                  num_inference_steps=1000, initial_noise=None):
    """
    Runs full reverse-diffusion sampling for one batch and returns the
    decoded PET image tensor. Shared by visualize_generation (fixed
    viz batch) and the metric-computation hooks below, so there's one
    sampling implementation, not two that could drift apart.
    """
    batch_size = mri.shape[0]

    mri = mri.to(device)
    ld_pet = ld_pet.to(device)
    dose_level = dose_level.to(device)

    if initial_noise is not None:
        latents = initial_noise.clone().to(device)
    else:
        latents = torch.randn((batch_size, AE_CONFIG["latent_dim"], 16, 16), device=device)

    mri_features = mri_encoder(mri) if unet.use_mri else None
    ldpet_features = ldpet_encoder(ld_pet) if unet.use_ldpet else None

    scheduler.set_timesteps(num_inference_steps)

    for t in scheduler.timesteps:
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        noise_pred = unet(latents, t_batch, dose_level, mri_features, ldpet_features)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents * latent_std
    generated_pet = autoencoder.decoder(latents)

    return generated_pet

def plot_sample_grid(image_pairs, n_samples, save_path, title_prefix=""):
    n_samples = min(n_samples, len(image_pairs))
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i in range(n_samples):
        gen_np, gt_np = image_pairs[i]
        axes[i][0].imshow(gt_np, cmap="hot")
        axes[i][0].set_title("Ground Truth")
        axes[i][0].axis("off")

        axes[i][1].imshow(gen_np, cmap="hot")
        axes[i][1].set_title("Generated")
        axes[i][1].axis("off")

        error_map = np.abs(gen_np - gt_np)
        axes[i][2].imshow(error_map, cmap="inferno")
        axes[i][2].set_title("Absolute Error")
        axes[i][2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

@torch.no_grad()
def evaluate_metrics_on_dataset(unet, mri_encoder, ldpet_encoder, autoencoder,
                                 dataset, noise_scheduler, latent_std, device,
                                 lpips_model, n_samples=None, seed=123,
                                 num_inference_steps=1000, collect_images=False,
                                 collect_fid_features=False):
    total = len(dataset)
    if n_samples is None or n_samples >= total:
        indices = list(range(total))
    else:
        indices = random.Random(seed).sample(range(total), n_samples)

    raw = []
    images = [] if collect_images else None
    fid_gt_feats = [] if collect_fid_features else None
    fid_gen_feats = [] if collect_fid_features else None

    for idx in tqdm(indices, desc="Computing metrics", leave=False):
        pet, mri, ld_pet, dose_level = dataset[idx]

        generated_pet = sample_batch(
            unet, mri_encoder, ldpet_encoder, autoencoder, noise_scheduler,
            mri.unsqueeze(0), ld_pet.unsqueeze(0), dose_level.unsqueeze(0),
            latent_std, device, num_inference_steps=num_inference_steps,
        )

        gt_np = pet[0].cpu().numpy()
        gen_np = generated_pet[0, 0].cpu().numpy()

        raw.append(compute_all_metrics(gen_np, gt_np, lpips_model=lpips_model, device=device))

        if collect_images:
            images.append((gen_np, gt_np))

        if collect_fid_features:
            # Extract the Inception feature immediately, discard the
            # raw image right after -- this is what avoids storing
            # full-resolution arrays for every test sample.
            fid_gt_feats.append(get_fid_features([gt_np], device)[0])
            fid_gen_feats.append(get_fid_features([gen_np], device)[0])

    fid_features = (fid_gt_feats, fid_gen_feats) if collect_fid_features else None
    return raw, aggregate(raw), images, fid_features
# ===============================================================================
def train_one_epoch(unet,mri_encoder,ldpet_encoder, pet_autoencoder,loader,optimizer,noise_scheduler,
    lr_scheduler,latent_std,device,):
    unet.train()

    if mri_encoder is not None:
        mri_encoder.train()
    if ldpet_encoder is not None:
        ldpet_encoder.train()

    total_loss = 0

    for pet,mri, ld_pet, dose_level in tqdm(loader, desc="Training", leave=False):

        pet = pet.to(device)
        mri = mri.to(device)
        ld_pet = ld_pet.to(device)
        dose_level = dose_level.to(device)

        # Get raw latents
        with torch.no_grad():
            mu, logvar = pet_autoencoder.encoder(pet)
            latents = mu/latent_std

        mri_features = mri_encoder(mri) if unet.use_mri else None
        ldpet_features = ldpet_encoder(ld_pet) if unet.use_ldpet else None

        #print(latents.shape)
        #print(mri_tokens.shape)

        # Standard noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device
        ).long()

        # Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        noise_pred = unet(noisy_latents, timesteps,dose_level, mri_features,ldpet_features)

        # Loss on noise prediction
        loss = F.mse_loss(noise_pred, noise)

        # Backpropagate (will train both the Unet and the MRI Encoder)
        optimizer.zero_grad()
        loss.backward()

        clip_params = list(unet.parameters())
        if unet.use_mri:
            clip_params += list(mri_encoder.parameters())
        if unet.use_ldpet:
            clip_params += list(ldpet_encoder.parameters())

        torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(unet,mri_encoder,ldpet_encoder, pet_autoencoder,loader,
    noise_scheduler,latent_std, device,):

    unet.eval()

    if mri_encoder is not None:
        mri_encoder.eval()
    if ldpet_encoder is not None:
        ldpet_encoder.eval()

    total_loss = 0

    for pet,mri, ld_pet,dose_level in tqdm(loader, desc="Validation", leave=False):
        pet = pet.to(device)
        mri = mri.to(device)
        ld_pet = ld_pet.to(device)
        dose_level = dose_level.to(device)

        #encode PET
        mu, logvar = pet_autoencoder.encoder(pet)

        latents = mu / latent_std

        mri_features = mri_encoder(mri) if unet.use_mri else None
        ldpet_features = ldpet_encoder(ld_pet) if unet.use_ldpet else None

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device
        ).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        noise_pred = unet(noisy_latents, timesteps, dose_level, mri_features,ldpet_features)
        loss = F.mse_loss(noise_pred, noise) #this loss is timestep dependent and might be interesting consider linking this to a specific timestep

        total_loss += loss.item()

    return total_loss / len(loader)


def visualize_generation(
    unet,
    mri_encoder,
    ldpet_encoder,
    autoencoder,
    scheduler,
    pet,
    mri,
    ld_pet,
    dose_level,
    fixed_noise,
    latent_std,
    device,
    epoch,
    num_inference_steps=1000,
):

    unet.eval()
    if mri_encoder is not None:
        mri_encoder.eval()
    if ldpet_encoder is not None:
        ldpet_encoder.eval()

    # ============================================
    # Move data to device
    # ============================================
    pet = pet.to(device)
    mri = mri.to(device)
    ld_pet = ld_pet.to(device)
    dose_level = dose_level.to(device)

    # Clone noise so original stays unchanged
    latents = fixed_noise.clone().to(device)

    generated_pet = sample_batch(
        unet, mri_encoder, ldpet_encoder, autoencoder, scheduler,
        mri, ld_pet, dose_level, latent_std, device,
        num_inference_steps=num_inference_steps, initial_noise=fixed_noise,
    )

    # ============================================
    # Debug statistics
    # ============================================
    print("\nGenerated latent stats:")
    print(
        f"mean={latents.mean():.4f}, "
        f"std={latents.std():.4f}, "
        f"min={latents.min():.4f}, "
        f"max={latents.max():.4f}"
    )

    print("\nGenerated PET stats:")
    print(
        f"mean={generated_pet.mean():.4f}, "
        f"std={generated_pet.std():.4f}, "
        f"min={generated_pet.min():.4f}, "
        f"max={generated_pet.max():.4f}"
    )

    # ============================================
    # Visualization
    # ============================================
    batch_size = pet.shape[0]

    fig, axes = plt.subplots(
        batch_size,
        5,
        figsize=(20, 4 * batch_size)
    )

    # Handle batch_size == 1
    if batch_size == 1:
        axes = [axes]

    for i in range(batch_size):

        # ----------------------------------------
        # MRI
        # ----------------------------------------
        axes[i][0].imshow(
            mri[i, 0].cpu().numpy(),
            cmap="gray"
        )
        axes[i][0].set_title("MRI")
        axes[i][0].axis("off")

        # ----------------------------------------
        # LD-PET (input)
        # ----------------------------------------
        axes[i][1].imshow(ld_pet[i, 0].cpu().numpy(), cmap="hot")
        axes[i][1].set_title(f"LD-PET ({int(dose_level[i].item())}%)") #important dose visualization
        axes[i][1].axis("off")

        # ----------------------------------------
        # Ground Truth PET
        # ----------------------------------------
        axes[i][2].imshow(
            pet[i, 0].cpu().numpy(),
            cmap="hot"
        )
        axes[i][2].set_title("Ground Truth PET")
        axes[i][2].axis("off")

        # ----------------------------------------
        # Generated PET
        # ----------------------------------------
        axes[i][3].imshow(
            generated_pet[i, 0].cpu().numpy(),
            cmap="hot"
        )
        axes[i][3].set_title("Generated PET")
        axes[i][3].axis("off")

        # ----------------------------------------
        # Absolute Error Map
        # ----------------------------------------
        error_map = torch.abs(
            generated_pet[i, 0] - pet[i, 0]
        )

        axes[i][4].imshow(
            error_map.cpu().numpy(),
            cmap="inferno"
        )
        axes[i][4].set_title("Absolute Error")
        axes[i][4].axis("off")

    plt.tight_layout()

    save_path = PROJECT_ROOT / ("Images")
    save_path.mkdir(parents=True, exist_ok=True)
    save_figure = save_path /  f"generated_epoch_{epoch+1}_{run_name}.png"

    plt.savefig(save_figure, dpi=150)
    plt.close()

    wandb.log({
        "generated_samples": wandb.Image(save_figure)
    })

# ============================================
# MAIN TRAINING
# ============================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # ============================================
    # Load Conditional Encoders
    # ============================================

    print("\n" + "=" * 50)
    print(f"Ablation config: USE_MRI={USE_MRI}, USE_LDPET={USE_LDPET}")
    print("=" * 50)


    # ============================================
    # Create dataset and dataloaders
    # ============================================
    print("\n" + "=" * 50)
    print("Loading dataset...")
    print("=" * 50)

    all_exams = sorted([d for d in DATASET_ROOT.iterdir() if d.is_dir()])

    #all_exams=all_exams[:100] #!!!!!!!!!!!!!!!!!!!!!!!!!
    train_dose_filter = None if TRAIN_ON_ALL_DOSES else [TARGET_DOSE]
    lpips_model = lpips.LPIPS(net='alex').to(DEVICE)
    lpips_model.eval()
    for param in lpips_model.parameters():
        param.requires_grad = False
    # Split data
    splits = make_kfold_splits(
        all_exams, N_FOLDS, val_ratio=0.15, seed=42)

    #splits = splits[:1]  # <-- hpsearch only

    if USE_EXTERNAL_TEST:
        external_test_exams = sorted([d for d in EXTERNAL_TEST_DATASET_ROOT.iterdir() if d.is_dir()])
        print(f"External dataset: {len(external_test_exams)} exams total")

        if TRAIN_ON_EXTERNAL:
            external_exams_shuffled = external_test_exams.copy()
            random.Random(42).shuffle(external_exams_shuffled)  # fixed seed, reproducible split

            n_external_train = int(len(external_exams_shuffled) * EXTERNAL_TRAIN_RATIO)
            external_train_exams = external_exams_shuffled[:n_external_train]
            external_test_exams = external_exams_shuffled[n_external_train:]  # <-- reassigned to the held-out remainder

            print(f"  -> {len(external_train_exams)} exams into training (fixed across all folds)")
            print(f"  -> {len(external_test_exams)} exams held out as external test")
        else:
            external_train_exams = []

        external_test_dataset = SliceDataset(external_test_exams, target_size=AE_CONFIG["target_size"],
                                             dose_levels=[TARGET_DOSE])
        external_test_loader = DataLoader(external_test_dataset, batch_size=TRAIN_CONFIG["batch_size"],
                                          shuffle=False, num_workers=TRAIN_CONFIG["num_workers"],
                                          pin_memory=True)
    else:
        external_test_dataset = None
        external_test_loader = None
        external_train_exams = []

    all_fold_results = []
    all_gt_feats_pool = []
    all_gen_feats_pool = []
    all_external_gt_feats_pool = []
    all_external_gen_feats_pool = []

    for fold_idx, (train_exams, val_exams, test_exams) in enumerate(splits):

        # Load THIS FOLD's VAE -- trained on the exact same train/val split
        fold_ae_checkpoint_path = CHECKPOINT_DIR / f"kl_autoencoder_best_fold{fold_idx}.pth"
        if not fold_ae_checkpoint_path.exists():
            raise FileNotFoundError(
                f"No fold-specific VAE checkpoint found at {fold_ae_checkpoint_path}. "
                f"Train the 5-fold VAE first."
            )

        autoencoder = KLAutoencoder().to(DEVICE)
        ae_checkpoint = torch.load(fold_ae_checkpoint_path, map_location=DEVICE, weights_only=False)
        autoencoder.load_state_dict(ae_checkpoint["model_state_dict"])
        autoencoder.eval()
        for param in autoencoder.parameters():
            param.requires_grad = False

        print(f"[Fold {fold_idx}] Loaded VAE from epoch {ae_checkpoint['epoch'] + 1}, "
              f"val_loss={ae_checkpoint['val_loss']:.6f}")


        fold_run_name = f"{run_name}_fold{fold_idx}"
        fold_ckpt_best = CHECKPOINT_DIR / f"diffusion_best_{fold_run_name}.pth"
        fold_ckpt_final = CHECKPOINT_DIR / f"diffusion_final_{fold_run_name}.pth"

        print(f"\n{'=' * 50}\nFold {fold_idx + 1}/{N_FOLDS} -- {fold_run_name}\n{'=' * 50}")
        print(f"Train: {len(train_exams)} exams (+{len(external_train_exams)} external)" if TRAIN_ON_EXTERNAL else f"Train: {len(train_exams)} exams")
        print(f"Val: {len(val_exams)} exams")
        print(f"Test: {len(test_exams)} exams")

        # ============================================
        # Initialize models
        # ===========================================
        mri_encoder = Condition_Encoder(in_channels=1).to(DEVICE) if USE_MRI else None
        ldpet_encoder = Condition_Encoder(in_channels=1).to(DEVICE) if USE_LDPET else None
        unet = ConditionalUNet(latent_dim=32, time_dim=256, use_mri=USE_MRI, use_ldpet=USE_LDPET,
                               use_dose_cond=USE_DOSE_COND
                               ).to(DEVICE)


        print(f"U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")
        if mri_encoder is not None:
            print(f"MRI encoder parameters: {sum(p.numel() for p in mri_encoder.parameters()):,}")
        if ldpet_encoder is not None:
            print(f"LD-PET encoder parameters: {sum(p.numel() for p in ldpet_encoder.parameters()):,}")

        # ============================================
        # Setup noise scheduler
        # ============================================
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=DIFFUSION_CONFIG["num_train_timesteps"],
            beta_start=DIFFUSION_CONFIG["beta_start"],
            beta_end=DIFFUSION_CONFIG["beta_end"],
            beta_schedule=DIFFUSION_CONFIG["beta_schedule"],
            prediction_type=DIFFUSION_CONFIG["prediction_type"],
            clip_sample=False,
            clip_sample_range=3,
        )


        test_dataset = SliceDataset(test_exams, target_size=AE_CONFIG["target_size"], dose_levels = [TARGET_DOSE] )
        test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False,
                                 num_workers=TRAIN_CONFIG["num_workers"], pin_memory=True)

        if not SKIP_TRAINING:

            # Create PET datasets
            fold_train_exams = train_exams + external_train_exams  # same external patients added to every fold
            train_dataset = SliceDataset(fold_train_exams, target_size=AE_CONFIG["target_size"], transform=True,
                                         dose_levels=train_dose_filter)
            val_dataset = SliceDataset(val_exams, target_size=AE_CONFIG["target_size"], dose_levels=[TARGET_DOSE])

            # Dataloaders
            train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True,
                                             num_workers=TRAIN_CONFIG["num_workers"], pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False,
                                           num_workers=TRAIN_CONFIG["num_workers"], pin_memory=True)

            # Fixed validation examples
            viz_indices = build_fixed_viz_set(val_dataset, n_samples=2, seed=42)

            viz_batch = [val_dataset[i] for i in viz_indices]
            viz_pet = torch.stack([b[0] for b in viz_batch])
            viz_mri = torch.stack([b[1] for b in viz_batch])
            viz_ld = torch.stack([b[2] for b in viz_batch])
            viz_dose = torch.stack([b[3] for b in viz_batch])

            # Same initial noise, repeated across all 5 rows
            g = torch.Generator().manual_seed(42)
            single_noise = torch.randn((1, 32, 16, 16), generator=g)
            fixed_noise = single_noise.repeat(viz_pet.shape[0], 1, 1, 1).to(DEVICE) #this guarantees that the shape of the noise tensor matches the one from the images, because we have 5 dose levels


            # ============================================
            # Optimizer and scheduler
            # ============================================
            params = list(unet.parameters())

            if USE_MRI:
                params += list(mri_encoder.parameters())

            if USE_LDPET:
                params += list(ldpet_encoder.parameters())

            optimizer = torch.optim.AdamW(params, lr=TRAIN_CONFIG["learning_rate"])

            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=TRAIN_CONFIG["warmup_steps"],
                num_training_steps=len(train_loader) * TRAIN_CONFIG["num_epochs"],
            )

            # ============================================
            # Initialize wandb
            # ============================================
            wandb.init(project="PET-MRI-LDM-DOSE", name=fold_run_name, group=run_name, job_type =f"fold{fold_idx}", reinit = True, config={
                "diffusion_steps": DIFFUSION_CONFIG["num_train_timesteps"],
                "batch_size": TRAIN_CONFIG["batch_size"],
                "learning_rate": TRAIN_CONFIG["learning_rate"],
                "latent_dim": AE_CONFIG["latent_dim"],
                "use_mri": USE_MRI,
                "use_ldpet": USE_LDPET,
                "train_on_external": TRAIN_ON_EXTERNAL,
                "external_train_ratio": EXTERNAL_TRAIN_RATIO if TRAIN_ON_EXTERNAL else None,
                "use_dose_cond": USE_DOSE_COND, "train_on_all_doses": TRAIN_ON_ALL_DOSES,
                "target_dose": TARGET_DOSE
            })

            # ============================================
            # Training loop
            # ============================================
            best_loss = float("inf")
            early_stopping_patience = 20
            early_stopping_counter = 0

            latent_std = compute_latent_std(
                train_loader,
                autoencoder,
                DEVICE
            )
            for epoch in range(TRAIN_CONFIG["num_epochs"]):

                train_loss = train_one_epoch( unet, mri_encoder, ldpet_encoder, autoencoder, train_loader,
                    optimizer, noise_scheduler, lr_scheduler, latent_std,
                    DEVICE)

                val_loss = validate( unet, mri_encoder, ldpet_encoder, autoencoder, val_loader,
                    noise_scheduler, latent_std,
                    DEVICE)

                print(f"[Fold {fold_idx}] Epoch {epoch + 1:03d}/{TRAIN_CONFIG['num_epochs']} "
                      f"- Train: {train_loss:.6f} - Val: {val_loss:.6f}")

                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': lr_scheduler.get_last_lr()[0],
                })

                # Save best model
                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stopping_counter = 0

                    checkpoint_dict ={
                        "epoch": epoch,
                        "model_state_dict": unet.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "latent_std": latent_std,
                        "val_loss": val_loss,
                        "use_mri": USE_MRI,
                        "use_ldpet": USE_LDPET,
                        "use_dose_cond": USE_DOSE_COND,
                        "target_dose": TARGET_DOSE,
                        "fold":fold_idx
                    }


                    if USE_MRI:
                        checkpoint_dict["mri_encoder_state_dict"] = mri_encoder.state_dict()
                    if USE_LDPET:
                        checkpoint_dict["ldpet_encoder_state_dict"] = ldpet_encoder.state_dict()
                    torch.save(checkpoint_dict, fold_ckpt_best)
                    print(f"  -> Saved best model (val_loss: {best_loss:.6f})")
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
                # Visualize every N epochs

                if (epoch + 1) % TRAIN_CONFIG["log_every"] == 0:

                    _, periodic_metrics, _, _ = evaluate_metrics_on_dataset(
                        unet, mri_encoder, ldpet_encoder, autoencoder, val_dataset,
                        noise_scheduler, latent_std, DEVICE, lpips_model,
                        n_samples=METRIC_SAMPLE_SIZE,
                    )
                    wandb.log({f"val_metrics/{k}": v["mean"] for k, v in periodic_metrics.items()})

                    visualize_generation(unet, mri_encoder, ldpet_encoder, autoencoder, noise_scheduler,
                        viz_pet, viz_mri, viz_ld, viz_dose, fixed_noise, latent_std, DEVICE, epoch,
                    )


            checkpoint = torch.load(fold_ckpt_best, weights_only=False, map_location=DEVICE)
            unet.load_state_dict(checkpoint["model_state_dict"])
            if USE_MRI:
                mri_encoder.load_state_dict(checkpoint["mri_encoder_state_dict"])
            if USE_LDPET:
                ldpet_encoder.load_state_dict(checkpoint["ldpet_encoder_state_dict"])
            latent_std = checkpoint["latent_std"]

        else:
            print(f"SKIP_TRAINING=True -- loading existing checkpoint for fold {fold_idx}, skipping fit")

            if not fold_ckpt_best.exists():
                raise FileNotFoundError(
                    f"SKIP_TRAINING=True but no checkpoint found at {fold_ckpt_best}. "
                    f"Train this fold first, or set SKIP_TRAINING=False."
                )

            checkpoint = torch.load(fold_ckpt_best, weights_only=False, map_location=DEVICE)
            unet.load_state_dict(checkpoint["model_state_dict"])

            if USE_MRI:
                mri_encoder.load_state_dict(checkpoint["mri_encoder_state_dict"])

            if USE_LDPET:
                ldpet_encoder.load_state_dict(checkpoint["ldpet_encoder_state_dict"])
            latent_std = checkpoint["latent_std"]
            best_loss = checkpoint.get("val_loss", float("nan"))

            wandb.init(project="PET-MRI-LDM-DOSE", name=f"{fold_run_name}_evalonly", group=run_name,
                       job_type=f"fold{fold_idx}_eval", reinit=True, config={
                    "fold": fold_idx, "use_mri": USE_MRI, "use_ldpet": USE_LDPET,
                    "use_dose_cond": USE_DOSE_COND, "train_on_all_doses": TRAIN_ON_ALL_DOSES,
                    "target_dose": TARGET_DOSE, "skip_training": True,
                })

        print(f"  Loaded from epoch {checkpoint['epoch'] + 1} with val_loss: {checkpoint['val_loss']:.6f}")

        if RUN_TEST_EVAL:

            test_loss = validate( unet, mri_encoder, ldpet_encoder, autoencoder,
                test_loader, noise_scheduler, latent_std, DEVICE,
            )
            print(f"[Fold {fold_idx}] Test loss (noise-pred MSE): {test_loss:.6f}")

            test_raw_metrics, test_agg_metrics, _, test_fid_features = evaluate_metrics_on_dataset(
                unet, mri_encoder, ldpet_encoder, autoencoder, test_dataset,
                noise_scheduler, latent_std, DEVICE, lpips_model,
                n_samples=TEST_METRICS_MAX_SAMPLES, seed=999,
                collect_images=False, collect_fid_features=True,
            )
            test_gt_feats, test_gen_feats = test_fid_features
            all_gt_feats_pool.extend(test_gt_feats)
            all_gen_feats_pool.extend(test_gen_feats)

            if USE_EXTERNAL_TEST:

                external_test_loss = validate(unet, mri_encoder, ldpet_encoder, autoencoder,
                                              external_test_loader, noise_scheduler, latent_std, DEVICE)
                print(f"[Fold {fold_idx}] EXTERNAL test loss (noise-pred MSE): {external_test_loss:.6f}")

                external_raw_metrics, external_agg_metrics, _, external_fid_features = evaluate_metrics_on_dataset(
                    unet, mri_encoder, ldpet_encoder, autoencoder, external_test_dataset,
                    noise_scheduler, latent_std, DEVICE, lpips_model,
                    n_samples=TEST_METRICS_MAX_SAMPLES, seed=999,
                    collect_images=False, collect_fid_features=True,
                )
                external_gt_feats, external_gen_feats = external_fid_features
                all_external_gen_feats_pool.extend(external_gen_feats)
                if fold_idx == 0:
                    all_external_gt_feats_pool.extend(external_gt_feats)

                print(f"[Fold {fold_idx}] EXTERNAL test metrics:")
                for k, v in external_agg_metrics.items():
                     print(f"    {k}: {v['mean']:.4f} +/- {v['std']:.4f}")

                wandb.log({f"external_test_metrics/{k}": v["mean"] for k, v in external_agg_metrics.items()})
            else:
                external_test_loss = None
                external_agg_metrics = {}
                external_raw_metrics = []

            print(f"[Fold {fold_idx}] Test metrics:")
            for k, v in test_agg_metrics.items():
                print(f"    {k}: {v['mean']:.4f} +/- {v['std']:.4f}")

            wandb.log({"final_test_loss": test_loss})
            wandb.log({f"test_metrics/{k}": v["mean"] for k, v in test_agg_metrics.items()})

            torch.save({
                "model_state_dict": unet.state_dict(),
                "mri_encoder_state_dict": mri_encoder.state_dict() if USE_MRI else None,
                "ldpet_encoder_state_dict": ldpet_encoder.state_dict() if USE_LDPET else None,
                "latent_std": latent_std,
                "test_loss": test_loss,
                "test_metrics": test_agg_metrics,
                "fold": fold_idx,
                "config": {"use_mri": USE_MRI, "use_ldpet": USE_LDPET,
                           "use_dose_cond": USE_DOSE_COND, "train_on_all_doses": TRAIN_ON_ALL_DOSES,
                           "target_dose": TARGET_DOSE},
            }, fold_ckpt_final)


            all_fold_results.append({"fold": fold_idx, "best_val_loss": best_loss, "test_loss": test_loss,
                                     "test_metrics": test_agg_metrics, "test_raw_metrics": test_raw_metrics,
                                     "external_test_loss": external_test_loss,
                                     "external_test_metrics": external_agg_metrics,
                                     "external_test_raw_metrics": external_raw_metrics
                                     })
        wandb.finish()
        del unet, mri_encoder, ldpet_encoder, autoencoder
        del test_dataset, test_loader

        if not SKIP_TRAINING:
            del optimizer, lr_scheduler
            del train_dataset, val_dataset, train_loader, val_loader

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ============================================================
    # Aggregate across all folds
    # ============================================================
    if RUN_TEST_EVAL:
        test_losses = [r["test_loss"] for r in all_fold_results]
        mean_test = sum(test_losses) / len(test_losses)
        std_test = (sum((x - mean_test) ** 2 for x in test_losses) / len(test_losses)) ** 0.5

        all_raw_metrics = [m for r in all_fold_results for m in r["test_raw_metrics"]]
        overall_metrics = aggregate(all_raw_metrics)
        fid_score = compute_fid_from_features(all_gt_feats_pool, all_gen_feats_pool)
        print(f"\n{'=' * 50}\nInternal test summary ({run_name})\n{'=' * 50}")
        print(f"  Mean test loss: {mean_test:.6f} +/- {std_test:.6f}")
        print(f"  FID (n={len(all_gt_feats_pool)}): {fid_score:.2f}")
        for k, v in overall_metrics.items():
            print(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}")


        if USE_EXTERNAL_TEST:
            external_test_losses = [r["external_test_loss"] for r in all_fold_results]
            external_mean_test = sum(external_test_losses) / len(external_test_losses)
            external_std_test = (sum((x - external_mean_test) ** 2 for x in external_test_losses)
                                 / len(external_test_losses)) ** 0.5

            all_external_raw_metrics = [m for r in all_fold_results for m in r["external_test_raw_metrics"]]
            external_overall_metrics = aggregate(all_external_raw_metrics)
            external_fid_score = compute_fid_from_features(all_external_gt_feats_pool, all_external_gen_feats_pool)

            print(f"\n{'=' * 50}\nINTERNAL vs EXTERNAL comparison ({run_name})\n{'=' * 50}")
            print(
                f"  Internal -- mean test loss: {mean_test:.6f} +/- {std_test:.6f}, FID(n={len(all_gt_feats_pool)}): {fid_score:.2f}")
            print(
                f"  External -- mean test loss: {external_mean_test:.6f} +/- {external_std_test:.6f}, FID(n={len(all_external_gt_feats_pool)}): {external_fid_score:.2f}")
            print(f"\n  Metric        Internal              External")
            for k in overall_metrics:
                print(f"  {k:<14} {overall_metrics[k]['mean']:.4f}+/-{overall_metrics[k]['std']:.4f}    "
                      f"{external_overall_metrics[k]['mean']:.4f}+/-{external_overall_metrics[k]['std']:.4f}")
        else:
            print(f"\nUSE_EXTERNAL_TEST=False -- skipped external evaluation. Internal-only summary printed above.")

    else:
        print(f"\nRUN_TEST_EVAL=False -- all {N_FOLDS} folds trained and checkpointed. "
              f"Rerun with SKIP_TRAINING=True, RUN_TEST_EVAL=True to evaluate.")