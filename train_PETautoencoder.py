"""
Train KL-Regularized Autoencoder for Latent Diffusion
Produces clean latents with N(0,1) distribution
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import random
import wandb
import lpips
from tqdm import tqdm
from sklearn.model_selection import KFold
import gc


from metrics import compute_all_metrics, aggregate
from models.PET_KL_Autoencoder import KLAutoencoder
from dataset_MRI import SliceDataset


# Paths
CHECKPOINT_DIR = Path("/home/pedrocarreiro/Desktop/Latent_Diffusion_Model/checkpoints/")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "kl_autoencoder_best.pth"
FINAL_MODEL_PATH = CHECKPOINT_DIR / "kl_autoencoder_final.pth"

SKIP_TRAINING = False

# ============================================
# CONFIGURATION
# ============================================


BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
TARGET_SIZE = (256, 256)
TARGET_DOSE = 1   # just avoids duplication

# Loss weights - CRITICAL: KL weight must be small
W_RECON = 1.0
W_KL = 0.00035

N_FOLDS = 5
VAL_RATIO = 0.15       # <-- must match the diffusion script exactly
SPLIT_SEED = 42        # <-- must match the diffusion script exactly


# ============================================
# EARLY STOPPING
# ============================================
class EarlyStopping:
    def __init__(self, epochs=10, min_delta=0.0001):
        self.epochs = epochs
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.epochs:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop



def make_kfold_splits(exams, n_folds=5, val_ratio=0.15, seed=42):
    """
    MUST produce identical folds to the diffusion training script --
    same KFold call, same seed, same val_ratio, same underlying
    dataset root -- or fold i's VAE and fold i's diffusion model won't
    actually correspond to the same train/test patients.
    """
    exams = list(exams)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    splits = []
    for remaining_idx, test_idx in kf.split(exams):
        remaining = [exams[i] for i in remaining_idx]
        test_exams = [exams[i] for i in test_idx]

        random.Random(seed).shuffle(remaining)
        n_val = int(len(remaining) * val_ratio)
        val_exams = remaining[:n_val]
        train_exams = remaining[n_val:]

        splits.append((train_exams, val_exams, test_exams))

    return splits


# ============================================
# LOSS FUNCTIONS
# ============================================
def kl_loss(mu, log_var):
    """KL divergence between N(mu, exp(log_var)) and N(0, 1)"""
    # Standard formula: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


def compute_loss(recon, target, mu, log_var):
    """Combined loss: Reconstruction + Perceptual + KL"""
    recon_loss = F.l1_loss(recon, target)
    kl_loss_val = kl_loss(mu, log_var)
    
    total = W_RECON * recon_loss  + W_KL * kl_loss_val
    return total, recon_loss, kl_loss_val


# ============================================
# VIEWING FUNCTIONS
# ============================================
def log_reconstructions(model, loader, device, epoch, num_samples=4):
    """Log reconstructed images to wandb with latent statistics"""
    model.eval()

    pet, _, _, _ = next(iter(loader))
    batch = pet[:num_samples].to(device)

    with torch.no_grad():
        recon, mu, log_var, z = model(batch)

    wandb.log({
        'latent_mean': mu.mean().item(),
        'latent_std': mu.std().item(),
        'latent_min': mu.min().item(),
        'latent_max': mu.max().item(),
        'kl_loss': kl_loss(mu, log_var).item(),
        'epoch': epoch,
    })

    for i in range(num_samples):
        original = batch[i, 0].cpu().numpy()
        reconstructed = recon[i, 0].cpu().numpy()

        wandb.log({
            f'original_{i}': wandb.Image(original, caption=f'Original {i}'),
            f'reconstructed_{i}': wandb.Image(reconstructed, caption=f'Recon {i}'),
        })

        comparison = np.concatenate([original, reconstructed], axis=1)
        wandb.log({f'comparison_{i}': wandb.Image(comparison, caption=f'Original vs Recon')})


# ============================================
# TRAINING FUNCTIONS
# ============================================
def train_val_test_split(exams, test_ratio=0.15, val_ratio=0.15, seed=42):
    random.seed(seed)
    exams_shuffled = exams.copy()
    random.shuffle(exams_shuffled)
    
    n_total = len(exams_shuffled)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    
    test_exams = exams_shuffled[:n_test]
    val_exams = exams_shuffled[n_test:n_test+n_val]
    train_exams = exams_shuffled[n_test+n_val:]
    
    return train_exams, val_exams, test_exams


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    for pet, _, _, _ in tqdm(loader, desc="Training", leave=False):
        batch = pet.to(device)
        optimizer.zero_grad()

        recon, mu, log_var, _ = model(batch)
        loss, recon_loss, kl_loss_val = compute_loss(recon, batch, mu, log_var)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss_val.item()

    n_batches = len(loader)
    return (total_loss / n_batches, total_recon / n_batches, total_kl / n_batches)


@torch.no_grad()
def validate(model, loader, device, lpips_model = None ):
    model.eval()
    total_loss = 0
    total_recon = 0
    #total_percep = 0
    total_kl = 0
    per_sample_metrics = []

    for pet, _, _, _ in tqdm(loader, desc="Validation", leave=False):
        batch = pet.to(device)

        with torch.amp.autocast('cuda'):
            recon, mu, log_var, _ = model(batch)

            if torch.isnan(recon).any() or torch.isinf(recon).any():
                print("NaN/Inf in decoder output")
                print("mu:", mu.abs().max(), "log_var:", log_var.max())
                breakpoint()

            loss, recon_loss, kl_loss_val = compute_loss(recon, batch, mu, log_var)

        total_loss += loss.item()
        total_recon += recon_loss.item()
        #total_percep += percep_loss.item()
        total_kl += kl_loss_val.item()

        # SSIM for first 8 images
        recon_np = recon[:8].cpu().float().numpy()
        batch_np = batch[:8].cpu().float().numpy()
        
        for i in range(recon_np.shape[0]):
            try:
                m=compute_all_metrics(recon_np[i,0], batch_np[i,0], lpips_model= lpips_model, device=device)
                per_sample_metrics.append(m)
            except:
                continue

    n_batches = len(loader)
    metrics_agg = aggregate(per_sample_metrics)
    return (total_loss/n_batches, total_recon/n_batches, total_kl/n_batches,
            metrics_agg)

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = Path("/home/pedrocarreiro/Desktop/Dataset_Normalized3/")
    all_exams = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"Found {len(all_exams)} exam folders")

    splits = make_kfold_splits(all_exams, N_FOLDS, val_ratio=VAL_RATIO, seed=SPLIT_SEED)

    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    for p in lpips_model.parameters():
        p.requires_grad = False

    all_fold_results = []

    for fold_idx, (train_exams, val_exams, test_exams) in enumerate(splits):


        fold_best_path = CHECKPOINT_DIR / f"kl_autoencoder_best_fold{fold_idx}.pth"
        fold_final_path = CHECKPOINT_DIR / f"kl_autoencoder_final_fold{fold_idx}.pth"

        print(f"\n{'=' * 50}\nFold {fold_idx + 1}/{N_FOLDS}\n{'=' * 50}")
        print(f"Train: {len(train_exams)} exams")
        print(f"Val: {len(val_exams)} exams")
        print(f"Test: {len(test_exams)} exams")

        # Fresh model + optimizer every fold -- no weight carryover
        model = KLAutoencoder().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_loss = float("inf")
        best_ssim = 0.0

        if not SKIP_TRAINING:
            train_dataset = SliceDataset(train_exams, target_size=TARGET_SIZE, transform=True,
                                         dose_levels=[TARGET_DOSE])
            val_dataset = SliceDataset(val_exams, target_size=TARGET_SIZE, dose_levels=[TARGET_DOSE])


            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                       num_workers=NUM_WORKERS, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=NUM_WORKERS, pin_memory=True)

            print(f"Train batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}")

            wandb.init(project='KL-PET-Autoencoder', name=f"fold{fold_idx}",
                       group="kl_autoencoder_5fold", job_type=f"fold{fold_idx}", reinit=True)

            print("\nStarting training...")
            early_stopping = EarlyStopping(epochs=10, min_delta=0.0001)

            for epoch in range(NUM_EPOCHS):
                train_loss, train_recon, train_kl = train_one_epoch(
                    model, train_loader, optimizer, device
                )
                val_loss, val_recon, val_kl, val_metrics = validate(
                    model, val_loader, lpips_model=lpips_model, device=device
                )
                val_ssim = val_metrics["ssim"]["mean"]

                log_reconstructions(model, val_loader, device, epoch, num_samples=4)

                print(f"\n[Fold {fold_idx}] Epoch {epoch+1:03d}/{NUM_EPOCHS}")
                print(f"  Train - Loss: {train_loss:.4f} | Recon: {train_recon:.4f} | KL: {train_kl:.6f}")
                print(f"  Val   - Loss: {val_loss:.4f} | Recon: {val_recon:.4f} | KL: {val_kl:.6f} | SSIM: {val_ssim:.4f}")

                wandb.log({
                    'epoch': epoch, 'train_loss': train_loss, 'train_recon': train_recon,
                    'train_kl': train_kl, 'val_loss': val_loss, 'val_recon': val_recon,
                    'val_kl': val_kl, 'val_ssim': val_ssim,
                    'val_psnr': val_metrics['psnr']['mean'],
                    'val_masked_mse': val_metrics['masked_mse']['mean'],
                    'val_lpips': val_metrics['lpips']['mean'],
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ssim = val_ssim
                    torch.save({
                        "epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss, "val_ssim": val_ssim, "val_kl": val_kl,
                        "fold": fold_idx,
                    }, fold_best_path)
                    print(f"  -> Saved best model (fold {fold_idx})")

                if early_stopping(val_loss):
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        else:
            print(f"SKIP_TRAINING=True -- loading existing checkpoint for fold {fold_idx}")
            if not fold_best_path.exists():
                raise FileNotFoundError(
                    f"SKIP_TRAINING=True but no checkpoint found at {fold_best_path}. "
                    f"Train this fold first, or set SKIP_TRAINING=False.")

            wandb.init(project='KL-PET-Autoencoder', name=f"fold{fold_idx}_evalonly",
                       group="kl_autoencoder_5fold", job_type=f"fold{fold_idx}_eval", reinit=True)

        # ---- Final test evaluation for this fold (always runs) ----
        test_dataset = SliceDataset(test_exams, target_size=TARGET_SIZE, dose_levels=[TARGET_DOSE])
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True)

        checkpoint = torch.load(fold_best_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_val_loss = checkpoint.get("val_loss", best_val_loss)
        best_ssim = checkpoint.get("val_ssim", best_ssim)
        print(f"  Loaded fold {fold_idx} best checkpoint from epoch {checkpoint['epoch']+1} "
              f"with SSIM: {checkpoint['val_ssim']:.4f}")

        test_loss, test_recon, test_kl, test_metrics = validate(
            model, test_loader, device, lpips_model=lpips_model
        )
        test_ssim = test_metrics["ssim"]["mean"]

        print(f"[Fold {fold_idx}] Test Results:")
        print(f"  Loss: {test_loss:.4f} | Recon: {test_recon:.4f} | KL: {test_kl:.6f}")
        for k, v in test_metrics.items():
            print(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}")

        wandb.log({"test_loss": test_loss, "test_ssim": test_ssim,
                    **{f"test_{k}": v["mean"] for k, v in test_metrics.items()}})

        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {"latent_dim": 32, "target_size": TARGET_SIZE, "fold": fold_idx},
            "test_metrics": {"loss": test_loss, "ssim": test_ssim, "kl": test_kl,
                              **{k: v["mean"] for k, v in test_metrics.items()}},
        }, fold_final_path)

        wandb.finish()

        all_fold_results.append({
            "fold": fold_idx, "best_val_loss": best_val_loss, "best_ssim": best_ssim,
            "test_loss": test_loss, "test_metrics": test_metrics,
        })

        # ---- Cleanup between folds ----
        del model, optimizer, test_dataset, test_loader
        if not SKIP_TRAINING:
            del train_dataset, val_dataset, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ============================================================
    # Aggregate across all 5 folds
    # ============================================================
    print(f"\n{'=' * 50}\n5-Fold VAE Summary\n{'=' * 50}")
    test_losses = [r["test_loss"] for r in all_fold_results]
    mean_test = sum(test_losses) / len(test_losses)
    std_test = (sum((x - mean_test) ** 2 for x in test_losses) / len(test_losses)) ** 0.5
    print(f"Mean test loss: {mean_test:.4f} +/- {std_test:.4f}")

    for key in all_fold_results[0]["test_metrics"]:
        vals = [r["test_metrics"][key]["mean"] for r in all_fold_results]
        mean_v = sum(vals) / len(vals)
        std_v = (sum((x - mean_v) ** 2 for x in vals) / len(vals)) ** 0.5
        print(f"  {key}: {mean_v:.4f} +/- {std_v:.4f}")