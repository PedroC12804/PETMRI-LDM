"""
Train KL-Regularized Autoencoder for Latent Diffusion
Produces clean latents with N(0,1) distribution
"""

import torch
import torch.nn.functional as F
from mpmath.identification import transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import random
import wandb
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from torch.amp import GradScaler
#from monai.losses import PerceptualLoss
import argparse
from models.PET_KL_Autoencoder import KLAutoencoder
from dataset_black import PETSliceDataset
import torchvision.transforms.v2 as torchvision

# Paths
CHECKPOINT_DIR = Path("/home/pedrocarreiro/Desktop/Latent_Diffusion_Model/checkpoints/")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "kl_autoencoder_best.pth"
FINAL_MODEL_PATH = CHECKPOINT_DIR / "kl_autoencoder_final.pth"

# Create the parser
parser = argparse.ArgumentParser(description='Train KL-Regularized Autoencoder for Latent Diffusion')

# Add arguments
parser.add_argument('--mode', type=bool, help='Train/Test')

args = parser.parse_args()

# ============================================
# CONFIGURATION
# ============================================
wandb.init(project='KL-PET-Autoencoder')

BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
TARGET_SIZE = (256, 256)

# Loss weights - CRITICAL: KL weight must be small
W_RECON = 1.0
#W_PERCEPTUAL = 0.002
W_KL = 0.00035

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
    #percept_loss = perceptual_loss(recon, target)
    kl_loss_val = kl_loss(mu, log_var)
    
    total = W_RECON * recon_loss  + W_KL * kl_loss_val
    return total, recon_loss, kl_loss_val


# ============================================
# VIEWING FUNCTIONS
# ============================================
def log_reconstructions(model, loader, device, epoch, num_samples=4):
    """Log reconstructed images to wandb with latent statistics"""
    model.eval()
    
    batch = next(iter(loader))
    batch = batch[:num_samples].to(device)
    
    with torch.no_grad():
        recon, mu, log_var, z = model(batch)
    
    # Log latent statistics
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
    #total_percep = 0
    total_kl = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        #with torch.amp.autocast('cuda'):
        recon, mu, log_var, _ = model(batch)
        loss, recon_loss, kl_loss_val = compute_loss(recon, batch, mu, log_var)

        loss.backward()
        optimizer.step()

        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        #total_percep += percep_loss.item()
        total_kl += kl_loss_val.item()

    n_batches = len(loader)
    return (total_loss/n_batches, total_recon/n_batches, total_kl/n_batches)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    total_recon = 0
    #total_percep = 0
    total_kl = 0
    ssim_scores = []

    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = batch.to(device)

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
                score = ssim(batch_np[i, 0], recon_np[i, 0], data_range=1.0)
                ssim_scores.append(score)
            except:
                continue

    n_batches = len(loader)
    return (total_loss/n_batches, total_recon/n_batches, total_kl/n_batches,
            np.mean(ssim_scores) if ssim_scores else 0.0)


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find all exam directories
    root = Path("/home/pedrocarreiro/Desktop/Dataset_Normalized3/")
    all_exams = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"Found {len(all_exams)} exam folders")
    
    # Split data
    train_exams, val_exams, test_exams = train_val_test_split(
        all_exams, test_ratio=0.15, val_ratio=0.15, seed=42
    )
    #if args.mode == "train":

    print(f"Train: {len(train_exams)} exams")
    print(f"Val: {len(val_exams)} exams")

    # Save split
    split = {
            "train": [d.name for d in train_exams],
            "val":   [d.name for d in val_exams],
            "test":  [d.name for d in test_exams],
    }
    with open("dataset_split.json", "w") as f:
        json.dump(split, f, indent=2)
    
    # Create datasets
    train_dataset = PETSliceDataset(train_exams, target_size=TARGET_SIZE)
    val_dataset = PETSliceDataset(val_exams, target_size=TARGET_SIZE)


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

        # Initialize model
    model = KLAutoencoder().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #scaler = GradScaler('cuda')

        # Training loop
    best_val_loss = float("inf")
    best_ssim = 0.0

    print("\nStarting training...")
    print("=" * 50)
    early_stopping = EarlyStopping(epochs=10, min_delta=0.0001)

    for epoch in range(NUM_EPOCHS):
            train_loss, train_recon, train_kl = train_one_epoch(
                model, train_loader, optimizer, device
            )

            val_loss, val_recon, val_kl, val_ssim = validate(
                model, val_loader, device
            )

            log_reconstructions(model, val_loader, device, epoch, num_samples=4)

            print(f"\nEpoch {epoch+1:03d}/{NUM_EPOCHS}")
            print(f"  Train - Loss: {train_loss:.4f} | Recon: {train_recon:.4f} | KL: {train_kl:.6f}")
            print(f"  Val   - Loss: {val_loss:.4f} | Recon: {val_recon:.4f} | KL: {val_kl:.6f} | SSIM: {val_ssim:.4f}")

            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_recon': train_recon,
                'train_kl': train_kl,
                'val_loss': val_loss,
                'val_recon': val_recon,
                'val_kl': val_kl,
                'val_ssim': val_ssim,
            })

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ssim = val_ssim
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_ssim": val_ssim,
                    "val_kl": val_kl,
                }, BEST_MODEL_PATH)
                print(f"  -> Saved best model")

            if early_stopping(val_loss):
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

#if args.mode == "test":

    print(f"Test: {len(test_exams)} exams")
    test_dataset = PETSliceDataset(test_exams, target_size=TARGET_SIZE)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    if Path(BEST_MODEL_PATH).exists():
        print("Loading best model checkpoint...")
        checkpoint = torch.load(BEST_MODEL_PATH, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        print(f"  Loaded from epoch {checkpoint['epoch']+1} with SSIM: {checkpoint['val_ssim']:.4f}")
    else:
        print("No saved best model found. Using current model state.")

    test_loss, test_recon, test_kl, test_ssim = validate(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Recon: {test_recon:.4f}")
    #print(f"  Percep: {test_percep:.4f}")
    print(f"  KL: {test_kl:.6f}")
    print(f"  SSIM: {test_ssim:.4f}")
    
    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "latent_dim": 32,
            "target_size": TARGET_SIZE,
        },
        "test_metrics": {
            "loss": test_loss,
            "ssim": test_ssim,
            "kl": test_kl,
        }
    }, FINAL_MODEL_PATH)
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f} (SSIM: {best_ssim:.4f})")
    print(f"Final test SSIM: {test_ssim:.4f}")