
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb
from pathlib import Path
import matplotlib.pyplot as plt
import random
import torchvision.transforms.v2 as torchvision

from dataset_black import PETSliceDataset
from models.PET_KL_Autoencoder import KLAutoencoder
from models.diffusion_model import create_diffusion_unet


# ---------------------- CONFIG --------------------
PROJECT_ROOT = Path("/home/pedrocarreiro/Desktop/Latent_Diffusion_Model/")
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
AE_CHECKPOINT_PATH = CHECKPOINT_DIR / "kl_autoencoder_best.pth"
DIFFUSION_BEST_PATH = CHECKPOINT_DIR / "diffusion_best.pth"
DIFFUSION_FINAL_PATH = CHECKPOINT_DIR / "diffusion_final.pth"

DATASET_ROOT = Path("/home/pedrocarreiro/Desktop/Dataset_Normalized3/")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# DIFFUSION CONFIG
# ============================================
DIFFUSION_CONFIG = {
    "num_train_timesteps": 500,
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
    "learning_rate": 1e-4,
    "num_epochs": 350,
    "num_workers": 4,
    "log_every": 2,
    "save_every": 20,
    "warmup_steps": 500,
}

# ============================================
# HELPER FUNCTIONS
# ============================================
class EMA:
    """Exponential Moving Average for model weights"""

    def __init__(self, beta=0.9999):
        self.beta = beta
        self.ema_weights = None
        self.step = 0

    def update(self, model):
        """Update EMA weights with current model weights"""
        self.step += 1

        if self.ema_weights is None:
            # First update: copy all weights
            self.ema_weights = {}
            for name, param in model.state_dict().items():
                self.ema_weights[name] = param.clone().detach()
        else:
            # Exponential moving average update
            with torch.no_grad():
                for name, param in model.state_dict().items():
                    if name in self.ema_weights:
                        self.ema_weights[name] = (
                                self.beta * self.ema_weights[name] +
                                (1 - self.beta) * param.detach()
                        )

    def apply_to(self, model):
        """Replace model weights with EMA weights"""
        old_weights = {}
        for name, param in model.state_dict().items():
            if name in self.ema_weights:
                old_weights[name] = param.clone().detach()
                param.data.copy_(self.ema_weights[name])
        return old_weights

    def restore_from(self, model, old_weights):
        """Restore original model weights"""
        for name, param in model.state_dict().items():
            if name in old_weights:
                param.data.copy_(old_weights[name])

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


def encode_dataset(loader, encoder, device, seed =42):
    """
    Pre-encode all PET slices to latents.
    Returns raw latents (no scaling, no normalization).
    """
    latents = []
    print("Pre-encoding dataset to latents...")
    
    for batch in tqdm(loader, desc="Encoding"):
        batch = batch.to(device)
        with torch.no_grad():
            mu, logvar = encoder(batch)
            z = mu
        latents.append(z.cpu())
    
    latents = torch.cat(latents, dim=0)
    
    print(f"  Latents - mean: {latents.mean():.4f}, std: {latents.std():.4f}")
    print(f"  Range: [{latents.min():.4f}, {latents.max():.4f}]")
    
    return latents

def train_one_epoch(model, loader, optimizer, noise_scheduler, lr_scheduler, device):
    model.train()
    total_loss = 0
        
    for batch in tqdm(loader, desc="Training", leave=False):
        # Get raw latents
        latents = batch[0].to(device)
        
        # Standard noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device
        ).long()
        
        # Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        noise_pred = model(noisy_latents, timesteps).sample
        
        # Loss on noise prediction
        loss = F.mse_loss(noise_pred, noise)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, noise_scheduler, device):
    model.eval()
    total_loss = 0
        
    for batch in tqdm(loader, desc="Validation", leave=False):
        latents = batch[0].to(device)
    
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device
        ).long()
        
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        noise_pred = model(noisy_latents, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def visualize_generation(model, decoder, scheduler, device, epoch, num_samples=4 ):
    """Generate and visualize PET samples from random noise"""
    model.eval()
    
    # Start from standard Gaussian noise (mean=0, std=1)
    latent_shape = (num_samples, 32, 16, 16)
    latents = torch.randn(latent_shape).to(device)
    
    # Denoise step by step
    scheduler.set_timesteps(500)

    with torch.no_grad(): #CHANGED THISSSSS TIMESTEP AS A TENSOR
        for t in tqdm(scheduler.timesteps, desc="Sampling", leave=False):
            t_batch = torch.full((latents.shape[0],), t.item(), device=latents.device, dtype=torch.long)
            noise_pred = model(latents, t_batch).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode directly
    with torch.no_grad():
        print(latents.mean(), latents.std(), latents.min(), latents.max())

        generated_pets = decoder(latents)
    
    # Plot results
    fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        img = generated_pets[i, 0].cpu().numpy()
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f"Generated {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"generated_epoch_{epoch+1}.png", dpi=150)
    plt.close(fig) 
    
    wandb.log({f"generated_samples": wandb.Image(f"generated_epoch_{epoch+1}.png")})


# ============================================
# MAIN TRAINING
# ============================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # ============================================
    # 1. Load trained KL-VAE autoencoder
    # ============================================
    print("\n" + "="*50)
    print("Loading trained KL-VAE autoencoder...")
    print("="*50)
    
    autoencoder = KLAutoencoder().to(DEVICE)
    checkpoint = torch.load(AE_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.eval()
    
    print(f"Autoencoder loaded from epoch {checkpoint['epoch']+1}")
    print(f"  Val loss: {checkpoint['val_loss']:.6f}")
    print(f"  Val SSIM: {checkpoint['val_ssim']:.4f}")
    print(f"  Val KL: {checkpoint['val_kl']:.6f}")
    
    # ============================================
    # 2. Create dataset and dataloaders
    # ============================================
    print("\n" + "="*50)
    print("Loading dataset...")
    print("="*50)
    
    all_exams = sorted([d for d in DATASET_ROOT.iterdir() if d.is_dir()])
    
    #all_exams=all_exams[:100] #!!!!!!!!!!!!!!!!!!!!!!!!!

    # Split data
    train_exams, val_exams, test_exams = train_val_test_split(
        all_exams, test_ratio=0.15, val_ratio=0.15, seed=42
    )
    
    print(f"Train: {len(train_exams)} exams")
    print(f"Val: {len(val_exams)} exams")
    print(f"Test: {len(test_exams)} exams")

    mytransform = torchvision.Compose([
        torchvision.RandomHorizontalFlip(p=0.5),
        torchvision.RandomAffine(degrees=5, translate=(0.02, 0.02)),
    ])
    
    # Create PET datasets
    train_dataset = PETSliceDataset(train_exams, target_size=AE_CONFIG["target_size"])
    val_dataset = PETSliceDataset(val_exams, target_size=AE_CONFIG["target_size"])
    test_dataset = PETSliceDataset(test_exams, target_size=AE_CONFIG["target_size"])
    
    # Dataloaders for encoding (small batch size for encoding)
    encode_batch_size = 4
    train_encode_loader = DataLoader(train_dataset, batch_size=encode_batch_size, shuffle=False,
                                     num_workers=TRAIN_CONFIG["num_workers"], pin_memory=True)
    val_encode_loader = DataLoader(val_dataset, batch_size=encode_batch_size, shuffle=False,
                                   num_workers=TRAIN_CONFIG["num_workers"], pin_memory=True)
    test_encode_loader = DataLoader(test_dataset, batch_size=encode_batch_size, shuffle=False,
                                    num_workers=TRAIN_CONFIG["num_workers"], pin_memory=True)
    
    # ============================================
    # 3. Pre-encode all PET slices to latents (raw, no normalization)
    # ============================================
    print("\n" + "="*50)
    print("Pre-encoding dataset to latent space...")
    print("="*50)
    
    train_latents = encode_dataset(train_encode_loader, autoencoder.encoder, DEVICE)
    print(f"Train latents - Mean: {train_latents.mean():.4f}, Std: {train_latents.std():.4f}")
    std = train_latents.std()
    val_latents = encode_dataset(val_encode_loader, autoencoder.encoder, DEVICE)
    print(f"Val latents - Mean: {val_latents.mean():.4f}, Std: {val_latents.std():.4f}")
    test_latents = encode_dataset(test_encode_loader, autoencoder.encoder, DEVICE)
    print(f"Test latents - Mean: {test_latents.mean():.4f}, Std: {test_latents.std():.4f}")

    train_latents = train_latents / std
    print(f"Train latents - Mean: {train_latents.mean():.4f}, Std: {train_latents.std():.4f}")
    val_latents = val_latents / std
    test_latents = test_latents / std

    # Create TensorDatasets
    train_latent_dataset = torch.utils.data.TensorDataset(train_latents)
    val_latent_dataset = torch.utils.data.TensorDataset(val_latents)
    test_latent_dataset = torch.utils.data.TensorDataset(test_latents)
    
    train_loader = DataLoader(train_latent_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True,
                              num_workers=TRAIN_CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_latent_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False,
                            num_workers=TRAIN_CONFIG["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_latent_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False,
                             num_workers=TRAIN_CONFIG["num_workers"], pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # ============================================
    # 4. Initialize diffusion model
    # ============================================
    print("\n" + "="*50)
    print("Initializing diffusion model...")
    print("="*50)
    
    unet = create_diffusion_unet(
        latent_dim=AE_CONFIG["latent_dim"],
        latent_size=32,
        block_out_channels=(256,512,512),
    ).to(DEVICE)

    #ema = EMA(beta=0.9999)
    
    print(f"U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # ============================================
    # 5. Setup noise scheduler
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
    
    # ============================================
    # 6. Optimizer and scheduler
    # ============================================
    optimizer = torch.optim.AdamW(unet.parameters(), lr=TRAIN_CONFIG["learning_rate"])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=TRAIN_CONFIG["warmup_steps"],
        num_training_steps=len(train_loader) * TRAIN_CONFIG["num_epochs"],
    )
    
    # ============================================
    # 7. Initialize wandb
    # ============================================
    wandb.init(project="PET-LDM", config={
        "diffusion_steps": DIFFUSION_CONFIG["num_train_timesteps"],
        "batch_size": TRAIN_CONFIG["batch_size"],
        "learning_rate": TRAIN_CONFIG["learning_rate"],
        "latent_dim": AE_CONFIG["latent_dim"],
    })
    
    # ============================================
    # 8. Training loop
    # ============================================
    print("\n" + "="*50)
    print("Starting diffusion training...")
    print("="*50)
    
    best_loss = float("inf")
    early_stopping_patience = 20
    early_stopping_counter = 0
    
    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        train_loss = train_one_epoch(unet, train_loader, optimizer, noise_scheduler, lr_scheduler, DEVICE)

        # 2. Update EMA with current model weights
        #ema.update(unet)

        # 3. For validation, use EMA weights
        #old_weights = ema.apply_to(unet)  # Switch to EMA weights
        val_loss = validate(unet, val_loader, noise_scheduler, DEVICE)
        #ema.restore_from(unet, old_weights)  # Switch back to training weights

        print(f"\nEpoch {epoch+1:03d}/{TRAIN_CONFIG['num_epochs']}")
        print(f"  Train - Loss: {train_loss:.6f}")
        print(f"  Val   - Loss: {val_loss:.6f}")
        
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
            #best_ema = ema.apply_to(unet)
            torch.save({
                "epoch": epoch,
                "model_state_dict": unet.state_dict(),
                #"ema_state_dict": ema.ema_weights,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, DIFFUSION_BEST_PATH)
            #ema.restore_from(unet, best_ema)

            print(f"  -> Saved best model (val_loss: {best_loss:.6f})")
        else:
            early_stopping_counter += 1
            #if early_stopping_counter >= early_stopping_patience:
                #print(f"\nEarly stopping at epoch {epoch+1}")
                #break
        
        # Visualize every N epochs
        if (epoch + 1) % TRAIN_CONFIG["log_every"] == 0:
            #old_weights_viz = ema.apply_to(unet)
            visualize_generation(
                unet, autoencoder.decoder, noise_scheduler, DEVICE, epoch, num_samples=4,
            )
            #ema.restore_from(unet, old_weights_viz)
    
    # ============================================
    # 9. Final evaluation on test set
    # ============================================
    print("\n" + "="*50)
    print("Final evaluation on test set...")
    
    if Path(DIFFUSION_BEST_PATH).exists():
        print("Loading best diffusion model checkpoint...")
        checkpoint = torch.load(DIFFUSION_BEST_PATH, weights_only=False, map_location=DEVICE)
        unet.load_state_dict(checkpoint["model_state_dict"])

        print(f"  Loaded from epoch {checkpoint['epoch']+1} with val_loss: {checkpoint['val_loss']:.6f}")
    
    test_loss = validate(unet, test_loader, noise_scheduler, DEVICE)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.6f}")
    
    # Save final model
    torch.save({
        "model_state_dict": unet.state_dict(),
        "config": {
            "latent_dim": AE_CONFIG["latent_dim"],
            "num_train_timesteps": DIFFUSION_CONFIG["num_train_timesteps"],
        },
        "test_loss": test_loss,
    }, DIFFUSION_FINAL_PATH)
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Final test loss: {test_loss:.6f}")