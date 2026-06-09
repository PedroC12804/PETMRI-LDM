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

from dataset_MRI import PETMRISliceDataset
from models.PET_KL_Autoencoder import KLAutoencoder
from models.diffusion_model_MRI import ConditionalUNet
from models.MRI_Encoder import MRIEncoder

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
    "learning_rate": 1e-4,
    "num_epochs": 350,
    "num_workers": 0,
    "log_every": 10,
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
    val_exams = exams_shuffled[n_test:n_test + n_val]
    train_exams = exams_shuffled[n_test + n_val:]

    return train_exams, val_exams, test_exams


@torch.no_grad()
def compute_latent_std(loader, autoencoder, device):

    latents = []

    for pet, _ in tqdm(loader, desc="Computing latent std"):

        pet = pet.to(device)

        mu, logvar = autoencoder.encoder(pet)

        latents.append(mu.cpu())

    latents = torch.cat(latents)

    std = latents.std()

    print(f"Latent std: {std:.4f}")

    return std.to(device)

def train_one_epoch(unet,mri_encoder,pet_autoencoder,loader,optimizer,noise_scheduler,
    lr_scheduler,latent_std,device,):
    unet.train()
    mri_encoder.train()
    total_loss = 0

    for pet,mri in tqdm(loader, desc="Training", leave=False):

        pet = pet.to(device)
        mri = mri.to(device)

        # Get raw latents
        with torch.no_grad():
            mu, logvar = pet_autoencoder.encoder(pet)
            latents = mu/latent_std

        #Encode MRI
        mri_features = mri_encoder(mri)

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
        noise_pred = unet(noisy_latents, timesteps,mri_features)

        # Loss on noise prediction
        loss = F.mse_loss(noise_pred, noise)

        # Backpropagate (will train both the Unet and the MRI Encoder)
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(list(unet.parameters()) +
            list(mri_encoder.parameters()), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(unet,mri_encoder,pet_autoencoder,loader,
    noise_scheduler,latent_std, device,):

    unet.eval()
    mri_encoder.eval()

    total_loss = 0

    for pet,mri in tqdm(loader, desc="Validation", leave=False):
        pet = pet.to(device)
        mri = mri.to(device)

        #encode PET
        mu, logvar = pet_autoencoder.encoder(pet)

        latents = mu / latent_std

        #encode MRI
        mri_features = mri_encoder(mri)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device
        ).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        noise_pred = unet(noisy_latents, timesteps,mri_features)
        loss = F.mse_loss(noise_pred, noise)

        total_loss += loss.item()

    return total_loss / len(loader)


def visualize_generation(
    unet,
    mri_encoder,
    autoencoder,
    scheduler,
    pet,
    mri,
    fixed_noise,
    latent_std,
    device,
    epoch,
    num_inference_steps=1000,
):

    unet.eval()
    mri_encoder.eval()

    # ============================================
    # Move data to device
    # ============================================
    pet = pet.to(device)
    mri = mri.to(device)

    # Clone noise so original stays unchanged
    latents = fixed_noise.clone().to(device)

    # ============================================
    # Encode MRI -> conditioning tokens
    # ============================================
    with torch.no_grad():
        mri_features = mri_encoder(mri)

    # ============================================
    # Diffusion sampling
    # ============================================
    scheduler.set_timesteps(num_inference_steps)

    with torch.no_grad():

        for t in tqdm(
            scheduler.timesteps,
            desc="Sampling",
            leave=False
        ):

            t_batch = torch.full(
                (latents.shape[0],),
                t,
                device=device,
                dtype=torch.long,
            )

            # Predict noise
            noise_pred = unet(
                latents,
                t_batch,
                mri_features,
            )

            # DDPM step
            latents = scheduler.step(
                noise_pred,
                t,
                latents
            ).prev_sample

    # ============================================
    # Undo latent normalization
    # ============================================
    latents = latents * latent_std

    # ============================================
    # Decode PET
    # ============================================
    with torch.no_grad():
        generated_pet = autoencoder.decoder(latents)

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
        4,
        figsize=(16, 4 * batch_size)
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
        # Ground Truth PET
        # ----------------------------------------
        axes[i][1].imshow(
            pet[i, 0].cpu().numpy(),
            cmap="hot"
        )
        axes[i][1].set_title("Ground Truth PET")
        axes[i][1].axis("off")

        # ----------------------------------------
        # Generated PET
        # ----------------------------------------
        axes[i][2].imshow(
            generated_pet[i, 0].cpu().numpy(),
            cmap="hot"
        )
        axes[i][2].set_title("Generated PET")
        axes[i][2].axis("off")

        # ----------------------------------------
        # Absolute Error Map
        # ----------------------------------------
        error_map = torch.abs(
            generated_pet[i, 0] - pet[i, 0]
        )

        axes[i][3].imshow(
            error_map.cpu().numpy(),
            cmap="inferno"
        )
        axes[i][3].set_title("Absolute Error")
        axes[i][3].axis("off")

    plt.tight_layout()

    save_path = PROJECT_ROOT / "Images"
    save_path.mkdir(parents=True, exist_ok=True)
    save_figure = save_path /  f"generated_epoch_{epoch+1}.png"

    plt.savefig(save_figure, dpi=150)

    plt.close()

    # ============================================
    # Log to wandb
    # ============================================
    wandb.log({
        "generated_samples": wandb.Image(save_figure)
    })

# ============================================
# MAIN TRAINING
# ============================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # ============================================
    # 1. Load trained KL-VAE autoencoder
    # ============================================
    print("\n" + "=" * 50)
    print("Loading trained KL-VAE autoencoder...")
    print("=" * 50)

    autoencoder = KLAutoencoder().to(DEVICE)

    mri_encoder = MRIEncoder(in_channels=1).to(DEVICE)

    checkpoint = torch.load(AE_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False

    print(f"Autoencoder loaded from epoch {checkpoint['epoch'] + 1}")
    print(f"  Val loss: {checkpoint['val_loss']:.6f}")
    print(f"  Val SSIM: {checkpoint['val_ssim']:.4f}")
    print(f"  Val KL: {checkpoint['val_kl']:.6f}")

    # ============================================
    # 2. Create dataset and dataloaders
    # ============================================
    print("\n" + "=" * 50)
    print("Loading dataset...")
    print("=" * 50)

    all_exams = sorted([d for d in DATASET_ROOT.iterdir() if d.is_dir()])

    # all_exams=all_exams[:100] #!!!!!!!!!!!!!!!!!!!!!!!!!

    # Split data
    train_exams, val_exams, test_exams = train_val_test_split(
        all_exams, test_ratio=0.15, val_ratio=0.15, seed=42
    )
    print(f"Train: {len(train_exams)} exams")
    print(f"Val: {len(val_exams)} exams")
    print(f"Test: {len(test_exams)} exams")


    # Create PET datasets
    train_dataset = PETMRISliceDataset(train_exams, target_size=AE_CONFIG["target_size"], transform=True)
    val_dataset = PETMRISliceDataset(val_exams, target_size=AE_CONFIG["target_size"])
    test_dataset = PETMRISliceDataset(test_exams, target_size=AE_CONFIG["target_size"])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True,
                                     num_workers=TRAIN_CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False,
                                   num_workers=TRAIN_CONFIG["num_workers"], pin_memory=True)
    # Fixed validation examples
    viz_pet, viz_mri = next(iter(val_loader))

    viz_pet = viz_pet[:4]
    viz_mri = viz_mri[:4]

    # Fixed initial noise
    g = torch.Generator().manual_seed(42)

    fixed_noise = torch.randn(
        (4, 32, 16, 16),
        generator=g
    ).to(DEVICE)

    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False,
                                    num_workers=TRAIN_CONFIG["num_workers"], pin_memory=True)

    # ============================================
    # 4. Initialize diffusion model
    # ============================================
    print("\n" + "=" * 50)
    print("Initializing diffusion model...")
    print("=" * 50)

    unet = ConditionalUNet(
    latent_dim=32,
    time_dim=256,
    cond_dim=512,
).to(DEVICE)

    # ema = EMA(beta=0.9999)

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
    optimizer = torch.optim.AdamW(list(unet.parameters()) +
    list(mri_encoder.parameters()), lr=TRAIN_CONFIG["learning_rate"])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=TRAIN_CONFIG["warmup_steps"],
        num_training_steps=len(train_loader) * TRAIN_CONFIG["num_epochs"],
    )

    # ============================================
    # 7. Initialize wandb
    # ============================================
    wandb.init(project="PET-MRI-LDM", config={
        "diffusion_steps": DIFFUSION_CONFIG["num_train_timesteps"],
        "batch_size": TRAIN_CONFIG["batch_size"],
        "learning_rate": TRAIN_CONFIG["learning_rate"],
        "latent_dim": AE_CONFIG["latent_dim"],
    })

    # ============================================
    # 8. Training loop
    # ============================================
    print("\n" + "=" * 50)
    print("Starting diffusion training...")
    print("=" * 50)

    best_loss = float("inf")
    early_stopping_patience = 20
    early_stopping_counter = 0

    latent_std = compute_latent_std(
        train_loader,
        autoencoder,
        DEVICE
    )
    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        train_loss = train_one_epoch(
            unet,
            mri_encoder,
            autoencoder,
            train_loader,
            optimizer,
            noise_scheduler,
            lr_scheduler,
            latent_std,
            DEVICE,
        )
        # 2. Update EMA with current model weights
        # ema.update(unet)

        # 3. For validation, use EMA weights
        # old_weights = ema.apply_to(unet)  # Switch to EMA weights
        val_loss = validate(
            unet,
            mri_encoder,
            autoencoder,
            val_loader,
            noise_scheduler,
            latent_std,
            DEVICE,
        )
        # ema.restore_from(unet, old_weights)  # Switch back to training weights

        print(f"\nEpoch {epoch + 1:03d}/{TRAIN_CONFIG['num_epochs']}")
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
            # best_ema = ema.apply_to(unet)
            torch.save({
                "epoch": epoch,
                "model_state_dict": unet.state_dict(),
                "mri_encoder_state_dict": mri_encoder.state_dict(),
                # "ema_state_dict": ema.ema_weights,
                "optimizer_state_dict": optimizer.state_dict(),
                "latent_std": latent_std,
                "val_loss": val_loss,
            }, DIFFUSION_BEST_PATH)
            # ema.restore_from(unet, best_ema)

            print(f"  -> Saved best model (val_loss: {best_loss:.6f})")
        else:
            early_stopping_counter += 1
            # if early_stopping_counter >= early_stopping_patience:
            # print(f"\nEarly stopping at epoch {epoch+1}")
            # break

        # Visualize every N epochs
        if (epoch + 1) % TRAIN_CONFIG["log_every"] == 0:
            # old_weights_viz = ema.apply_to(unet)
            visualize_generation(
                unet,
                mri_encoder,
                autoencoder,
                noise_scheduler,
                viz_pet,
                viz_mri,
                fixed_noise,
                latent_std,
                DEVICE,
                epoch,
            )
            # ema.restore_from(unet, old_weights_viz)

    # ============================================
    # 9. Final evaluation on test set
    # ============================================
    print("\n" + "=" * 50)
    print("Final evaluation on test set...")

    if Path(DIFFUSION_BEST_PATH).exists():
        print("Loading best diffusion model checkpoint...")
        checkpoint = torch.load(DIFFUSION_BEST_PATH, weights_only=False, map_location=DEVICE)
        unet.load_state_dict(checkpoint["model_state_dict"])
        mri_encoder.load_state_dict(checkpoint["mri_encoder_state_dict"])

        print(f"  Loaded from epoch {checkpoint['epoch'] + 1} with val_loss: {checkpoint['val_loss']:.6f}")

    test_loss = validate(
        unet,
        mri_encoder,
        autoencoder,
        test_loader,
        noise_scheduler,
        latent_std,
        DEVICE,
    )

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.6f}")

    # Save final model
    torch.save({
        "model_state_dict": unet.state_dict(),
        "mri_encoder_state_dict": mri_encoder.state_dict(),
        "latent_std": latent_std,
        "config": {
            "latent_dim": AE_CONFIG["latent_dim"],
            "num_train_timesteps": DIFFUSION_CONFIG["num_train_timesteps"],
        },
        "test_loss": test_loss,
    }, DIFFUSION_FINAL_PATH)

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Final test loss: {test_loss:.6f}")