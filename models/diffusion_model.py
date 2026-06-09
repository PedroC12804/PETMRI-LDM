"""
U-Net for DDPM in latent space
Uses Hugging Face Diffusers library
"""

from diffusers import UNet2DModel
import torch.nn as nn
import torch

def create_diffusion_unet(
    latent_dim: int = 32,
    latent_size: int = 16,
    block_out_channels: tuple = (256, 512, 512),
) -> nn.Module:
    """
    Create a U-Net for DDPM in latent space.
    
    Args:
        latent_dim: Number of channels in latent space (16)
        latent_size: Spatial size of latent (16x16)
        block_out_channels: Channels at each down/up block
    
    Returns:
        UNet2DModel configured for latent space diffusion
    """
    unet = UNet2DModel(
        sample_size=latent_size,           # 8x8 latent
        in_channels=latent_dim,            # 8 channels
        out_channels=latent_dim,           # 8 channels
        layers_per_block=2,
        block_out_channels=block_out_channels,
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        act_fn="silu",
        attention_head_dim=32,
        norm_num_groups=32,
    )
    
    return unet


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = create_diffusion_unet(latent_dim=16, latent_size=16)
    print(f"U-Net parameters: {count_parameters(model):,}")
    
    # Test forward pass
    
    x = torch.randn(4, 32, 8, 8)
    timestep = torch.randint(0, 500, (1,))
    out = model(x, timestep).sample
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")