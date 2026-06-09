import math
import torch
import torch.nn as nn


# ============================================================
# Sinusoidal timestep embeddings
# ============================================================

class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):

        device = time.device
        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)

        emb = torch.exp(
            torch.arange(half_dim, device=device) * -emb
        )

        emb = time[:, None] * emb[None, :]

        emb = torch.cat(
            (emb.sin(), emb.cos()),
            dim=-1
        )

        return emb


# ============================================================
# Residual block
# ============================================================

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_dim):

        super().__init__()

        self.time_mlp = nn.Linear(time_dim, out_channels)

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):

        h = self.block1(x)

        time_emb = self.time_mlp(t)
        time_emb = time_emb[:, :, None, None]

        h = h + time_emb

        h = self.block2(h)

        return h + self.res_conv(x)


# ============================================================
# Downsample
# ============================================================

class Downsample(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            4,
            stride=2,
            padding=1
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================
# Upsample
# ============================================================

class Upsample(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            channels,
            channels,
            4,
            stride=2,
            padding=1
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================
# Diffusion UNet
# ============================================================

class ConditionalUNet(nn.Module):

    def __init__(
        self,
        latent_dim=32,
        time_dim=256,
        cond_dim = 512,
    ):

        super().__init__()

        # ====================================================
        # Time embedding
        # ====================================================

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ====================================================
        # Initial projection
        # ====================================================

        self.init_conv = nn.Conv2d(
            latent_dim,
            128,
            3,
            padding=1
        )

        # ====================================================
        # Encoder
        # ====================================================

        self.down1 = ResBlock(128, 256, time_dim)
        self.pool1 = Downsample(256)

        self.down2 = ResBlock(256, 512, time_dim)
        self.pool2 = Downsample(512)

        self.down3 = ResBlock(512, 512, time_dim)

        # ====================================================
        # Bottleneck
        # ====================================================

        self.mid1 = ResBlock(1024, 512, time_dim)

        self.mid2 = ResBlock(512, 512, time_dim)

        # ====================================================
        # Decoder
        # ====================================================

        self.up1 = Upsample(512)
        self.dec1 = ResBlock(1536, 512, time_dim)

        self.up2 = Upsample(512)
        self.dec2 = ResBlock(1024, 256, time_dim)

        self.final = nn.Sequential(
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, latent_dim, 1)
        )


    def forward(self, x, timesteps, mri_features):

        # ====================================================
        # timestep embeddings
        # ====================================================

        t = self.time_mlp(timesteps)

        # ====================================================
        # initial
        # ====================================================

        x = self.init_conv(x)

        # ====================================================
        # encoder
        # ====================================================

        s1 = self.down1(x, t)
        x = self.pool1(s1)

        s2 = self.down2(x, t)
        x = self.pool2(s2)

        s3 = self.down3(x, t)
        x = s3

        # ====================================================
        # bottleneck
        # ====================================================

        x = torch.cat(
            [x, mri_features["4"]],
            dim=1
        )
        x = self.mid1(x, t)

        x = self.mid2(x, t)

        # ====================================================
        # decoder
        # ====================================================

        x = self.up1(x)
        x = torch.cat([x, s2, mri_features["8"]], dim=1)
        x = self.dec1(x, t)

        x = self.up2(x)
        x = torch.cat([x, s1, mri_features["16"]], dim=1)
        x = self.dec2(x, t)

        return self.final(x)