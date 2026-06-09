"""
KL-Regularized Autoencoder for Latent Diffusion
Produces latents with N(0,1) distribution naturally
"""

import torch
import torch.nn as nn

LATENT_DIM = 32
class KLEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 128 -> 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 64 -> 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 32 -> 16
            nn.Conv2d(256, LATENT_DIM*2, kernel_size=4, stride=2, padding=1),
        )
        self.latent_dim = LATENT_DIM


    def forward(self, x):
        h = self.net(x)
        mu = h[:, :self.latent_dim]
        #print(mu.shape)
        log_var = h[:, self.latent_dim:]
        return mu, log_var


class KLDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 16 -> 32
            nn.ConvTranspose2d(LATENT_DIM, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 32 -> 64
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 64 -> 128
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 128 -> 256
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 128 -> 256
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            #nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.net(z)
        return torch.clamp(x, 0, 1)


class KLAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = KLEncoder()
        self.decoder = KLDecoder()
        self.latent_dim = LATENT_DIM
    
    def encode(self, x):
        """Encode to distribution parameters"""
        mu, log_var = self.encoder(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick for sampling"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent to image"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass (training)"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var, z