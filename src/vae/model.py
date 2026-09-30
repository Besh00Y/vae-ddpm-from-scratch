import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.base = base

        # Encoder: 32x32 -> 16 -> 8 -> 4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, 2, 1),
            nn.BatchNorm2d(base), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.BatchNorm2d(base * 2), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.BatchNorm2d(base * 4), 
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        flat = base * 4 * 4 * 4
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

        # Decoder: 4x4 -> 8 -> 16 -> 32
        self.fc_dec = nn.Linear(latent_dim, flat)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),
            nn.BatchNorm2d(base * 2), 
            nn.ReLU(),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.BatchNorm2d(base), 
            nn.ReLU(),
            nn.ConvTranspose2d(base, in_channels, 4, 2, 1),
            nn.Tanh(),  # output in [-1, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self.base * 4, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)