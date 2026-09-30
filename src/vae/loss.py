import torch
import torch.nn.functional as F


def vae_loss(recon, x, mu, logvar, beta=1.0):
    batch = x.size(0)
    # sum over pixels, mean over batch
    recon_loss = F.mse_loss(recon, x, reduction="sum") / batch
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch
    total = recon_loss + beta * kl
    return total, recon_loss, kl