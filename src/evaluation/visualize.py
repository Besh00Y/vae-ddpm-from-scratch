import argparse
import os

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.utils import make_grid


def grid_from_dir(directory, n=64):
    files = sorted(f for f in os.listdir(directory) if f.endswith(".png"))[:n]
    imgs = [read_image(os.path.join(directory, f)).float() / 255 for f in files]
    return make_grid(torch.stack(imgs), nrow=8, padding=2)


def real_grid(root="data", n=64, seed=0):
    ds = datasets.CIFAR10(root, train=False, download=True, transform=transforms.ToTensor())
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n]
    return make_grid(torch.stack([ds[int(i)][0] for i in idx]), nrow=8, padding=2)


def compare(vae_dir, ddpm_dir, out_path, data_root="data"):
    panels = [
        ("Real CIFAR-10", real_grid(data_root)),
        ("VAE", grid_from_dir(vae_dir)),
        ("DDPM", grid_from_dir(ddpm_dir)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, (title, grid) in zip(axes, panels):
        ax.imshow(grid.permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Saved", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vae_gen", required=True, help="folder of generated VAE PNGs")
    p.add_argument("--ddpm_gen", required=True, help="folder of generated DDPM PNGs")
    p.add_argument("--out", default="outputs/comparison.png")
    p.add_argument("--data_root", default="data")
    a = p.parse_args()
    compare(a.vae_gen, a.ddpm_gen, a.out, a.data_root)