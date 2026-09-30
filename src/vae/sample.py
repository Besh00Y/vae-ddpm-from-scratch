import argparse
import os

import torch
import yaml
from torchvision.utils import save_image

from src.datasets.dataset import DATASET_INFO
from src.utils.checkpoint import latest_checkpoint, load_checkpoint
from src.utils.seed import set_seed
from src.vae.model import VAE


@torch.no_grad()
def generate(model, n, out_dir, device, batch_size=500, individual=False):
    """يولّد n صورة. individual=True بيحفظ كل صورة لوحدها (مطلوب لحساب الـ FID)."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    count = 0
    while count < n:
        b = min(batch_size, n - count)
        imgs = (model.sample(b, device) + 1) / 2   # [-1,1] -> [0,1]
        if individual:
            for img in imgs:
                save_image(img, os.path.join(out_dir, f"{count:06d}.png"))
                count += 1
        else:
            count += b
    return count


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/vae.yaml")
    p.add_argument("--ckpt")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--out", default="outputs/vae/samples")
    p.add_argument("--individual", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    channels, _ = DATASET_INFO[cfg["dataset"]]
    model = VAE(channels, cfg["latent_dim"], cfg["base_channels"]).to(device)

    ckpt = args.ckpt or latest_checkpoint(
        cfg.get("ckpt_dir") or os.path.join(cfg["output_dir"], "checkpoints"))
    load_checkpoint(ckpt, model, device=device)

    if args.individual:
        generate(model, args.n, args.out, device, individual=True)
    else:
        os.makedirs(args.out, exist_ok=True)
        imgs = (model.sample(args.n, device) + 1) / 2
        save_image(imgs, os.path.join(args.out, "grid.png"), nrow=8)
    print("Saved to", args.out)
    