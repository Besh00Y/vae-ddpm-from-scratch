import argparse
import os
import time

import torch
import yaml
from torchvision.utils import save_image

from src.ddpm.train import build
from src.utils.checkpoint import latest_checkpoint, load_checkpoint
from src.utils.ema import EMA
from src.utils.seed import set_seed


@torch.no_grad()
def generate(diffusion, model, n, shape, out_dir, device, batch_size=250, individual=True):
    """Generate n images. individual=True saves one PNG per image (needed for FID).
    Returns the total sampling time in seconds."""
    os.makedirs(out_dir, exist_ok=True)
    count, t0 = 0, time.time()
    while count < n:
        b = min(batch_size, n - count)
        imgs = (diffusion.sample(b, shape, device, model=model) + 1) / 2
        if individual:
            for img in imgs:
                save_image(img, os.path.join(out_dir, f"{count:06d}.png"))
                count += 1
        else:
            save_image(imgs, os.path.join(out_dir, "grid.png"), nrow=8)
            count += b
    if device == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ddpm.yaml")
    p.add_argument("--ckpt")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--out", default="outputs/ddpm/samples")
    p.add_argument("--individual", action="store_true")
    p.add_argument("--batch_size", type=int, default=250)
    p.add_argument("--timesteps", type=int, help="quick tests only")
    p.add_argument("--no_ema", action="store_true")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.timesteps:
        cfg["timesteps"] = args.timesteps

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, diffusion, channels, size = build(cfg, device)

    ckpt = args.ckpt or latest_checkpoint(
        cfg.get("ckpt_dir") or os.path.join(cfg["output_dir"], "checkpoints"))
    _, extra = load_checkpoint(ckpt, model, device=device)

    if args.no_ema:
        sampler = model
    else:
        ema = EMA(model)
        ema.shadow.load_state_dict(extra["ema"])
        sampler = ema.shadow

    secs = generate(diffusion, sampler, args.n, (channels, size, size), args.out,
                    device, args.batch_size, args.individual)
    print(f"Generated {args.n} images in {secs:.1f}s ({secs / args.n:.3f}s per image). Saved to {args.out}")