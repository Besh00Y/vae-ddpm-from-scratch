import argparse
import json
import os
import time

import torch
import yaml
from torchvision.utils import save_image

from src.datasets.dataset import DATASET_INFO
from src.evaluation.metrics import compute_metrics
from src.utils.checkpoint import latest_checkpoint, load_checkpoint
from src.utils.ema import EMA
from src.utils.seed import set_seed


def read_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_ckpt(model_dir, explicit=None):
    ckpt = explicit or latest_checkpoint(os.path.join(model_dir, "checkpoints"))
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}/checkpoints")
    return ckpt


def load_vae(cfg, model_dir, ckpt, device):
    from src.vae.model import VAE

    channels, _ = DATASET_INFO[cfg["dataset"]]
    model = VAE(channels, cfg["latent_dim"], cfg["base_channels"]).to(device)
    _, extra = load_checkpoint(find_ckpt(model_dir, ckpt), model, device=device)
    model.eval()
    return model, (lambda b: model.sample(b, device)), extra


def load_ddpm(cfg, model_dir, ckpt, device):
    from src.ddpm.train import build

    model, diffusion, channels, size = build(cfg, device)
    _, extra = load_checkpoint(find_ckpt(model_dir, ckpt), model, device=device)
    ema = EMA(model)
    ema.shadow.load_state_dict(extra["ema"])
    sample_fn = lambda b: diffusion.sample(b, (channels, size, size), device, model=ema.shadow)
    return model, sample_fn, extra


@torch.no_grad()
def generate_images(sample_fn, n, out_dir, batch_size):
    """Saves n PNGs. Resumes if the folder is partly filled (Colab can disconnect)."""
    os.makedirs(out_dir, exist_ok=True)
    done = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
    if done >= n:
        print(f"{out_dir}: {done} images already there, skipping generation")
        return
    while done < n:
        b = min(batch_size, n - done)
        imgs = (sample_fn(b) + 1) / 2          # [-1, 1] -> [0, 1]
        for img in imgs:
            save_image(img, os.path.join(out_dir, f"{done:06d}.png"))
            done += 1
        print(f"  {done}/{n}", flush=True)


@torch.no_grad()
def seconds_per_image(sample_fn, device, n=64, warmup=False):
    """Pure sampling speed (no PNG saving)."""
    if warmup:
        sample_fn(8)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    sample_fn(n)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["vae", "ddpm"], choices=["vae", "ddpm"])
    p.add_argument("--vae_config", default="configs/vae.yaml")
    p.add_argument("--ddpm_config", default="configs/ddpm.yaml")
    p.add_argument("--vae_dir", default="outputs/vae", help="folder that contains checkpoints/")
    p.add_argument("--ddpm_dir", default="outputs/ddpm", help="folder that contains checkpoints/")
    p.add_argument("--vae_ckpt")
    p.add_argument("--ddpm_ckpt")
    p.add_argument("--n", type=int, default=10000, help="generated images per model")
    p.add_argument("--batch_size", type=int, default=250)
    p.add_argument("--reference", default="cifar10-train")
    p.add_argument("--gen_root", default="outputs/generated")
    p.add_argument("--out", default="outputs/results.json")
    p.add_argument("--baseline", action="store_true",
                   help="also compute FID/IS of real test images vs the reference (sanity check)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    results = {}
    if os.path.exists(args.out):               # keep results of models evaluated earlier
        with open(args.out, encoding="utf-8") as f:
            results = json.load(f)

    if args.baseline:
        print("Baseline: real CIFAR-10 test images vs", args.reference)
        results["real_test_baseline"] = compute_metrics("cifar10-val", args.reference, device == "cuda")

    for name in args.models:
        print(f"\n=== {name.upper()} ===")
        set_seed(42)
        if name == "vae":
            cfg = read_cfg(args.vae_config)
            model, sample_fn, extra = load_vae(cfg, args.vae_dir, args.vae_ckpt, device)
        else:
            cfg = read_cfg(args.ddpm_config)
            model, sample_fn, extra = load_ddpm(cfg, args.ddpm_dir, args.ddpm_ckpt, device)

        gen_dir = os.path.join(args.gen_root, f"{name}_{args.n}")
        generate_images(sample_fn, args.n, gen_dir, args.batch_size)

        print("Computing FID / IS ...")
        scores = compute_metrics(gen_dir, args.reference, device == "cuda")
        spi = seconds_per_image(sample_fn, device, warmup=(name == "vae"))

        results[name] = {
            **scores,
            "n_generated": args.n,
            "reference": args.reference,
            "sec_per_image": spi,
            "images_per_sec": 1 / spi,
            "train_time_min": extra.get("train_time", 0.0) / 60,
            "params_millions": sum(p.numel() for p in model.parameters()) / 1e6,
            "epochs_trained": len(extra.get("history", [])),
            "device": device,
        }
        print(results[name])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n| Model | FID | IS | train (min) | sec/image | params (M) |")
    print("|---|---|---|---|---|---|")
    for k, r in results.items():
        if k == "real_test_baseline":
            print(f"| real test set | {r['fid']:.2f} | {r['is_mean']:.2f} ± {r['is_std']:.2f} | - | - | - |")
        else:
            print(f"| {k.upper()} | {r['fid']:.2f} | {r['is_mean']:.2f} ± {r['is_std']:.2f} | "
                  f"{r['train_time_min']:.1f} | {r['sec_per_image']:.4f} | {r['params_millions']:.1f} |")
    print("\nSaved", args.out)


if __name__ == "__main__":
    main()