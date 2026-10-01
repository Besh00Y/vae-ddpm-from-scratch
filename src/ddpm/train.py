import argparse
import json
import os
import time

import torch
import yaml
from torch.optim import Adam
from torchvision.utils import save_image
from tqdm import tqdm

from src.datasets.dataset import DATASET_INFO, get_dataloader
from src.ddpm.diffusion import GaussianDiffusion
from src.ddpm.model import UNet
from src.utils.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from src.utils.ema import EMA
from src.utils.seed import set_seed


def build(cfg, device):
    channels, size = DATASET_INFO[cfg["dataset"]]
    model = UNet(
        in_channels=channels,
        base=cfg["base_channels"],
        ch_mult=tuple(cfg["ch_mult"]),
        num_res=cfg["num_res_blocks"],
        attn_resolutions=tuple(cfg["attn_resolutions"]),
        image_size=size,
        dropout=cfg["dropout"],
    ).to(device)
    diffusion = GaussianDiffusion(
        model, cfg["timesteps"], cfg["beta_start"], cfg["beta_end"]
    ).to(device)
    return model, diffusion, channels, size


def train(cfg, resume=True, max_batches=None):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(cfg.get("amp", False)) and device == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    loader = get_dataloader(cfg["dataset"], cfg["batch_size"], train=True,
                            root=cfg["data_root"], num_workers=cfg["num_workers"])
    model, diffusion, channels, size = build(cfg, device)
    optimizer = Adam(model.parameters(), lr=cfg["lr"])
    ema = EMA(model, cfg["ema_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = cfg["output_dir"]
    ckpt_dir = cfg.get("ckpt_dir") or os.path.join(out, "checkpoints")
    samples_dir = os.path.join(out, "samples")
    plots_dir = os.path.join(out, "plots")
    for d in (ckpt_dir, samples_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    start_epoch, history, train_time = 1, [], 0.0
    last = latest_checkpoint(ckpt_dir) if resume else None
    if last:
        start_epoch, extra = load_checkpoint(last, model, optimizer, device)
        ema.shadow.load_state_dict(extra["ema"])
        history = extra.get("history", [])
        train_time = extra.get("train_time", 0.0)
        print(f"Resumed from {last} (next epoch: {start_epoch})")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0

        for i, (x, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}")):
            if max_batches and i >= max_batches:
                break
            x = x.to(device)

            with torch.autocast(device_type=device, enabled=use_amp):
                loss = diffusion.loss(x)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            total += loss.item()
            n += 1

        train_time += time.time() - t0
        entry = {"epoch": epoch, "loss": total / n}
        history.append(entry)
        print(entry)

        last_epoch = epoch == cfg["epochs"]
        if epoch % cfg["sample_every"] == 0 or last_epoch:
            imgs = (diffusion.sample(16, (channels, size, size), device, model=ema.shadow) + 1) / 2
            save_image(imgs, os.path.join(samples_dir, f"epoch_{epoch}.png"), nrow=4)

        if epoch % cfg["save_every"] == 0 or last_epoch:
            save_checkpoint(
                os.path.join(ckpt_dir, f"epoch_{epoch}.pt"), model, optimizer, epoch,
                {"ema": ema.shadow.state_dict(), "history": history, "train_time": train_time},
            )

    with open(os.path.join(plots_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump({"history": history, "train_time_sec": train_time}, f, indent=2)
    print(f"Done. Total training time: {train_time / 60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ddpm.yaml")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--timesteps", type=int, help="quick tests only")
    p.add_argument("--output_dir")
    p.add_argument("--max_batches", type=int, help="quick tests only")
    p.add_argument("--no_resume", action="store_true")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ("epochs", "batch_size", "timesteps", "output_dir"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    train(cfg, resume=not args.no_resume, max_batches=args.max_batches)