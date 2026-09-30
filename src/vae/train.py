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
from src.utils.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from src.utils.seed import set_seed
from src.vae.loss import vae_loss
from src.vae.model import VAE


def train(cfg, resume=True, max_batches=None):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    channels, _ = DATASET_INFO[cfg["dataset"]]
    loader = get_dataloader(cfg["dataset"], cfg["batch_size"], train=True,
                            root=cfg["data_root"], num_workers=cfg["num_workers"])

    model = VAE(channels, cfg["latent_dim"], cfg["base_channels"]).to(device)
    optimizer = Adam(model.parameters(), lr=cfg["lr"])

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
        history = extra.get("history", [])
        train_time = extra.get("train_time", 0.0)
        print(f"Resumed from {last} (next epoch: {start_epoch})")

    # نفس الـ z في كل مرة عشان نشوف تطور الجودة
    g = torch.Generator().manual_seed(0)
    fixed_z = torch.randn(64, cfg["latent_dim"], generator=g).to(device)

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        tot = rec = kl_sum = 0.0
        n_batches = 0

        for i, (x, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}")):
            if max_batches and i >= max_batches:
                break
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, r, k = vae_loss(recon, x, mu, logvar, cfg["beta"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot += loss.item(); rec += r.item(); kl_sum += k.item()
            n_batches += 1

        train_time += time.time() - t0
        entry = {"epoch": epoch, "loss": tot / n_batches,
                 "recon": rec / n_batches, "kl": kl_sum / n_batches}
        history.append(entry)
        print(entry)

        if epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
            model.eval()
            with torch.no_grad():
                imgs = (model.decode(fixed_z) + 1) / 2
            save_image(imgs, os.path.join(samples_dir, f"epoch_{epoch}.png"), nrow=8)
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch}.pt"), model, optimizer,
                            epoch, {"history": history, "train_time": train_time})

    with open(os.path.join(plots_dir, "history.json"), "w") as f:
        json.dump({"history": history, "train_time_sec": train_time}, f, indent=2)
    print(f"Done. Total training time: {train_time / 60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/vae.yaml")
    p.add_argument("--epochs", type=int)
    p.add_argument("--output_dir")
    p.add_argument("--max_batches", type=int, help="للتجربة السريعة بس")
    p.add_argument("--no_resume", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    train(cfg, resume=not args.no_resume, max_batches=args.max_batches)