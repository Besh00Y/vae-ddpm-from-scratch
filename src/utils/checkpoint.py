import os

import torch


def save_checkpoint(path, model, optimizer, epoch, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "extra": extra or {},
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Returns the epoch to resume from."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state["epoch"] + 1, state.get("extra", {})


def latest_checkpoint(directory):
    """Path of the most recent epoch_*.pt file in the directory, or None."""
    if not os.path.isdir(directory):
        return None
    files = [f for f in os.listdir(directory) if f.startswith("epoch_") and f.endswith(".pt")]
    if not files:
        return None
    files.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))
    return os.path.join(directory, files[-1])