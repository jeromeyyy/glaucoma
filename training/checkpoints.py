import os
import torch
from config import Config

def checkpoint_exists(path=None):
    if path is None:
        path = Config.BEST_MODEL_PATH
    return os.path.exists(path)

def save_checkpoint(
    model,
    optimizer=None,
    epoch=0,
    val_metrics=None,
    save_path=None,
    is_best=True
):
    if save_path is None:
        save_path = Config.BEST_MODEL_PATH

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_metrics": val_metrics or {},
        "config_mode": getattr(model, "config_mode", "full_xhgnn_vlt")
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(state, save_path)
    print(f"Checkpoint saved to {save_path} (Epoch {epoch})")
    return save_path

def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Safely loads checkpoint weights into model.
    Returns:
        dict with metadata (epoch, val_metrics)
    Raises:
        FileNotFoundError if checkpoint does not exist.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No trained model checkpoint found at '{checkpoint_path}'. "
            "Please train the model or provide a valid .pth checkpoint file."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    val_metrics = checkpoint.get("val_metrics", {})
    return {
        "epoch": epoch,
        "val_metrics": val_metrics,
        "config_mode": checkpoint.get("config_mode", "full_xhgnn_vlt")
    }
