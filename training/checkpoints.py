import os
import hashlib
import urllib.request
import logging
import torch
from config import Config

logger = logging.getLogger("glucom.checkpoints")

CHECKPOINT_METADATA = {
    "best_model.pth": {
        "sha256": "dbdbe131f9de84b162dded670eb1827d63291a51b9aa4d303c4b872db93d8cba",
        "size": 48774411,
        "url": "https://media.githubusercontent.com/media/jeromeyyy/glaucoma/main/checkpoints/best_model.pth"
    },
    "baseline_cnn.pth": {
        "sha256": "a2236df998a9c25e95f019a934728e6b99e2f4372c3f24afe7ab6de9e0d3ae49",
        "size": 5102457,
        "url": "https://media.githubusercontent.com/media/jeromeyyy/glaucoma/main/checkpoints/baseline_cnn.pth"
    }
}

def compute_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def is_lfs_pointer(path: str) -> bool:
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if size < 1000:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.read(64)
                return "git-lfs" in header or "oid sha256" in header
        except Exception:
            return True
    return False

def hydrate_checkpoint(checkpoint_path: str) -> str:
    """
    Ensures the checkpoint is an actual binary weights file,
    hydrating from verified repository media storage if cloned as an LFS pointer.
    """
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    filename = os.path.basename(checkpoint_path)
    meta = CHECKPOINT_METADATA.get(filename)

    needs_download = False
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint '{filename}' not found locally. Initiating cloud hydration...")
        needs_download = True
    elif is_lfs_pointer(checkpoint_path) or os.path.getsize(checkpoint_path) < 100000:
        logger.warning(f"Checkpoint '{filename}' is a Git LFS pointer text file ({os.path.getsize(checkpoint_path)} bytes). Hydrating real binary weights...")
        needs_download = True

    if needs_download and meta:
        url = meta["url"]
        logger.info(f"Downloading verified weights from {url}...")
        try:
            temp_path = checkpoint_path + ".tmp"
            urllib.request.urlretrieve(url, temp_path)
            actual_sha = compute_sha256(temp_path)
            expected_sha = meta["sha256"]
            if actual_sha != expected_sha:
                logger.error(f"SHA-256 mismatch for {filename}! Expected: {expected_sha}, Got: {actual_sha}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise ValueError(f"Integrity check failed for {filename}")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            os.rename(temp_path, checkpoint_path)
            logger.info(f"Successfully hydrated {filename} ({os.path.getsize(checkpoint_path):,} bytes, SHA-256 verified).")
        except Exception as e:
            logger.error(f"Failed to download verified weights for {filename}: {e}")
            raise

    return checkpoint_path

def checkpoint_exists(path=None):
    if path is None:
        path = Config.BEST_MODEL_PATH
    if not os.path.exists(path):
        return False
    return not is_lfs_pointer(path)

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
    Safely and strictly loads checkpoint weights into model.
    Returns metadata with verified validation metrics and checksum.
    """
    hydrate_checkpoint(checkpoint_path)

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
    sha256 = compute_sha256(checkpoint_path)
    
    return {
        "epoch": epoch,
        "val_metrics": val_metrics,
        "config_mode": checkpoint.get("config_mode", "full_xhgnn_vlt"),
        "sha256": sha256
    }
