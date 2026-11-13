import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import argparse
import json
import torch

from config import Config
from datasets.dataset_loader import get_dataloaders
from models.full_model import ExplainableGlaucomaModel
from models.baseline_cnn import BaselineCNN
from training.checkpoints import load_checkpoint, checkpoint_exists
from training.validate import validate_epoch
from evaluation.plots import plot_confusion_matrix, plot_roc_curve, plot_pr_curve
from utils.logging import logger

def evaluate_model(
    model_type="proposed",
    checkpoint_path=None,
    dataset_name="combined",
    batch_size=8,
    device=None,
    output_dir=None
):
    if device is None:
        device = Config.DEVICE
    output_dir = output_dir or Config.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. DataLoader - evaluate strictly on the held-out TEST set
    _, _, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        image_size=Config.IMAGE_SIZE,
        num_workers=0
    )
    
    # 2. Model initialization
    if model_type == "baseline_cnn":
        model = BaselineCNN(num_classes=Config.NUM_CLASSES)
        default_ckpt = Config.BASELINE_MODEL_PATH
    else:
        config_mode = model_type if model_type in ["efficientnet_b3", "full_xhgnn_vlt"] else "full_xhgnn_vlt"
        model = ExplainableGlaucomaModel(
            num_classes=Config.NUM_CLASSES,
            pretrained=False,
            fine_tune=False,
            config_mode=config_mode
        )
        default_ckpt = Config.BEST_MODEL_PATH

    ckpt = checkpoint_path or default_ckpt
    if os.path.exists(ckpt):
        logger.info(f"Loading checkpoint from: {ckpt}")
        load_checkpoint(ckpt, model, device=device)
    else:
        logger.warning(f"No checkpoint found at '{ckpt}'. Evaluating initialized model weights.")

    model = model.to(device)

    # 3. Test Evaluation
    metrics, targets, preds, probs = validate_epoch(model, test_loader, device=device)
    logger.info(f"Test Evaluation Results ({model_type}): {json.dumps(metrics, indent=2)}")

    # 4. Save results and plots
    metrics_path = os.path.join(output_dir, f"metrics_{model_type}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm_path = os.path.join(output_dir, f"confusion_matrix_{model_type}.png")
    plot_confusion_matrix(metrics["confusion_matrix"], save_path=cm_path)

    roc_path = os.path.join(output_dir, f"roc_curve_{model_type}.png")
    plot_roc_curve(targets, probs, save_path=roc_path)

    pr_path = os.path.join(output_dir, f"pr_curve_{model_type}.png")
    plot_pr_curve(targets, probs, save_path=pr_path)

    logger.info(f"Saved evaluation artifacts to {output_dir}/")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Glaucoma Models")
    parser.add_argument("--model", type=str, default="proposed", choices=["proposed", "baseline_cnn", "efficientnet_b3"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="combined")
    args = parser.parse_args()

    evaluate_model(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset
    )
