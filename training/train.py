import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import argparse
import json
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from utils.seed import set_seed
from utils.logging import logger
from datasets.dataset_loader import get_dataloaders
from models.full_model import ExplainableGlaucomaModel
from models.baseline_cnn import BaselineCNN
from training.losses import GlaucomaMultiTaskLoss
from training.validate import validate_epoch
from training.checkpoints import save_checkpoint
from evaluation.plots import plot_training_curves

def train_model(
    model_type="proposed",
    dataset_name="combined",
    num_epochs=5,
    batch_size=8,
    lr=1e-4,
    device=None,
    save_path=None
):
    if device is None:
        device = Config.DEVICE
    set_seed(Config.SEED)
    logger.info(f"Starting training on {device} | Model: {model_type} | Dataset: {dataset_name} | Epochs: {num_epochs}")

    # 1. Load data
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        image_size=Config.IMAGE_SIZE,
        num_workers=Config.NUM_WORKERS
    )
    logger.info(f"DataLoaders prepared: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples, {len(test_loader.dataset)} test samples")

    # 2. Instantiate Model
    if model_type == "baseline_cnn":
        model = BaselineCNN(num_classes=Config.NUM_CLASSES)
        criterion = None  # Standard CrossEntropy inside loop
        save_path = save_path or Config.BASELINE_MODEL_PATH
    else:
        config_mode = model_type if model_type in ["efficientnet_b3", "full_xhgnn_vlt"] else "full_xhgnn_vlt"
        model = ExplainableGlaucomaModel(
            num_classes=Config.NUM_CLASSES,
            pretrained=True,
            fine_tune=True,
            config_mode=config_mode
        )
        criterion = GlaucomaMultiTaskLoss(
            sparsity_weight=Config.SPARSITY_LOSS_WEIGHT,
            entropy_weight=Config.ENTROPY_LOSS_WEIGHT,
            concept_weight=Config.CONCEPT_LOSS_WEIGHT
        )
        save_path = save_path or Config.BEST_MODEL_PATH

        # Staged training: warmup epochs 1-2 freeze backbone to train HGNN & heads cleanly
        if num_epochs > 2:
            for param in model.backbone.features.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen for initial warmup epochs (staged fine-tuning).")

    model = model.to(device)

    # 3. Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Temperature annealing for Gumbel-Softmax
    temp = Config.GUMBEL_TEMPERATURE
    temp_min = Config.GUMBEL_MIN_TEMP
    anneal_rate = Config.GUMBEL_ANNEAL_RATE

    best_f1 = -1.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }

    for epoch in range(1, num_epochs + 1):
        # Stage 2: unfreeze upper backbone blocks on epoch 3
        if model_type != "baseline_cnn" and epoch == 3 and num_epochs > 2:
            logger.info("Unfreezing upper convolutional blocks (features[6:]) for fine-tuning...")
            for block in model.backbone.features[6:]:
                for param in block.parameters():
                    param.requires_grad = True
            # Recreate optimizer with updated parameter set
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = AdamW(trainable_params, lr=lr * 0.7, weight_decay=Config.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - epoch + 1, eta_min=1e-6)

        model.train()
        train_loss = 0.0
        n_samples = 0

        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            if model_type == "baseline_cnn":
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, targets)
            else:
                outputs = model(images, temperature=temp, hard_selection=False)
                loss, loss_details = criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            n_samples += images.size(0)

        scheduler.step()
        temp = max(temp * anneal_rate, temp_min)

        epoch_train_loss = train_loss / max(n_samples, 1)

        # Validation
        val_metrics, _, _, _ = validate_epoch(model, val_loader, criterion=criterion, device=device)
        epoch_val_loss = val_metrics["val_loss"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1_score"]

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        logger.info(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val AUC: {val_metrics.get('roc_auc', 0.5):.4f}"
        )

        # Checkpointing
        if val_f1 >= best_f1:
            best_f1 = val_f1
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_metrics=val_metrics,
                save_path=save_path
            )

    # Save training curves
    curve_path = os.path.join(Config.RESULTS_DIR, f"training_curve_{model_type}.png")
    plot_training_curves(history, save_path=curve_path)

    # Save training history JSON
    hist_path = os.path.join(Config.RESULTS_DIR, f"training_history_{model_type}.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training completed! Best Val F1: {best_f1:.4f}. Model saved to {save_path}")
    return model, history, val_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Glaucoma Detection Models")
    parser.add_argument("--model", type=str, default="proposed", choices=["proposed", "baseline_cnn", "efficientnet_b3"])
    parser.add_argument("--dataset", type=str, default="combined", help="Dataset name: combined, acrima, fundus, synthetic")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train_model(
        model_type=args.model,
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
