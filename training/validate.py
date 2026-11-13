import torch
import torch.nn.functional as F
from evaluation.metrics import compute_classification_metrics

def validate_epoch(model, val_loader, criterion=None, device="cpu"):
    """
    Evaluates model on validation set.
    Returns:
        metrics: dict of accuracy, precision, recall, f1, auc, confusion matrix
        avg_loss: float
    """
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            images, targets, _ = batch
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
                if criterion is not None:
                    loss, _ = criterion(outputs, targets)
                    total_loss += loss.item() * images.size(0)
            else:
                logits = outputs
                if criterion is not None:
                    loss = F.cross_entropy(logits, targets)
                    total_loss += loss.item() * images.size(0)

            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    n_samples = max(len(val_loader.dataset), 1)
    avg_loss = total_loss / n_samples
    metrics = compute_classification_metrics(all_targets, all_preds, all_probs)
    metrics["val_loss"] = round(avg_loss, 4)

    return metrics, all_targets, all_preds, all_probs
