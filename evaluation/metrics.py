import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score
)

def compute_classification_metrics(y_true, y_pred, y_probs=None):
    """
    Computes genuine standard clinical and statistical evaluation metrics:
    Accuracy, Precision, Recall, Sensitivity, Specificity, F1, ROC-AUC, PR-AUC, and Confusion Matrix.
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    # Clinical Sensitivity and Specificity
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC-AUC and PR-AUC calculation
    auc = 0.5
    pr_auc = 0.5
    if y_probs is not None:
        y_probs = np.array(y_probs)
        # If 2D probabilities [N, 2], extract positive class probability
        if y_probs.ndim == 2 and y_probs.shape[1] == 2:
            glaucoma_probs = y_probs[:, 1]
        else:
            glaucoma_probs = y_probs

        if len(np.unique(y_true)) > 1:
            try:
                auc = float(roc_auc_score(y_true, glaucoma_probs))
            except Exception:
                auc = 0.5
            try:
                pr_auc = float(average_precision_score(y_true, glaucoma_probs))
            except Exception:
                pr_auc = 0.5

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total_samples": int(len(y_true))
    }
