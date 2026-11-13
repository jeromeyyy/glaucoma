import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_confusion_matrix(cm, class_names=["Normal", "Glaucoma"], save_path="results/confusion_matrix.png"):
    """
    Plots and saves confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    cm_array = np.array(cm)
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"}
    )

    ax.set_title("Confusion Matrix", color="#F8FAFC", fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", color="#94A3B8", fontsize=10)
    ax.set_ylabel("True Class", color="#94A3B8", fontsize=10)
    ax.tick_params(colors="#CBD5E1")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path

def plot_roc_curve(y_true, y_probs, save_path="results/roc_curve.png"):
    """
    Plots and saves ROC curve.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    if y_probs.ndim == 2:
        y_probs = y_probs[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_score = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    ax.plot(fpr, tpr, color="#38BDF8", linewidth=2.5, label=f"Proposed Model (AUC = {roc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="#64748B", linestyle="--", linewidth=1.5, label="Random Chance (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", color="#94A3B8", fontsize=10)
    ax.set_ylabel("True Positive Rate", color="#94A3B8", fontsize=10)
    ax.set_title("Receiver Operating Characteristic (ROC)", color="#F8FAFC", fontsize=12, pad=12, fontweight="bold")
    ax.tick_params(colors="#CBD5E1")
    ax.grid(color="#334155", linestyle=":", alpha=0.7)
    ax.legend(loc="lower right", facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path

def plot_pr_curve(y_true, y_probs, save_path="results/pr_curve.png"):
    """
    Plots and saves Precision-Recall curve.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    if y_probs.ndim == 2:
        y_probs = y_probs[:, 1]

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_score = average_precision_score(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    ax.plot(recall, precision, color="#10B981", linewidth=2.5, label=f"Proposed Model (PR-AUC = {pr_score:.3f})")
    baseline = np.sum(y_true) / len(y_true)
    ax.plot([0, 1], [baseline, baseline], color="#64748B", linestyle="--", linewidth=1.5, label=f"No Skill ({baseline:.2f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall / Sensitivity", color="#94A3B8", fontsize=10)
    ax.set_ylabel("Precision / PPV", color="#94A3B8", fontsize=10)
    ax.set_title("Precision-Recall Curve (PR-AUC)", color="#F8FAFC", fontsize=12, pad=12, fontweight="bold")
    ax.tick_params(colors="#CBD5E1")
    ax.grid(color="#334155", linestyle=":", alpha=0.7)
    ax.legend(loc="lower left", facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path

def plot_training_curves(history, save_path="results/training_curve.png"):
    """
    Plots and saves training & validation loss and accuracy curves.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("#0F172A")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="#CBD5E1")
        ax.grid(color="#334155", linestyle=":", alpha=0.7)

    # Loss plot
    ax1.plot(epochs, history["train_loss"], color="#F59E0B", linewidth=2.0, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], color="#EF4444", linewidth=2.0, label="Val Loss")
    ax1.set_title("Loss Trajectory", color="#F8FAFC", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", color="#94A3B8")
    ax1.set_ylabel("Loss", color="#94A3B8")
    ax1.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    # Accuracy / F1 plot
    ax2.plot(epochs, history["val_acc"], color="#10B981", linewidth=2.0, label="Val Accuracy")
    ax2.plot(epochs, history["val_f1"], color="#38BDF8", linewidth=2.0, label="Val F1-Score")
    ax2.set_title("Validation Performance", color="#F8FAFC", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", color="#94A3B8")
    ax2.set_ylabel("Score", color="#94A3B8")
    ax2.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path
