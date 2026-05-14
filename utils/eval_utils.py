
# eval_utils.py
# Author: Aman Kumar
# Date: May 2026
# Description: Evaluation, plotting and reporting utilities for
#              AI Image Detector project.
#              I wrote these functions to visualize and understand
#              how well my model performs on unseen test data.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLASS_NAMES, ASSETS_DIR

# 1. Model Evaluation


def evaluate_model(model, test_gen):
    """
    Evaluate model on complete test set and return metrics.

    I always evaluate on test set only ONCE after training
    is completely finished. Using test set during training
    would give overly optimistic results.

    Metrics I track:
    - Accuracy: overall correct predictions
    - AUC-ROC: ability to separate AI vs Real
    - Precision: when model says AI, how often correct
    - Recall: out of all AI images, how many detected
    - F1-Score: balance of precision and recall

    Returns dict with all metrics and raw predictions
    """
    test_gen.reset()

    # Get predictions for all test images
    y_prob = model.predict(test_gen, verbose=1).flatten()

    # Convert probabilities to binary predictions
    # threshold 0.5: above = Real, below = AI Generated
    y_pred = (y_prob >= 0.6).astype(int)
    y_true = test_gen.classes

    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)

    report = classification_report(
        y_true, y_pred,
        target_names=["AI-Generated", "Real Photo"],
        output_dict=True,
    )

    print(f"\n[EVAL] Accuracy : {acc:.4f}")
    print(f"[EVAL] AUC-ROC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["AI-Generated", "Real Photo"]
    ))

    return dict(
        accuracy=acc,
        auc=auc,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        report=report,
    )


# 2. Training History Plot


def plot_training_history(history1, history2=None, save_path=None):
    """
    Plot accuracy and loss curves for both training phases.

    I added a vertical line to show where Phase 2 starts
    so I can visually see the impact of fine-tuning.

    Blue line = training data
    Red line  = validation data

    Gap between blue and red = overfitting
    Both lines going up = good learning
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Training History — AI Image Detector",
        fontsize=15, fontweight="bold"
    )

    def _merge(h1, h2, key):
        vals = h1.history[key]
        if h2:
            vals = vals + h2.history[key]
        return vals

    for ax, metric, title in zip(
        axes,
        [("accuracy", "val_accuracy"), ("loss", "val_loss")],
        ["Accuracy", "Loss"],
    ):
        train_vals = _merge(history1, history2, metric[0])
        val_vals   = _merge(history1, history2, metric[1])
        epochs     = range(1, len(train_vals) + 1)

        ax.plot(epochs, train_vals, "b-o",
                markersize=4, label="Train")
        ax.plot(epochs, val_vals, "r-o",
                markersize=4, label="Validation")

        # Show where Phase 2 fine-tuning starts
        if history2:
            split_epoch = len(history1.history[metric[0]]) + 0.5
            ax.axvline(
                split_epoch, color="grey",
                linestyle="--", alpha=0.7,
                label="Fine-tuning starts"
            )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Training history saved: {save_path}")
    return fig



# 3. Confusion Matrix


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Visualize confusion matrix as heatmap.

    I use this to understand what types of errors
    my model makes:

    Top-left  (0,0): AI images correctly identified
    Top-right (0,1): AI images wrongly called Real
    Bottom-left(1,0): Real images wrongly called AI
    Bottom-right(1,1): Real images correctly identified

    During testing I noticed bottom-left was high
    meaning model was calling real photos as AI.
    This led me to investigate the sky image problem.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["AI-Generated", "Real Photo"],
        yticklabels=["AI-Generated", "Real Photo"],
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(
        "Confusion Matrix",
        fontsize=14, fontweight="bold", pad=12
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Confusion matrix saved: {save_path}")
    return fig



# 4. ROC Curve


def plot_roc_curve(y_true, y_prob, auc_score, save_path=None):
    """
    Plot ROC curve showing model discrimination ability.

    ROC curve shows tradeoff between:
    - True Positive Rate (correctly detected AI)
    - False Positive Rate (real photos called AI)

    AUC = area under this curve
    AUC = 0.5 means random guessing
    AUC = 1.0 means perfect classifier
    Our AUC = 0.9645 is excellent!

    The blue shaded area represents the AUC.
    Larger area = better model.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, "b-", lw=2,
            label=f"Model (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "r--", lw=1.5,
            label="Random Classifier (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.08, color="blue")

    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] ROC curve saved: {save_path}")
    return fig



# 5. Sample Predictions


def plot_sample_predictions(model, test_gen, n=8, save_path=None):
    """
    Show sample predictions with green/red borders.

    Green border = correct prediction
    Red border   = wrong prediction

    I added this visualization to quickly spot
    patterns in model failures during testing.
    For example I noticed red borders appeared
    more on sky and landscape images.
    """
    test_gen.reset()
    images, labels = next(test_gen)
    probs = model.predict(images, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)

    label_map = {0: "AI-Gen", 1: "Real"}
    n    = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 3.5, rows * 3.5)
    )
    axes = axes.flatten()

    for i in range(n):
        img     = images[i].astype(np.uint8)
        true    = int(labels[i])
        pred    = int(preds[i])
        conf    = probs[i] if pred == 1 else 1 - probs[i]
        correct = (true == pred)

        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {label_map[true]}\n"
            f"Pred: {label_map[pred]} ({conf:.0%})",
            fontsize=8,
            color="green" if correct else "red",
        )
        for spine in axes[i].spines.values():
            spine.set_edgecolor("green" if correct else "red")
            spine.set_linewidth(3)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Sample Predictions — Green=Correct, Red=Wrong",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Sample predictions saved: {save_path}")
    return fig