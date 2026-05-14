# analyze.py
# Author: Aman Kumar
# Description: My custom analysis script to understand
#              model predictions and failure cases.
#              I wrote this after noticing model was
#              misclassifying sky images.

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_PATH, IMG_SIZE, CLASS_LABELS
from utils.model_utils import load_model, preprocess_image


def find_failure_cases(model, test_dir, n=10):
    """
    I wrote this function to understand WHERE my model fails.
    This helped me identify the sky image problem.
    """
    failures = {"false_ai": [], "false_real": []}

    for cls_idx, cls_name in enumerate(["ai", "real"]):
        cls_dir = os.path.join(test_dir, cls_name)
        for img_file in os.listdir(cls_dir)[:50]:
            img_path = os.path.join(cls_dir, img_file)
            try:
                img  = Image.open(img_path)
                arr  = preprocess_image(img)
                prob = float(model.predict(arr, verbose=0)[0][0])
                pred = 1 if prob >= 0.60 else 0     #changed this after changing in models.utils earlier to many false ai        predictions

                if cls_idx == 0 and pred == 1:
                    failures["false_real"].append({
                        "path": img_path,
                        "confidence": prob
                    })
                elif cls_idx == 1 and pred == 0:
                    failures["false_ai"].append({
                        "path": img_path,
                        "confidence": 1 - prob
                    })
            except Exception:
                continue

    return failures


def plot_failures(failures, save_path="assets/failures.png"):
    """Plot failure cases for analysis."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Model Failure Cases", fontsize=14)

    for i, case in enumerate(failures["false_ai"][:5]):
        img = Image.open(case["path"])
        axes[0][i].imshow(img)
        axes[0][i].set_title(
            f"Real→AI\n{case['confidence']:.1%}", fontsize=8)
        axes[0][i].axis("off")

    for i, case in enumerate(failures["false_real"][:5]):
        img = Image.open(case["path"])
        axes[1][i].imshow(img)
        axes[1][i].set_title(
            f"AI→Real\n{case['confidence']:.1%}", fontsize=8)
        axes[1][i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")
    return fig


if __name__ == "__main__":
    from config import TEST_DIR
    model = load_model()
    failures = find_failure_cases(model, TEST_DIR)
    plot_failures(failures)
    print(f"False AI predictions: {len(failures['false_ai'])}")
    print(f"False Real predictions: {len(failures['false_real'])}")