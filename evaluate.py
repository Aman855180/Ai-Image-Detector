#!/usr/bin/env python3
# evaluate.py
# Author: Aman Kumar
# Date: May 2026
# Description: Standalone evaluation script for AI Image Detector.
#              I use this to test model performance on test set
#              without retraining. Useful for comparing different
#              model versions.
#
# Usage:
#   python evaluate.py
#   python evaluate.py --model model/ai_image_detector.keras


import os
import argparse

from config import MODEL_PATH, TEST_DIR, ASSETS_DIR
from utils.data_utils  import get_data_generators
from utils.model_utils import load_model
from utils.eval_utils  import (
    evaluate_model, plot_confusion_matrix,
    plot_roc_curve, plot_sample_predictions
)


def main(args):
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # Load model from specified path or default path
    model_path = args.model or MODEL_PATH
    print(f"[INFO] Loading model from: {model_path}")
    model = load_model(model_path)

    # Load only test generator
    # I only need test set for final evaluation
    print("[INFO] Loading test data...")
    _, _, test_gen = get_data_generators()

    print(f"[INFO] Test images: {test_gen.samples}")
    print(f"[INFO] Classes: {test_gen.class_indices}")

    # Run evaluation
    print("\n[INFO] Running evaluation on test set...")
    metrics = evaluate_model(model, test_gen)

    # Save evaluation charts
    print("\n[INFO] Saving evaluation charts...")
    plot_confusion_matrix(
        metrics["y_true"], metrics["y_pred"],
        save_path=os.path.join(ASSETS_DIR, "confusion_matrix.png")
    )
    plot_roc_curve(
        metrics["y_true"], metrics["y_prob"], metrics["auc"],
        save_path=os.path.join(ASSETS_DIR, "roc_curve.png")
    )
    plot_sample_predictions(
        model, test_gen, n=8,
        save_path=os.path.join(ASSETS_DIR, "sample_predictions.png")
    )

    # Final summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy : {metrics['accuracy']:.4f} "
          f"({metrics['accuracy']*100:.2f}%)")
    print(f"AUC-ROC  : {metrics['auc']:.4f}")
    print(f"Charts   : {ASSETS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate AI Image Detector — by Aman Kumar"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to .keras model file (default: from config.py)"
    )
    main(parser.parse_args())