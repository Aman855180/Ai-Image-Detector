#!/usr/bin/env python3
# train.py
# Author: Aman Kumar
# Date: May 2026
# Description: Main training script for AI Image Detector.
#              Implements two-phase transfer learning:
#              Phase 1: Train classification head (frozen backbone)
#              Phase 2: Fine-tune top EfficientNet layers
#
# Usage:
#   python train.py              # Full two-phase training
#   python train.py --head-only  # Phase 1 only (faster)


import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")

import tensorflow as tf

from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    MODEL_DIR, ASSETS_DIR,
    EPOCHS_FROZEN, EPOCHS_FINETUNE,
    LEARNING_RATE_HEAD, LEARNING_RATE_FINETUNE,
    UNFREEZE_LAYERS
)
from utils.data_utils  import get_data_generators, dataset_stats
from utils.model_utils import (
    build_model, unfreeze_top_layers,
    get_callbacks, load_model
)
from utils.eval_utils  import (
    evaluate_model, plot_training_history,
    plot_confusion_matrix, plot_roc_curve,
    plot_sample_predictions
)


def main(args):
    # Create output directories
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    #  Check GPU availability 
    # I trained on Kaggle T4 GPU which was 10x faster than CPU
    gpus   = tf.config.list_physical_devices("GPU")
    device = f"{len(gpus)} GPU(s)" if gpus else "CPU"
    print(f"[INFO] Training on: {device}")
    if not gpus:
        print("[WARN] No GPU found. Training will be slow on CPU.")
        print("[WARN] Consider using Kaggle or Google Colab for GPU.")

    # Dataset overview 
    print("\n[INFO] Dataset Statistics:")
    stats = dataset_stats()
    for split, counts in stats.items():
        total = sum(counts.values())
        print(
            f"[INFO] {split:5s}: "
            f"AI={counts.get('ai', 0):6d}, "
            f"Real={counts.get('real', 0):6d}, "
            f"Total={total:6d}"
        )

    # Load data 
    print("\n[INFO] Creating data generators...")
    train_gen, val_gen, test_gen = get_data_generators()

    #  Phase 1: Frozen backbone 
    # I freeze EfficientNetB0 completely in Phase 1
    # Only the custom classification head trains
    # This is fast and prevents destroying ImageNet features
    print("\n" + "="*60)
    print("PHASE 1 — Training head with frozen backbone")
    print(f"Epochs: {EPOCHS_FROZEN} | LR: {LEARNING_RATE_HEAD}")
    print("="*60)

    model = build_model(trainable_base=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE_HEAD
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ],
    )
    model.summary()

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FROZEN,
        callbacks=get_callbacks(),
    )

    best_phase1 = max(history1.history["val_accuracy"])
    print(f"\n[INFO] Phase 1 best val_accuracy: {best_phase1:.4f}")

    #  Phase 2: Fine-tuning 
    # I unfreeze top 20 EfficientNet layers with very small LR
    # This adapts higher-level features to our specific task
    # Using 1e-5 (100x smaller) prevents destroying learned features
    if not args.head_only:
        print("\n" + "="*60)
        print("PHASE 2 — Fine-tuning top EfficientNet layers")
        print(f"Epochs: {EPOCHS_FINETUNE} | LR: {LEARNING_RATE_FINETUNE}")
        print(f"Unfreezing top {UNFREEZE_LAYERS} layers")
        print("="*60)

        model = unfreeze_top_layers(model, num_layers=UNFREEZE_LAYERS)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE_FINETUNE
            ),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc")
            ],
        )

        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS_FINETUNE,
            callbacks=get_callbacks(),
        )

        best_phase2 = max(history2.history["val_accuracy"])
        print(f"\n[INFO] Phase 2 best val_accuracy: {best_phase2:.4f}")
        print(f"[INFO] Improvement from Phase 1: "
              f"+{(best_phase2 - best_phase1):.4f}")
    else:
        history2 = None
        print("\n[INFO] Skipping Phase 2 (--head-only flag set)")

    #  Save training plots 
    print("\n[INFO] Saving training plots...")
    plot_training_history(
        history1, history2,
        save_path=os.path.join(ASSETS_DIR, "training_history.png")
    )

    # Final evaluation on test set
    # Load best saved model for evaluation
    # This ensures we evaluate the best checkpoint
    # not necessarily the last epoch
    print("\n[INFO] Loading best saved model for evaluation...")
    best_model = load_model()

    print("[INFO] Evaluating on test set...")
    metrics = evaluate_model(best_model, test_gen)

    # Save evaluation plots
    plot_confusion_matrix(
        metrics["y_true"], metrics["y_pred"],
        save_path=os.path.join(ASSETS_DIR, "confusion_matrix.png")
    )
    plot_roc_curve(
        metrics["y_true"], metrics["y_prob"], metrics["auc"],
        save_path=os.path.join(ASSETS_DIR, "roc_curve.png")
    )
    plot_sample_predictions(
        best_model, test_gen,
        save_path=os.path.join(ASSETS_DIR, "sample_predictions.png")
    )

    #  Final summary 
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"[DONE] Test Accuracy : {metrics['accuracy']:.4f} "
          f"({metrics['accuracy']*100:.2f}%)")
    print(f"[DONE] Test AUC-ROC  : {metrics['auc']:.4f}")
    print(f"[DONE] Model saved   : model/ai_image_detector.keras")
    print(f"[DONE] Plots saved   : {ASSETS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AI Image Detector — by Aman Kumar"
    )
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="Run Phase 1 only — faster but lower accuracy"
    )
    args = parser.parse_args()
    main(args)