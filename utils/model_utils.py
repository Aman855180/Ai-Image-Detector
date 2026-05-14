# model_utils.py
# Author: Aman Kumar
# Description: Model architecture, training helpers and inference utilities
#              for AI Image Detector project.
#              I built this to detect AI generated vs real photographs
#              using EfficientNetB0 transfer learning.


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    IMG_SHAPE, IMG_SIZE, MODEL_PATH,
    LEARNING_RATE_HEAD, LEARNING_RATE_FINETUNE,
    DROPOUT_RATE, EPOCHS_FROZEN, EPOCHS_FINETUNE,
    CLASS_LABELS, CONFIDENCE_THRESHOLD, UNFREEZE_LAYERS
)



# 1. Model Architecture


def build_model(input_shape=IMG_SHAPE, trainable_base=False):
    """
    Binary image classifier using EfficientNetB0 as backbone.

    Why I chose EfficientNetB0:
    - I compared ResNet50, MobileNetV2 and EfficientNetB0
    - EfficientNetB0 gave best accuracy with least parameters
    - Only 5.3M parameters means fast inference on CPU
    - Compound scaling balances depth, width and resolution
    - Pre-trained on ImageNet (1.2M images, 1000 categories)

    My Architecture Decision:
    - GlobalAveragePooling instead of Flatten
      because it reduces overfitting on small datasets
    - BatchNormalization stabilizes training
    - Dense(256) after testing 128, 256, 512
    - Dropout(0.4) after testing 0.3, 0.4, 0.5
    - Single sigmoid output for binary classification

    Parameters:
    input_shape    : image dimensions, default (224, 224, 3)
    trainable_base : False in Phase 1, True in Phase 2
    """

    # Load EfficientNetB0 with ImageNet weights
    # include_top=False removes the 1000-class ImageNet head
    # We replace it with our own binary classification head
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    # Freeze backbone during Phase 1
    # This preserves valuable ImageNet learned features
    base_model.trainable = trainable_base

    inputs = tf.keras.Input(shape=input_shape)

    # Important: EfficientNetB0 has built-in preprocessing
    # I do NOT manually rescale pixels (no divide by 255)
    # Passing raw 0-255 values is correct for this model
    x = base_model(inputs, training=trainable_base)

    # Convert 7x7x1280 feature maps to 1280-dim vector
    x = layers.GlobalAveragePooling2D()(x)

    # Normalize activations for stable training
    x = layers.BatchNormalization()(x)

    # Classification head - learns task specific patterns
    x = layers.Dense(256, activation="relu")(x)

    # Randomly disable 40% neurons to prevent memorization
    x = layers.Dropout(DROPOUT_RATE)(x)

    # Output: probability of image being Real (label 1)
    # Values > 0.5 = Real Photo
    # Values < 0.5 = AI Generated
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="AI_Image_Detector_v1")
    return model


def unfreeze_top_layers(model, num_layers=UNFREEZE_LAYERS):
    """
    Unfreeze top layers of EfficientNetB0 for Phase 2 fine tuning.

    My reasoning for unfreezing only top 20 layers:
    - Bottom layers detect basic features like edges and colors
      These are universal and useful for any image task
      Changing them would hurt model performance

    - Top layers detect complex task-specific patterns
      These need to adapt to our AI vs Real detection task
      Fine tuning these with very small LR improves accuracy

    - I tested unfreezing 10, 20, 30 layers
      10 layers = not enough improvement over Phase 1
      30 layers = overfitting on validation set
      20 layers = best validation accuracy (my final choice)

    Phase 2 uses learning rate 1e-5 (100x smaller than Phase 1)
    This gently adjusts weights without destroying
    the ImageNet knowledge already learned.
    """
    base_model = model.layers[2]
    base_model.trainable = True

    # Keep all layers frozen except top num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    frozen   = sum(not l.trainable for l in base_model.layers)
    unfrozen = sum(l.trainable for l in base_model.layers)
    print(f"[INFO] Frozen layers  : {frozen}")
    print(f"[INFO] Unfrozen layers: {unfrozen}")
    return model



# 2. Callbacks


def get_callbacks(model_path=MODEL_PATH, log_dir="logs"):
    """
    Callbacks I use for robust training:

    ModelCheckpoint:
    - Saves model only when val_accuracy improves
    - So I always have the best version saved
    - I monitor val_accuracy not val_loss because
      accuracy is more meaningful for this task

    EarlyStopping:
    - Stops training if no improvement for 5 epochs
    - Prevents wasting time and overfitting
    - Restores best weights automatically

    ReduceLROnPlateau:
    - Halves learning rate when val_loss stops improving
    - Helps model find better minima
    - Minimum LR set to 1e-7 to prevent too small updates
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return [
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]



# 3. Inference


def load_model(model_path=MODEL_PATH):
    """
    Load trained model from disk.
    Raises clear error if model file not found.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            "Please train the model first using train.py"
        )
    print(f"[INFO] Loading model: {model_path}")
    return tf.keras.models.load_model(model_path)


def preprocess_image(image, img_size=IMG_SIZE):
    """
    Prepare PIL image for model inference.

    Steps I follow:
    1. Convert to RGB
       Handles any format: RGBA, grayscale, palette etc.
       Model expects exactly 3 channels

    2. Resize to 224x224 using LANCZOS
       LANCZOS gives best quality for downsampling
       Avoids pixelation or blurring artifacts

    3. Convert to float32 numpy array
       TensorFlow works with float32

    4. Add batch dimension
       Model expects shape (batch, H, W, C)
       Single image becomes (1, 224, 224, 3)

    Important: I do NOT divide by 255
    EfficientNetB0 expects raw pixel values 0-255
    It handles normalization internally
    """
    image = image.convert("RGB")
    image = image.resize(img_size, Image.LANCZOS)
    arr   = np.array(image, dtype=np.float32)
    arr   = np.expand_dims(arr, axis=0)
    return arr


def predict(model, image):
    """
    Run prediction on a PIL image.

    I added risk_level to my predict function because
    during user testing I noticed people were confused
    by raw confidence percentages like 0.73 or 0.81.

    Risk levels make predictions more understandable:
    Very High = model is very sure
    High      = model is confident
    Medium    = model has some doubt
    Low       = treat prediction with caution

    I also return ai_prob and real_prob separately
    so the app can show a proper probability breakdown
    to the user.

    Returns:
    label      : AI-Generated or Real Photo
    confidence : how confident model is (0 to 1)
    raw_score  : raw sigmoid output (P of being Real)
    uncertain  : True if confidence below threshold
    risk_level : my custom confidence description
    ai_prob    : probability of being AI generated
    real_prob  : probability of being real photo
    """
    arr       = preprocess_image(image)
    raw_score = float(model.predict(arr, verbose=0)[0][0])

    # raw_score = probability of being Real Photo
    # 1-raw_score = probability of being AI Generated
    DECISION_THRESHOLD = 0.60
    #if raw_score >= 0.5 changed this as many real images were being predicted as ai when I tried
    if raw_score >= DECISION_THRESHOLD:                                                                      
        label      = CLASS_LABELS[1]   # Real Photo
        confidence = raw_score
    else:
        label      = CLASS_LABELS[0]   # AI-Generated
        confidence = 1.0 - raw_score

    # My custom risk levels based on confidence score
    # I designed these thresholds based on testing
    # on 50+ images from both classes
    
    if confidence >= 0.90:
        risk_level = "Very High Confidence"
    elif confidence >= 0.75:
        risk_level = "High Confidence"
    elif confidence >= 0.60:
        risk_level = "Medium Confidence"
    else:
        risk_level = "Low Confidence — Treat with caution"

    return {
        "label"      : label,
        "confidence" : confidence,
        "raw_score"  : raw_score,
        "uncertain"  : confidence < CONFIDENCE_THRESHOLD,
        "risk_level" : risk_level,
        "ai_prob"    : 1.0 - raw_score,
        "real_prob"  : raw_score,
    }