# config.py
# Author: Aman Kumar
# Project: AI Image Detector
# Description: I built this project to detect whether an image
#              is AI-generated or a real photograph using
#              deep learning and transfer learning.
#              This file contains all central configurations
#              so I can change settings from one place.


import os

# Project Identity 
# I named it AI_Image_Detector because it clearly describes
# what the project does
PROJECT_NAME    = "AI_Image_Detector"
PROJECT_VERSION = "1.0.0"
AUTHOR          = "Aman Kumar"

#  Paths 
# Using os.path.join maked paths work on both Windows and Linux
# This was important when I shifted from PC to Kaggle/Colab
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOG_DIR    = os.path.join(BASE_DIR, "logs")

TRAIN_DIR  = os.path.join(DATA_DIR, "train")
VAL_DIR    = os.path.join(DATA_DIR, "val")
TEST_DIR   = os.path.join(DATA_DIR, "test")

# Model filename includes version so I can track experiments
MODEL_PATH = os.path.join(MODEL_DIR, "ai_image_detector.keras")

#  Image Settings 
# 224x224 is EfficientNetB0 native input size
# Using any other size would require resizing internally
IMG_SIZE  = (224, 224)
IMG_SHAPE = (224, 224, 3)   # 3 = RGB channels

# I experimented with batch sizes 16, 32 and 64
# Batch 16 = slow training
# Batch 64 = ran out of memory on Kaggle
# Batch 32 = best balance of speed and memory
BATCH_SIZE = 32

#  Class Configuration
# Folder names must match class names exactly
# ai/ folder = label 0
# real/ folder = label 1
CLASS_NAMES  = ["ai", "real"]
CLASS_LABELS = {
    0: "AI-Generated",
    1: "Real Photo"
}

#  Training Hyperparameters

# Phase 1: Only classification head trains
# EfficientNetB0 backbone stays completely frozen
# This preserves ImageNet features learned on 1.2M images
EPOCHS_FROZEN      = 10
LEARNING_RATE_HEAD = 1e-3   # Standard Adam learning rate

# Phase 2: Fine tuning
# I unfreeze top 20 layers of EfficientNetB0
# I tested 10, 20, 30 layers:
# 10 layers = not enough improvement
# 30 layers = overfitting on validation set
# 20 layers = best validation accuracy (my choice)
EPOCHS_FINETUNE        = 10
LEARNING_RATE_FINETUNE = 1e-5   # 100x smaller than Phase 1
UNFREEZE_LAYERS        = 20     # Top 20 EfficientNet layers

#  Regularization
# Dropout randomly disables neurons during training
# This forces model to learn redundant representations
# I tested dropout rates 0.3, 0.4, 0.5:
# 0.3 = slight overfitting (train >> val accuracy)
# 0.5 = underfitting (model too restricted)
# 0.4 = best balance (my final choice)
DROPOUT_RATE = 0.4

#  Augmentation 
# Applied only to training data to create artificial variety
# This helps model generalize to unseen images
# Validation and test sets are never augmented
AUGMENTATION = dict(
   
    rotation_range     = 15,                    # Small rotations simulate tilted camera angles

   
    width_shift_range  = 0.1,                   # Slight shifts simulate off-center subjects
    height_shift_range = 0.1,

   
    horizontal_flip    = True,                  # Flip is safe for this task (AI/real label doesn't change)

    
    zoom_range         = 0.1,                   # Small zoom simulates different distances

    # I added brightness range after noticing my model
    # was struggling with very dark and very bright images
    # during initial testing phase
    brightness_range   = [0.8, 1.2],

    fill_mode          = "nearest",
)

#  Inference Settings 
# During testing I noticed model was incorrectly classifying
# sky images and smooth gradient photos as AI-generated
# This is because AI images often have smooth color gradients
# similar to natural sky photos
#
# To reduce these false positives I raised threshold from
# default 0.60 to 0.75 after testing on 50 real photos
# Now model needs to be 75% confident before predicting
CONFIDENCE_THRESHOLD = 0.75 