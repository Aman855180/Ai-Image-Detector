# Ai-Image-Detector
A deep learning web application that detects whether an uploaded
image is **AI-generated** or a **real photograph** using
EfficientNetB0 transfer learning.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Accuracy](https://img.shields.io/badge/Accuracy-87.40%25-brightgreen)

---

##  Problem Statement

With the rise of AI image generators like Midjourney, DALL-E and
Stable Diffusion, it has become very difficult to tell if an image
was created by AI or captured by a camera.

I built this project to solve this problem using deep learning.

---

##  My Approach

I chose **EfficientNetB0** for transfer learning because:
- Compound scaling gives best accuracy per parameter
- Only 5.3M parameters — fast inference on CPU
- Strong ImageNet features transfer well to this task
- Native 224×224 input

I implemented **two-phase training**:
- Phase 1: Train classification head (backbone frozen)
- Phase 2: Fine-tune top 20 EfficientNet layers

---

##  Results

| Metric | Score |
|---|---|
| Test Accuracy | 87.40% |
| AUC-ROC | 96.45% |
| Training Images | 32,000 |
| Inference Time | < 1 second |

---

##  Architecture
```
Input Image (any resolution)
↓
Resize → 224 × 224 × 3
↓
EfficientNetB0 (ImageNet pretrained)
Phase 1: All layers frozen
Phase 2: Top 20 layers unfrozen
↓
GlobalAveragePooling2D
↓
BatchNormalization
↓
Dense(256, ReLU) → Dropout(0.4)
↓
Dense(1, Sigmoid)
↓
< 0.6 → AI-Generated

0.6 → Real Photo
```
---

##  Project Structure
```
AI_detector/
├── app.py              # Streamlit web application
├── train.py            # Two-phase training pipeline
├── evaluate.py         # Standalone evaluation script
├── analyze.py          # My custom failure analysis tool
├── config.py           # Central configuration
├── requirements.txt    # Dependencies
├── README.md           # This file
│
├── utils/
│   ├── data_utils.py   # Data loading and augmentation
│   ├── model_utils.py  # Model architecture and inference
│   └── eval_utils.py   # Metrics and visualization
│
├── model/
│   └── ai_image_detector.keras   # Trained model
│
└── assets/             # Training plots and charts
├── training_history.png
├── confusion_matrix.png
├── roc_curve.png
└── failure_analysis.png
```
---

##  Dataset

**CIFAKE** (Kaggle)
- 50,000 total images
- 25,000 AI-generated (Stable Diffusion v1.4)
- 25,000 real photos (CIFAR-10)

**My split:**
- Train: 32,000 (64%)
- Val: 8,000 (16%)
- Test: 10,000 (20%)

---

##  How to Run

### 1. Clone repository
```bash
git clone https://github.com/amankumar2107/ai-image-detector
cd ai-image-detector
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Mac/Linux
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

## 4. Model Weights

The trained EfficientNetB0 `.keras` model is included in the repository and automatically loaded by the Streamlit application during inference.

### 5. Run web app
```
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

##  Training Your Own Model

```
# Full two-phase training
python train.py

# Phase 1 only (faster)
python train.py --head-only

# Evaluate on test set
python evaluate.py

# Analyze failure cases
python analyze.py
```

---

##  Limitations I Found

During testing I discovered:

1. **Sky/gradient images** often misclassified as AI-Generated
   because smooth color gradients resemble AI patterns

2. **High-resolution real photos** sometimes confused the model
   because CIFAKE uses small 32×32 upscaled images

3. **New AI generators** like Midjourney v6 and DALL-E 3 were
   not in training data so may fool the detector

I raised the confidence threshold from 0.60 to 0.75 to reduce
false positives after testing on 50+ real world images.

---

##  My Future Improvement Plans

| Improvement | Expected Impact |
|---|---|
| Train on GenImage (1.3M high-res images) | Fix sky/gradient problem |
| Add Grad-CAM heatmap visualization | Explain model decisions |
| Ensemble CNN + frequency analysis | Push accuracy above 92% |
| Fine-tune on Midjourney/DALL-E images | Catch latest generators |
| Export to ONNX format | 3x faster CPU inference |

---

##  Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow, Keras |
| Model | EfficientNetB0 |
| Web App | Streamlit |
| Data | NumPy, Pillow, OpenCV |
| Evaluation | Scikit-learn, Matplotlib, Seaborn |
| Training | Kaggle GPU T4 |

---

##  What I Learned

- Transfer learning is essential for small datasets
- Two-phase training improves accuracy significantly
- Data augmentation reduces overfitting
- Model evaluation goes beyond just accuracy
- Importance of analyzing failure cases

---

*Built by **Aman Kumar** | AI Image Detector*
