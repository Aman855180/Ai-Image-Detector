
# app.py
# Author: Aman Kumar
# Date: May 2026
# Description: Streamlit web application for AI Image Detector.
#              Users can upload any image and get instant prediction
#              of whether it is AI-generated or a real photograph.
#
# Usage:
#   streamlit run app.py


import os
import time
import numpy as np
import streamlit as st
from PIL import Image

# Page config must be first Streamlit call
st.set_page_config(
    page_title="AI Image Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 60%, #ff0099 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 2rem;
    font-family: 'Space Mono', monospace;
}
.result-card {
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-top: 1.2rem;
    border: 2px solid transparent;
}
.result-ai {
    background: linear-gradient(135deg, #0f0c29, #1a0533);
    border-color: #7b2ff7;
    box-shadow: 0 0 30px rgba(123,47,247,0.35);
}
.result-real {
    background: linear-gradient(135deg, #0c2029, #003333);
    border-color: #00d4ff;
    box-shadow: 0 0 30px rgba(0,212,255,0.30);
}
.result-uncertain {
    background: linear-gradient(135deg, #1a1a0c, #2a2000);
    border-color: #f59e0b;
    box-shadow: 0 0 30px rgba(245,158,11,0.30);
}
.result-label {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.confidence-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    padding: 3px 10px;
    border-radius: 20px;
    background: rgba(255,255,255,0.08);
    display: inline-block;
    margin-top: 6px;
}
.metric-box {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 800;
    font-family: 'Space Mono', monospace;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}
div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #00d4ff, #7b2ff7) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] {
    background: #0b0d14 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
</style>
""", unsafe_allow_html=True)


#  Model loading 
@st.cache_resource(show_spinner=False)
def get_model():
    """
    Load trained model. Cached so it loads only once per session.
    I use @st.cache_resource instead of @st.cache_data because
    the model object should be shared across all users.
    """
    from config import MODEL_PATH
    from utils.model_utils import load_model
    try:
        return load_model(MODEL_PATH)
    except FileNotFoundError:
        return None


#  Sidebar 
with st.sidebar:
    st.markdown("## 👨‍💻 About")
    st.markdown("""
    AI Image Detector Project

    I built this to detect whether an image
    was created by AI or captured by a camera
    using deep learning.

    ---
    **How it works:**
    1. Upload any image
    2. Model resizes to 224×224
    3. EfficientNetB0 extracts features
    4. Classification head predicts
    5. Result shown with confidence

    ---
    **Model Details**
    - Backbone: EfficientNetB0
    - Framework: TensorFlow/Keras
    - Dataset: CIFAKE (50,000 images)
    - Test Accuracy: 87.40%
    - AUC-ROC: 96.45%
    - Training: Two-phase transfer learning

    ---
    **Known Limitations**
    - Sky/gradient images sometimes misclassified
    - High-res real photos may confuse model
    - New AI generators not in training data

    ---
    """)
    st.markdown("💻 [GitHub](https://github.com/Aman855180)")


#  Main Page 
st.markdown(
    '<div class="hero-title">AI Image Detector</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="hero-subtitle">'
    '→ Upload any image to detect if it was made by AI or a camera.'
    '</div>',
    unsafe_allow_html=True
)

# Load model
with st.spinner("Loading model..."):
    model = get_model()

if model is None:
    st.error(
        "⚠️ Model not found. "
        "Please ensure `model/ai_image_detector.keras` exists."
    )
    st.stop()

# Upload widget 
uploaded = st.file_uploader(
    "Drop an image here or click to browse",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Supported formats: JPG, PNG, WebP, BMP",
)

if uploaded is not None:
    image = Image.open(uploaded)
    col_img, col_result = st.columns([1, 1], gap="large")

    #  Display uploaded image 
    with col_img:
        st.markdown("#### 🖼️ Uploaded Image")
        st.image(image, use_container_width=True, caption=uploaded.name)
        w, h = image.size
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-value">{w} × {h}</div>'
            f'<div class="metric-label">Image Size (pixels)</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    #  Run prediction 
    with col_result:
        st.markdown("#### 🔍 Analysis Result")

        with st.spinner("Analysing image..."):
            time.sleep(0.3)
            from utils.model_utils import predict
            result = predict(model, image)

        label      = result["label"]
        confidence = result["confidence"]
        raw_score  = result["raw_score"]
        uncertain  = result["uncertain"]
        risk_level = result["risk_level"]
        ai_prob    = result["ai_prob"]
        real_prob  = result["real_prob"]

        #  Result card 
        if uncertain:
            card_class = "result-uncertain"
            emoji      = "🤔"
            verdict    = "Uncertain"
        elif label == "AI-Generated":
            card_class = "result-ai"
            emoji      = "🤖"
            verdict    = label
        else:
            card_class = "result-real"
            emoji      = "📷"
            verdict    = label

        st.markdown(
            f'<div class="result-card {card_class}">'
            f'<div class="result-label">{emoji} {verdict}</div>'
            f'<div class="confidence-tag">'
            f'Confidence: {confidence:.1%}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        #  Risk level 
        st.markdown(f"**{risk_level}**")

        # Confidence bar 
        st.markdown("**Confidence Score**")
        st.progress(confidence)

        # Probability breakdown 
        st.markdown("**Probability Breakdown**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value" style="color:#7b2ff7">'
                f'{ai_prob:.1%}</div>'
                f'<div class="metric-label">AI-Generated</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value" style="color:#00d4ff">'
                f'{real_prob:.1%}</div>'
                f'<div class="metric-label">Real Photo</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        #  Uncertainty warning 
        if uncertain:
            st.warning(
                "⚠️ Model confidence is below 75%. "
                "This can happen with sky/gradient images, "
                "heavily edited photos, or unusual styles. "
                "Treat this result with caution."
            )

        #  Technical details 
        with st.expander(" Technical Details"):
            st.markdown(f"""
            | Property | Value |
            |---|---|
            | Raw sigmoid score | `{raw_score:.6f}` |
            | Decision threshold | `0.6` |
            | Confidence threshold | `0.75` |
            | Model input size | `224 × 224 × 3` |
            | Backbone | EfficientNetB0 |
            | Predicted class | **{label}** |
            | Risk level | **{risk_level}** |
            """)

#  Homepage when no image uploaded 
if uploaded is None:
    st.markdown("---")
    st.markdown("###  Model Performance")

    c1, c2, c3, c4 = st.columns(4)
    metrics_display = [
        ("87.40%",        "Test Accuracy"),
        ("96.45%",        "AUC-ROC"),
        ("50,000",        "Training Images"),
        ("EfficientNetB0","Backbone"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], metrics_display):
        with col:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("####  My Architecture")
        st.markdown("""
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
        > 0.6 → Real Photo
    ```
    
    """)

    with col_r:
        st.markdown("####  Dataset I Used")
        st.markdown("""
        **CIFAKE Dataset (Kaggle)**
        - 50,000 images total
        - 25,000 AI-generated (Stable Diffusion v1.4)
        - 25,000 real photos (CIFAR-10)

        **My Train/Val/Test Split**
        - Train:      32,000 images (64%)
        - Validation:  8,000 images (16%)
        - Test:       10,000 images (20%)

        **Augmentation I Applied**
        - Random horizontal flip
        - ±15° rotation
        - ±10% zoom
        - ±20% brightness variation
        """)

    st.markdown("####  My Future Improvement Plans")
    st.markdown("""
    | Improvement | Expected Impact |
    |---|---|
    | Train on GenImage dataset (1.3M high-res) | Fix sky/gradient misclassification |
    | Add Grad-CAM heatmap visualization | Show WHY model made decision |
    | Ensemble CNN + frequency analysis | Push accuracy above 92% |
    | Fine-tune on Midjourney/DALL-E images | Detect latest AI generators |
    | Export to ONNX format | 3x faster CPU inference |
    """)

    st.markdown("---")
    st.markdown(
        "*Built by **Aman Kumar** | "
        "AI Image Detector "
    )