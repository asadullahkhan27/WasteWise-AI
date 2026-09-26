# ============================================================
# WASTEWISE AI
# AI-Powered Waste Classification & Recycling Assistant
# ============================================================

import base64
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WasteWise AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "wastewise_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"

ICODEGURU_LOGO = BASE_DIR / "assets" / "icodeguru_logo.png"
SCHOOL_LOGO = BASE_DIR / "assets" / "school_logo.png"

IMAGE_SIZE = (224, 224)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# CUSTOM CSS
# ALL CSS IS INSIDE APP.PY
# ============================================================

CUSTOM_CSS = """
<style>

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(34, 197, 94, 0.10),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(16, 185, 129, 0.08),
            transparent 30%
        ),
        #07130f;
    color: #f3fdf7;
}

/* Main container */

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #081a14 0%,
        #06120f 100%
    );
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #e8f8ee !important;
}


/* Header */

.brand-wrapper {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    padding: 18px 0 10px 0;
}

.brand-left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-icon {
    width: 54px;
    height: 54px;
    border-radius: 16px;
    background: linear-gradient(
        135deg,
        #16a34a,
        #059669
    );
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 29px;
    box-shadow: 0 10px 30px rgba(16,185,129,0.20);
}

.brand-name {
    font-size: 23px;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
}

.brand-subtitle {
    color: #9bc7ae;
    font-size: 13px;
    margin-top: 3px;
}


/* Logos */

.logo-box {
    display: flex;
    align-items: center;
    gap: 10px;
}

.logo-box img {
    height: 48px;
    width: auto;
    object-fit: contain;
    border-radius: 8px;
}


/* Hero */

.hero-wrapper {
    margin-top: 25px;
    padding: 50px 35px;
    border-radius: 28px;
    background:
        linear-gradient(
            135deg,
            rgba(22,163,74,0.20),
            rgba(5,150,105,0.10)
        ),
        rgba(8, 29, 22, 0.92);
    border: 1px solid rgba(74,222,128,0.16);
    box-shadow:
        0 25px 70px rgba(0,0,0,0.25),
        inset 0 1px 0 rgba(255,255,255,0.04);
    text-align: center;
}

.hero-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 999px;
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(74,222,128,0.25);
    color: #86efac;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 18px;
}

.hero-title {
    font-size: clamp(38px, 6vw, 70px);
    font-weight: 900;
    line-height: 1.05;
    margin: 0;
    background: linear-gradient(
        90deg,
        #ffffff,
        #86efac,
        #34d399
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    max-width: 760px;
    margin: 20px auto 0 auto;
    color: #b9d8c5;
    font-size: 17px;
    line-height: 1.8;
}

.hero-tagline {
    margin-top: 22px;
    color: #86efac;
    font-size: 14px;
    font-weight: 700;
}


/* Stats */

.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-top: 25px;
}

.stat-card {
    padding: 22px;
    border-radius: 18px;
    background: rgba(13, 37, 28, 0.82);
    border: 1px solid rgba(255,255,255,0.07);
    text-align: center;
    transition: 0.25s ease;
}

.stat-card:hover {
    transform: translateY(-3px);
    border-color: rgba(74,222,128,0.25);
}

.stat-icon {
    font-size: 27px;
    margin-bottom: 8px;
}

.stat-number {
    font-size: 25px;
    font-weight: 800;
    color: #86efac;
}

.stat-label {
    color: #91b6a1;
    font-size: 12px;
    margin-top: 5px;
}


/* Section */

.section-heading {
    margin-top: 45px;
    margin-bottom: 6px;
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
}

.section-description {
    color: #91b6a1;
    margin-bottom: 22px;
    line-height: 1.7;
}


/* Tabs */

button[data-baseweb="tab"] {
    color: #9fc3ae !important;
    font-weight: 700 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #86efac !important;
}

div[data-baseweb="tab-highlight"] {
    background-color: #22c55e !important;
}


/* File uploader */

section[data-testid="stFileUploaderDropzone"] {
    background: rgba(13, 37, 28, 0.65);
    border: 1px dashed rgba(74,222,128,0.30);
    border-radius: 18px;
}

section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #4ade80;
}


/* Camera */

div[data-testid="stCameraInput"] {
    border-radius: 18px;
    overflow: hidden;
}


/* Buttons */

.stButton > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 14px 22px;
    background: linear-gradient(
        135deg,
        #16a34a,
        #059669
    );
    color: white;
    font-size: 16px;
    font-weight: 800;
    box-shadow: 0 10px 25px rgba(22,163,74,0.18);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 30px rgba(22,163,74,0.28);
}


/* Preview */

.preview-card {
    padding: 22px;
    border-radius: 20px;
    background: rgba(13, 37, 28, 0.72);
    border: 1px solid rgba(255,255,255,0.07);
    margin-top: 18px;
}

.preview-title {
    font-size: 19px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 8px;
}

.preview-text {
    color: #91b6a1;
    font-size: 13px;
}


/* Result */

.result-box {
    margin-top: 25px;
    padding: 30px;
    border-radius: 24px;
    background:
        linear-gradient(
            135deg,
            rgba(22,163,74,0.13),
            rgba(5,150,105,0.06)
        );
    border: 1px solid rgba(74,222,128,0.20);
}

.result-icon {
    font-size: 58px;
    margin-bottom: 8px;
}

.result-title {
    font-size: 32px;
    font-weight: 900;
    color: #ffffff;
}

.confidence {
    display: inline-block;
    margin-top: 10px;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(34,197,94,0.13);
    color: #86efac;
    font-weight: 800;
    font-size: 13px;
}

.result-category {
    margin-top: 16px;
    color: #86efac;
    font-size: 16px;
    font-weight: 800;
}

.result-message {
    margin-top: 10px;
    color: #b9d8c5;
    line-height: 1.7;
}


/* Tips */

.tip-card {
    margin-top: 18px;
    padding: 22px;
    border-radius: 18px;
    background: rgba(8, 27, 20, 0.85);
    border: 1px solid rgba(255,255,255,0.06);
}

.tip-title {
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 12px;
}

.tip-item {
    padding: 8px 0;
    color: #b9d8c5;
}


/* Information Cards */

.info-card {
    padding: 25px;
    border-radius: 20px;
    background: rgba(13, 37, 28, 0.75);
    border: 1px solid rgba(255,255,255,0.07);
    height: 100%;
}

.info-icon {
    font-size: 35px;
    margin-bottom: 10px;
}

.info-title {
    font-size: 20px;
    font-weight: 800;
    color: #ffffff;
}

.info-text {
    color: #94b9a3;
    line-height: 1.7;
    margin-top: 8px;
}


/* Steps */

.step-card {
    padding: 25px;
    border-radius: 20px;
    background: rgba(13, 37, 28, 0.70);
    border: 1px solid rgba(255,255,255,0.07);
    height: 100%;
}

.step-number {
    display: inline-flex;
    width: 38px;
    height: 38px;
    border-radius: 50%;
    align-items: center;
    justify-content: center;
    background: #16a34a;
    color: white;
    font-weight: 900;
    margin-bottom: 14px;
}

.step-icon {
    font-size: 28px;
    margin-bottom: 8px;
}

.step-title {
    font-size: 18px;
    font-weight: 800;
}

.step-text {
    margin-top: 8px;
    color: #91b6a1;
    line-height: 1.65;
}


/* Disclaimer */

.disclaimer {
    margin-top: 35px;
    padding: 18px 20px;
    border-radius: 16px;
    background: rgba(120,53,15,0.16);
    border: 1px solid rgba(251,191,36,0.18);
    color: #d9c58c;
    font-size: 13px;
    line-height: 1.7;
}


/* Footer */

.footer-box {
    margin-top: 55px;
    padding: 35px 20px 15px 20px;
    border-top: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}

.footer-title {
    font-size: 22px;
    font-weight: 800;
    color: #ffffff;
}

.footer-subtitle {
    color: #91b6a1;
    margin-top: 7px;
}

.footer-partners {
    margin-top: 18px;
    color: #86efac;
    font-weight: 700;
}

.footer-small {
    margin-top: 16px;
    color: #617f6e;
    font-size: 12px;
}


/* Alerts */

div[data-testid="stAlert"] {
    border-radius: 14px;
}


/* Progress bar */

div[data-testid="stProgressBar"] > div {
    border-radius: 20px;
}

div[data-testid="stProgressBar"] > div > div {
    border-radius: 20px;
    background: linear-gradient(
        90deg,
        #16a34a,
        #34d399
    );
}


/* Images */

img {
    border-radius: 16px;
}


/* Responsive */

@media (max-width: 900px) {

    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .hero-wrapper {
        padding: 35px 20px;
    }

    .hero-title {
        font-size: 42px;
    }
}

@media (max-width: 600px) {

    .stats-grid {
        grid-template-columns: 1fr;
    }

    .hero-title {
        font-size: 36px;
    }

    .hero-subtitle {
        font-size: 15px;
    }

    .brand-wrapper {
        flex-direction: column;
        align-items: flex-start;
    }
}

</style>
"""


# ============================================================
# APPLY CSS
# IMPORTANT:
# CSS IS RENDERED, NOT DISPLAYED AS CODE
# ============================================================

st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def image_to_base64(image_path):
    """
    Convert local image to Base64 for displaying
    inside HTML.
    """

    if not image_path.exists():
        return None

    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        return encoded

    except Exception:
        return None


@st.cache_resource
def load_model():
    """
    Load trained TensorFlow model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


@st.cache_data
def load_class_names():
    """
    Load class names from JSON.
    """

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def predict_waste(image, model, class_names):
    """
    Run prediction on an image.
    """

    image = image.convert("RGB")

    resized_image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.array(
        resized_image
    ).astype(np.float32)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    if predicted_index >= len(class_names):
        raise ValueError(
            "Model output does not match class_names.json."
        )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        predictions
    )


# ============================================================
# WASTE INFORMATION
# ============================================================

WASTE_INFO = {

    "Hazardous": {

        "icon": "☣️",

        "category":
            "Handle Carefully / Special Disposal",

        "message":
            "This item has been classified as potentially "
            "hazardous. Avoid direct contact and keep it "
            "separate from ordinary recyclable waste.",

        "tips": [
            "Keep hazardous materials separate.",
            "Avoid direct contact with unknown substances.",
            "Do not burn hazardous waste.",
            "Follow local hazardous-waste disposal guidance."
        ]
    },

    "Non-Recyclable": {

        "icon": "🚫",

        "category":
            "Non-Recyclable",

        "message":
            "This item has been classified as non-recyclable "
            "by the AI model. Disposal depends on local "
            "waste-management rules.",

        "tips": [
            "Keep it separate from recyclable materials.",
            "Check local disposal guidelines.",
            "Avoid contaminating recyclable waste.",
            "Reduce unnecessary single-use materials."
        ]
    },

    "Organic": {

        "icon": "🌱",

        "category":
            "Compostable / Organic",

        "message":
            "This item has been classified as organic waste. "
            "It may be suitable for composting where an "
            "appropriate composting system is available.",

        "tips": [
            "Separate organic waste from recyclables.",
            "Use a suitable composting system where available.",
            "Keep compostable material free from contamination.",
            "Follow local composting guidelines."
        ]
    },

    "Recyclable": {

        "icon": "♻️",

        "category":
            "Recyclable",

        "message":
            "This item has been classified as recyclable. "
            "Actual recyclability depends on your local "
            "recycling system and material requirements.",

        "tips": [
            "Keep recyclable materials clean and dry.",
            "Separate materials according to local rules.",
            "Avoid contaminated recyclable materials.",
            "Check your local recycling guidelines."
        ]
    }
}


# ============================================================
# LOAD LOGOS
# ============================================================

icodeguru_logo = image_to_base64(
    ICODEGURU_LOGO
)

school_logo = image_to_base64(
    SCHOOL_LOGO
)


# ============================================================
# HEADER
# ============================================================

icodeguru_html = ""

if icodeguru_logo:
    icodeguru_html = f"""
        <img
            src="data:image/png;base64,{icodeguru_logo}"
            alt="iCodeGuru"
        >
    """
else:
    icodeguru_html = """
        <span>iCodeGuru</span>
    """


school_html = ""

if school_logo:
    school_html = f"""
        <img
            src="data:image/png;base64,{school_logo}"
            alt="School"
        >
    """
else:
    school_html = """
        <span>School Exhibition</span>
    """


st.markdown(
    f"""
    <div class="brand-wrapper">

        <div class="brand-left">

            <div class="brand-icon">
                ♻️
            </div>

            <div>
                <div class="brand-name">
                    WasteWise AI
                </div>

                <div class="brand-subtitle">
                    AI-Powered Waste Classification
                </div>
            </div>

        </div>

        <div class="logo-box">

            {icodeguru_html}

            {school_html}

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero-wrapper">

        <div class="hero-badge">
            🤖 AI • ♻️ Sustainability • 🌱 Smart Waste
        </div>

        <h1 class="hero-title">
            WasteWise AI
        </h1>

        <div class="hero-subtitle">
            An AI-powered waste classification and recycling
            assistant that helps identify waste categories
            from images and provides practical disposal guidance.
        </div>

        <div class="hero-tagline">
            Turn Waste into Better Decisions.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT STATS
# ============================================================

st.markdown(
    """
    <div class="stats-grid">

        <div class="stat-card">
            <div class="stat-icon">🤖</div>
            <div class="stat-number">AI</div>
            <div class="stat-label">
                Image Classification
            </div>
        </div>

        <div class="stat-card">
            <div class="stat-icon">♻️</div>
            <div class="stat-number">4</div>
            <div class="stat-label">
                Waste Categories
            </div>
        </div>

        <div class="stat-card">
            <div class="stat-icon">📷</div>
            <div class="stat-number">2</div>
            <div class="stat-label">
                Image Input Methods
            </div>
        </div>

        <div class="stat-card">
            <div class="stat-icon">🌱</div>
            <div class="stat-number">AI</div>
            <div class="stat-label">
                Sustainability Assistant
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        ## ♻️ WasteWise AI

        ### About the Project

        WasteWise AI uses a trained deep-learning image
        classification model to identify common waste categories.

        ### Supported Categories

        - ☣️ Hazardous
        - 🚫 Non-Recyclable
        - 🌱 Organic
        - ♻️ Recyclable

        ### Technology

        - Python
        - TensorFlow
        - EfficientNetB0
        - Streamlit
        - NumPy
        - Pillow

        ### Project Workflow

        Dataset → Training → Model → Streamlit → Prediction

        ---
        """,
        unsafe_allow_html=True
    )

    st.info(
        "For demonstration and educational purposes. "
        "Always follow local waste-management rules."
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-heading">🔍 Analyze Your Waste</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-description">
        Upload an image or use your camera to let WasteWise AI
        classify the waste item.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT TABS
# ============================================================

upload_tab, camera_tab = st.tabs(
    [
        "📁 Upload Image",
        "📷 Use Camera"
    ]
)


uploaded_file = None
camera_file = None


# ============================================================
# UPLOAD IMAGE
# ============================================================

with upload_tab:

    uploaded_file = st.file_uploader(
        "Upload Waste Image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        help="Upload a clear image of a waste item.",
        key="waste_upload"
    )


# ============================================================
# CAMERA
# ============================================================

with camera_tab:

    camera_file = st.camera_input(
        "Take a picture of the waste item",
        key="waste_camera"
    )


# ============================================================
# SELECT IMAGE
# ============================================================

if camera_file is not None:

    selected_file = camera_file
    selected_source = "Camera"

elif uploaded_file is not None:

    selected_file = uploaded_file
    selected_source = "Uploaded Image"

else:

    selected_file = None
    selected_source = None


# ============================================================
# IMAGE PREVIEW
# ============================================================

if selected_file is not None:

    try:

        image = Image.open(
            selected_file
        ).convert("RGB")

        st.markdown(
            """
            <div class="preview-card">

                <div class="preview-title">
                    🖼️ Selected Waste Image
                </div>

                <div class="preview-text">
                    Source: Your selected image
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        analyze_button = st.button(
            "♻️ Analyze Waste",
            type="primary",
            use_container_width=True
        )

        if analyze_button:

            with st.spinner(
                "🤖 AI is analyzing the waste image..."
            ):

                try:

                    # Load model
                    model = load_model()

                    # Load class names
                    class_names = load_class_names()

                    # Predict
                    (
                        predicted_class,
                        confidence,
                        predictions
                    ) = predict_waste(
                        image,
                        model,
                        class_names
                    )

                    # Get information
                    info = WASTE_INFO.get(
                        predicted_class,
                        {
                            "icon": "♻️",
                            "category": predicted_class,
                            "message":
                                "Prediction completed.",
                            "tips": []
                        }
                    )

                    # ====================================================
                    # RESULT
                    # ====================================================

                    st.markdown(
                        f"""
                        <div class="result-box">

                            <div class="result-icon">
                                {info["icon"]}
                            </div>

                            <div class="result-title">
                                {predicted_class}
                            </div>

                            <div class="confidence">
                                Confidence:
                                {confidence * 100:.2f}%
                            </div>

                            <div class="result-category">
                                Category:
                                {info["category"]}
                            </div>

                            <div class="result-message">
                                {info["message"]}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # ====================================================
                    # LOW CONFIDENCE WARNING
                    # ====================================================

                    if confidence < CONFIDENCE_THRESHOLD:

                        st.warning(
                            "⚠️ The model confidence is below "
                            f"{CONFIDENCE_THRESHOLD * 100:.0f}%. "
                            "Please treat this prediction as uncertain "
                            "and verify the item manually."
                        )

                    else:

                        st.success(
                            "✅ The model has sufficient confidence "
                            "for this demonstration."
                        )

                    # ====================================================
                    # WASTE MANAGEMENT TIPS
                    # ====================================================

                    st.markdown(
                        """
                        <div class="tip-card">

                            <div class="tip-title">
                                💡 Waste Management Tips
                            </div>

                        """,
                        unsafe_allow_html=True
                    )

                    for tip in info["tips"]:

                        st.markdown(
                            f"""
                            <div class="tip-item">
                                ✓ {tip}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                    # ====================================================
                    # PREDICTION BREAKDOWN
                    # ====================================================

                    st.markdown(
                        '<div class="section-heading">📊 Prediction Breakdown</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        """
                        <div class="section-description">
                            The chart below shows how strongly the model
                            predicted each waste category.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    top_indices = np.argsort(
                        predictions
                    )[::-1]

                    for index in top_indices:

                        if index >= len(class_names):
                            continue

                        class_name = class_names[index]

                        score = float(
                            predictions[index]
                        )

                        st.write(
                            f"**{class_name}** — "
                            f"{score * 100:.2f}%"
                        )

                        st.progress(
                            min(score, 1.0)
                        )

                except FileNotFoundError as error:

                    st.error(
                        "❌ Model loading error."
                    )

                    st.code(
                        str(error)
                    )

                    st.info(
                        """
                        Please check your project structure:

                        WasteWise-AI/
                        ├── app.py
                        ├── requirements.txt
                        ├── assets/
                        │   ├── icodeguru_logo.png
                        │   └── school_logo.png
                        └── model/
                            ├── wastewise_model.keras
                            └── class_names.json
                        """
                    )

                except Exception as error:

                    st.error(
                        "❌ An unexpected error occurred "
                        "while analyzing the image."
                    )

                    st.code(
                        str(error)
                    )

    except Exception as error:

        st.error(
            f"❌ Could not open the selected image: {error}"
        )


else:

    st.info(
        "📷 Upload an image or use the camera to start."
    )


# ============================================================
# SUPPORTED WASTE CATEGORIES
# ============================================================

st.markdown(
    '<div class="section-heading">♻️ Supported Waste Categories</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-description">
        WasteWise AI currently recognizes four major categories
        from the trained dataset.
    </div>
    """,
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


with col1:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                ☣️
            </div>

            <div class="info-title">
                Hazardous
            </div>

            <div class="info-text">
                Materials that may require special handling
                or disposal procedures.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")


    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                🌱
            </div>

            <div class="info-title">
                Organic
            </div>

            <div class="info-text">
                Biodegradable material such as food and
                other organic waste.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                🚫
            </div>

            <div class="info-title">
                Non-Recyclable
            </div>

            <div class="info-text">
                Waste that may not be accepted by common
                recycling systems.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")


    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                ♻️
            </div>

            <div class="info-title">
                Recyclable
            </div>

            <div class="info-text">
                Materials that may be recyclable depending
                on local recycling facilities and rules.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    '<div class="section-heading">⚙️ How WasteWise AI Works</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-description">
        The complete workflow from image input to AI-powered
        waste classification.
    </div>
    """,
    unsafe_allow_html=True
)


step1, step2, step3, step4 = st.columns(4)


with step1:

    st.markdown(
        """
        <div class="step-card">

            <div class="step-number">
                1
            </div>

            <div class="step-icon">
                📷
            </div>

            <div class="step-title">
                Capture
            </div>

            <div class="step-text">
                Upload a waste image or capture one using
                your device camera.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with step2:

    st.markdown(
        """
        <div class="step-card">

            <div class="step-number">
                2
            </div>

            <div class="step-icon">
                🧠
            </div>

            <div class="step-title">
                AI Processing
            </div>

            <div class="step-text">
                The trained EfficientNetB0 deep-learning
                model processes the image.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with step3:

    st.markdown(
        """
        <div class="step-card">

            <div class="step-number">
                3
            </div>

            <div class="step-icon">
                🔍
            </div>

            <div class="step-title">
                Classification
            </div>

            <div class="step-text">
                The model predicts one of four waste
                categories with a confidence score.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with step4:

    st.markdown(
        """
        <div class="step-card">

            <div class="step-number">
                4
            </div>

            <div class="step-icon">
                🌱
            </div>

            <div class="step-title">
                Guidance
            </div>

            <div class="step-text">
                WasteWise AI provides practical waste
                management tips based on the prediction.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

        ⚠️ <strong>Important:</strong>
        WasteWise AI is an educational AI demonstration.
        Its prediction should not be treated as a definitive
        determination of material safety, recyclability,
        contamination, or hazardousness.

        Always follow official local waste-management,
        recycling, and hazardous-material disposal guidelines.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer-box">

        <div class="footer-title">
            ♻️ WasteWise AI
        </div>

        <div class="footer-subtitle">
            AI-Powered Waste Classification & Recycling Assistant
        </div>

        <div class="footer-partners">
            Built with Python • TensorFlow • Streamlit • EfficientNetB0
        </div>

        <div class="footer-small">
            School Science Exhibition Project
            <br>
            In collaboration with iCodeGuru
            <br><br>
            © 2026 WasteWise AI
        </div>

    </div>
    """,
    unsafe_allow_html=True
)
