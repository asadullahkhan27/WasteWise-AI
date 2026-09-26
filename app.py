# ============================================================
# WASTEWISE AI
# AI-Powered Waste Classification & Recycling Assistant
# ============================================================

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
    page_title="WasteWise AI | AI Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "wastewise_model.keras"
)

CLASS_NAMES_PATH = (
    BASE_DIR
    / "model"
    / "class_names.json"
)

CSS_PATH = (
    BASE_DIR
    / "style.css"
)

ICODEGURU_LOGO = (
    BASE_DIR
    / "assets"
    / "icodeguru_logo.png"
)

SCHOOL_LOGO = (
    BASE_DIR
    / "assets"
    / "school_logo.png"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# LOAD CSS
# ============================================================

def load_css():

    if not CSS_PATH.exists():

        st.warning(
            "style.css not found. "
            "Please upload style.css to the project root."
        )

        return

    try:

        with open(
            CSS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            css = file.read()

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True
        )

    except Exception as error:

        st.error(
            f"CSS loading error: {error}"
        )


load_css()


# ============================================================
# WASTE INFORMATION
# ============================================================

WASTE_INFO = {

    "Hazardous": {

        "icon": "☣️",

        "category":
            "Handle Carefully / Special Disposal",

        "message":
            "Hazardous waste may require special handling. "
            "Do not mix potentially hazardous materials with "
            "ordinary household recycling.",

        "tips": [
            "Keep hazardous items separate.",
            "Avoid direct contact with unknown substances.",
            "Follow local hazardous-waste disposal guidance.",
            "Do not burn or improperly dump hazardous materials."
        ]
    },

    "Non-Recyclable": {

        "icon": "🚫",

        "category":
            "Non-Recyclable",

        "message":
            "This item was classified as non-recyclable "
            "by the AI model. Disposal options depend on "
            "local waste-management rules.",

        "tips": [
            "Keep it separate from recyclable materials.",
            "Check local waste-disposal guidelines.",
            "Avoid contaminating recyclable waste.",
            "Reduce single-use materials where possible."
        ]
    },

    "Organic": {

        "icon": "🌱",

        "category":
            "Compostable / Organic",

        "message":
            "Organic waste can potentially be composted "
            "when suitable facilities or composting systems "
            "are available.",

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
            "This item was classified as recyclable. "
            "Actual recyclability depends on your local "
            "recycling system and material requirements.",

        "tips": [
            "Keep recyclable materials clean and dry.",
            "Separate materials according to local rules.",
            "Avoid mixing contaminated waste with recyclables.",
            "Check your local recycling guidelines."
        ]
    }
}


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    default_classes = [
        "Hazardous",
        "Non-Recyclable",
        "Organic",
        "Recyclable"
    ]

    if not CLASS_NAMES_PATH.exists():

        return default_classes

    try:

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            classes = json.load(file)

        if (
            isinstance(classes, list)
            and len(classes) > 0
        ):

            return classes

        return default_classes

    except Exception:

        return default_classes


CLASS_NAMES = load_class_names()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        return None

    try:

        return tf.keras.models.load_model(
            MODEL_PATH
        )

    except Exception as error:

        st.error(
            f"Unable to load model: {error}"
        )

        return None


model = load_model()


# ============================================================
# BRAND HEADER
# ============================================================

icodeguru_exists = ICODEGURU_LOGO.exists()
school_exists = SCHOOL_LOGO.exists()


logo_left = ""

if icodeguru_exists:

    logo_left = """
        <img
            src="data:image/png;base64,LOGO_PLACEHOLDER"
            class="brand-logo"
        >
    """


# ============================================================
# HEADER BRANDING
# ============================================================

st.markdown(
    """
    <div class="brand-bar">

        <div class="brand-left">

            <div>

                <div class="brand-name">
                    ♻️ WasteWise AI
                </div>

                <div class="brand-subtitle">
                    AI-Powered Waste Classification
                </div>

            </div>

        </div>


        <div class="partner-logos">

            <div class="brand-subtitle">
                A Project by
            </div>

            <div class="brand-name">
                iCodeGuru
            </div>

            <div class="brand-name">
                🏫 School Science Exhibition
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-wrapper">

        <div class="hero-content">

            <div class="hero-badge">
                ♻️ AI • Computer Vision • Sustainability
            </div>

            <h1 class="hero-title">
                WasteWise <span>AI</span>
            </h1>

            <div class="hero-subtitle">
                An AI-powered waste classification and
                recycling awareness assistant designed
                to demonstrate how computer vision can
                support smarter waste management.
            </div>

            <div class="hero-tagline">
                🌱 Smarter Waste. Cleaner Future.
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# STATS
# ============================================================

st.markdown(
    """
    <div class="stats-grid">

        <div class="stat-card">

            <div class="stat-icon">
                🧠
            </div>

            <div class="stat-number">
                AI
            </div>

            <div class="stat-label">
                Computer Vision
            </div>

        </div>


        <div class="stat-card">

            <div class="stat-icon">
                ♻️
            </div>

            <div class="stat-number">
                4
            </div>

            <div class="stat-label">
                Waste Categories
            </div>

        </div>


        <div class="stat-card">

            <div class="stat-icon">
                📐
            </div>

            <div class="stat-number">
                224×224
            </div>

            <div class="stat-label">
                Input Resolution
            </div>

        </div>


        <div class="stat-card">

            <div class="stat-icon">
                🎯
            </div>

            <div class="stat-number">
                74.14%
            </div>

            <div class="stat-label">
                Validation Accuracy
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

    st.header(
        "♻️ WasteWise AI"
    )

    st.markdown(
        "### 🌱 About the Project"
    )

    st.write(
        """
        WasteWise AI is an educational computer-vision
        project that classifies uploaded waste images
        into four trained categories.
        """
    )

    st.markdown(
        "### 🗂️ Supported Classes"
    )

    for class_name in CLASS_NAMES:

        info = WASTE_INFO.get(
            class_name,
            {}
        )

        icon = info.get(
            "icon",
            "♻️"
        )

        st.write(
            f"{icon} {class_name}"
        )

    st.markdown("---")

    st.caption(
        "🧠 Model: EfficientNetB0"
    )

    st.caption(
        "📐 Input: 224 × 224"
    )

    st.caption(
        "♻️ Classes: 4"
    )

    st.caption(
        "🎓 Educational Exhibition Project"
    )


# ============================================================
# MODEL CHECK
# ============================================================

if model is None:

    st.error(
        "⚠️ Trained model not found."
    )

    st.markdown(
        """
        Please make sure the following files exist
        in your GitHub repository:

        `model/wastewise_model.keras`

        `model/class_names.json`
        """
    )

    st.stop()


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        🔍 Analyze Your Waste
    </div>

    <div class="section-description">
        Upload a clear image of a waste item and let
        WasteWise AI analyze its visual characteristics.
    </div>

    <div class="upload-container">
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(

    "Upload Waste Image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],

    label_visibility="collapsed",

    help="Upload a clear image of the waste item."
)


st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as error:

        st.error(
            f"Could not open image: {error}"
        )

        st.stop()


    st.markdown(
        """
        <div class="section-heading">
            🖼️ Image Preview
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="preview-card">',
        unsafe_allow_html=True
    )

    st.image(
        image,
        caption="Uploaded Waste Image",
        use_container_width=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # ========================================================
    # ANALYZE
    # ========================================================

    analyze_button = st.button(
        "🔍 Analyze Waste with AI",
        type="primary",
        use_container_width=True
    )


    if analyze_button:

        with st.spinner(
            "🤖 WasteWise AI is analyzing your image..."
        ):

            processed_image = image.resize(
                IMAGE_SIZE
            )

            image_array = np.array(
                processed_image
            ).astype("float32")

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

            predicted_class = CLASS_NAMES[
                predicted_index
            ]

            confidence = float(
                predictions[
                    predicted_index
                ]
            )


        # ====================================================
        # RESULT
        # ====================================================

        info = WASTE_INFO.get(
            predicted_class,
            {
                "icon": "♻️",
                "category": predicted_class,
                "message":
                    "No additional information available.",
                "tips": []
            }
        )


        st.markdown(
            """
            <div class="section-heading">
                🤖 AI Prediction
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="result-box">

                <div class="result-title">
                    {info["icon"]} {predicted_class}
                </div>

                <div class="confidence">
                    Confidence: {confidence * 100:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        if confidence >= 0.80:

            st.success(
                "🟢 High-confidence prediction"
            )

        elif confidence >= CONFIDENCE_THRESHOLD:

            st.warning(
                "🟡 Moderate-confidence prediction"
            )

        else:

            st.error(
                "🔴 Low-confidence prediction"
            )

            st.warning(
                "The AI is not sufficiently confident. "
                "Try a clearer image with better lighting."
            )


        # ====================================================
        # GUIDANCE
        # ====================================================

        st.markdown(
            """
            <div class="section-heading">
                💡 Waste Management Guidance
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="info-card">

                <div class="info-icon">
                    {info["icon"]}
                </div>

                <div class="info-title">
                    {info["category"]}
                </div>

                <div class="info-text">
                    {info["message"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # TIPS
        # ====================================================

        st.markdown(
            """
            <div class="section-heading">
                ✅ Recommended Practices
            </div>
            """,
            unsafe_allow_html=True
        )


        for tip in info["tips"]:

            st.markdown(
                f"""
                <div class="info-card">

                    <div class="info-text">
                        ✓ {tip}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # TOP PREDICTIONS
        # ====================================================

        st.markdown(
            """
            <div class="section-heading">
                📊 AI Prediction Breakdown
            </div>

            <div class="section-description">
                Probability distribution across the four
                supported waste categories.
            </div>
            """,
            unsafe_allow_html=True
        )


        sorted_indices = np.argsort(
            predictions
        )[::-1]


        for rank, index in enumerate(
            sorted_indices[:3],
            start=1
        ):

            class_name = CLASS_NAMES[
                index
            ]

            probability = float(
                predictions[index]
            )

            st.markdown(
                f"""
                <div class="info-card">

                    <div class="info-title">
                        #{rank} {class_name}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                min(
                    probability,
                    1.0
                )
            )

            st.caption(
                f"{probability * 100:.2f}%"
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.warning(
            """
            ⚠️ **Educational Demonstration**

            WasteWise AI provides an image-based model
            prediction. It does not guarantee material
            composition, recyclability, hazardousness,
            toxicity, or microbiological safety.

            Always follow local waste-management rules
            and seek appropriate professional guidance
            for hazardous materials.
            """
        )


else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.markdown(
        """
        <div class="section-heading">
            📸 How WasteWise AI Works
        </div>
        """,
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            """
            <div class="step-card">

                <div class="step-number">
                    1
                </div>

                <div class="step-icon">
                    📸
                </div>

                <div class="step-title">
                    Upload
                </div>

                <div class="step-text">
                    Upload a clear image of a waste item.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="step-card">

                <div class="step-number">
                    2
                </div>

                <div class="step-icon">
                    🖼️
                </div>

                <div class="step-title">
                    Preprocess
                </div>

                <div class="step-text">
                    The image is resized to 224 × 224
                    pixels before inference.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="step-card">

                <div class="step-number">
                    3
                </div>

                <div class="step-icon">
                    🧠
                </div>

                <div class="step-title">
                    AI Analysis
                </div>

                <div class="step-text">
                    EfficientNetB0 analyzes visual
                    patterns learned during training.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            """
            <div class="step-card">

                <div class="step-number">
                    4
                </div>

                <div class="step-icon">
                    ♻️
                </div>

                <div class="step-title">
                    Result
                </div>

                <div class="step-text">
                    The app displays the predicted
                    category and confidence.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# SUPPORTED CATEGORIES
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        ♻️ Supported Waste Categories
    </div>

    <div class="section-description">
        WasteWise AI currently supports four trained classes.
    </div>

    <div class="stats-grid">

        <div class="info-card">
            <div class="info-icon">☣️</div>
            <div class="info-title">Hazardous</div>
            <div class="info-text">
                Materials requiring special handling.
            </div>
        </div>

        <div class="info-card">
            <div class="info-icon">🚫</div>
            <div class="info-title">Non-Recyclable</div>
            <div class="info-text">
                Waste not classified as recyclable by the model.
            </div>
        </div>

        <div class="info-card">
            <div class="info-icon">🌱</div>
            <div class="info-title">Organic</div>
            <div class="info-text">
                Organic waste potentially suitable for composting.
            </div>
        </div>

        <div class="info-card">
            <div class="info-icon">♻️</div>
            <div class="info-title">Recyclable</div>
            <div class="info-text">
                Materials classified as potentially recyclable.
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-box">

        <div class="footer-logo">
            ♻️
        </div>

        <div class="footer-title">
            WasteWise AI
        </div>

        <div class="footer-subtitle">
            AI-Powered Waste Classification & Recycling Assistant
        </div>

        <div class="footer-subtitle">
            Science Exhibition Project • iCodeGuru
        </div>

        <div class="footer-partners">

            <div class="footer-partner">
                iCodeGuru
            </div>

            <div class="footer-partner">
                School Science Exhibition
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)
