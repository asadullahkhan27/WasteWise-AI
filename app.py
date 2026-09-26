import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="WasteWise AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "wastewise_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"
CSS_PATH = BASE_DIR / "style.css"

IMAGE_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 0.60


# =========================================================
# LOAD CSS
# =========================================================

def load_css():

    if not CSS_PATH.exists():
        st.warning(
            "style.css was not found. "
            "Please make sure style.css is in the project root."
        )
        return

    try:
        with open(CSS_PATH, "r", encoding="utf-8") as file:
            css = file.read()

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True
        )

    except Exception as error:
        st.error(f"CSS loading error: {error}")


load_css()


# =========================================================
# WASTE INFORMATION
# =========================================================

WASTE_INFO = {

    "Hazardous": {
        "icon": "☣️",
        "category": "Special / Hazardous Disposal",
        "description": (
            "This category may contain materials that require "
            "special handling or disposal."
        ),
        "tips": [
            "Keep hazardous materials separate.",
            "Do not mix with normal household waste.",
            "Follow your local disposal guidelines."
        ]
    },

    "Non-Recyclable": {
        "icon": "🚫",
        "category": "General / Non-Recyclable Waste",
        "description": (
            "This material is classified by the model as "
            "non-recyclable waste."
        ),
        "tips": [
            "Place it in the appropriate general waste stream.",
            "Avoid contaminating recyclable materials.",
            "Check local recycling rules."
        ]
    },

    "Organic": {
        "icon": "🌱",
        "category": "Organic / Compostable",
        "description": (
            "This material is classified as organic waste "
            "and may be suitable for composting."
        ),
        "tips": [
            "Separate organic waste from dry recyclables.",
            "Consider composting where available.",
            "Keep organic waste properly contained."
        ]
    },

    "Recyclable": {
        "icon": "♻️",
        "category": "Recyclable Material",
        "description": (
            "This material is classified as potentially "
            "recyclable by the trained model."
        ),
        "tips": [
            "Keep recyclable materials clean and dry.",
            "Separate recyclables from organic waste.",
            "Follow local recycling guidelines."
        ]
    }
}


# =========================================================
# LOAD CLASS NAMES
# =========================================================

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

        if isinstance(classes, list) and len(classes) > 0:
            return classes

        return default_classes

    except Exception:
        return default_classes


CLASS_NAMES = load_class_names()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        st.error(
            "Model file not found. "
            "Please check model/wastewise_model.keras"
        )

        return None

    try:

        return tf.keras.models.load_model(
            MODEL_PATH
        )

    except Exception as error:

        st.error(
            f"Model loading error: {error}"
        )

        return None


model = load_model()


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    """
    <section class="hero-wrapper">

        <div class="hero-content">

            <div class="hero-badge">
                ♻️ AI-Powered Waste Classification
            </div>

            <h1 class="hero-title">
                WasteWise <span>AI</span>
            </h1>

            <p class="hero-subtitle">
                Upload a waste image and let our trained AI model
                classify it into the most relevant waste category.
            </p>

            <div class="hero-tagline">
                🌱 Smarter Waste. Cleaner Future.
            </div>

        </div>

    </section>
    """,
    unsafe_allow_html=True
)


# =========================================================
# STATS
# =========================================================

st.markdown(
    """
    <div class="stats-grid">

        <div class="stat-card">
            <div class="stat-icon">🧠</div>
            <div class="stat-number">AI</div>
            <div class="stat-label">Computer Vision</div>
        </div>

        <div class="stat-card">
            <div class="stat-icon">♻️</div>
            <div class="stat-number">4</div>
            <div class="stat-label">Waste Categories</div>
        </div>

        <div class="stat-card">
            <div class="stat-icon">📐</div>
            <div class="stat-number">224×224</div>
            <div class="stat-label">Input Image Size</div>
        </div>

        <div class="stat-card">
            <div class="stat-icon">🚀</div>
            <div class="stat-number">Live</div>
            <div class="stat-label">Streamlit App</div>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL STATUS
# =========================================================

if model is not None:

    st.success(
        "✅ WasteWise AI model is loaded and ready."
    )

else:

    st.error(
        "❌ WasteWise AI model could not be loaded."
    )


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        Analyze Your Waste
    </div>

    <div class="section-description">
        Upload a clear image of a waste item for AI-powered
        classification.
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
    label_visibility="collapsed"
)


st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# =========================================================
# NO IMAGE STATE
# =========================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="ready-card">

            <div class="ready-icon">📸</div>

            <div class="ready-title">
                Ready to Analyze
            </div>

            <div class="ready-text">
                Upload a waste image above to start
                the AI classification process.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# IMAGE ANALYSIS
# =========================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.markdown(
            """
            <div class="section-heading">
                Uploaded Image
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
            use_container_width=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # Analyze Button
        # -------------------------------------------------

        analyze_button = st.button(
            "🔍 Analyze Waste with AI"
        )


        if analyze_button:

            if model is None:

                st.error(
                    "Model is not available."
                )

            else:

                with st.spinner(
                    "🤖 AI is analyzing your image..."
                ):

                    # Resize
                    processed_image = image.resize(
                        IMAGE_SIZE
                    )

                    # Convert to numpy
                    image_array = np.array(
                        processed_image
                    ).astype("float32")

                    # Add batch dimension
                    image_array = np.expand_dims(
                        image_array,
                        axis=0
                    )

                    # Prediction
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


                # -------------------------------------------------
                # Result Information
                # -------------------------------------------------

                waste = WASTE_INFO.get(
                    predicted_class,
                    {
                        "icon": "♻️",
                        "category": predicted_class,
                        "description": (
                            "The model classified this "
                            "image into this category."
                        ),
                        "tips": []
                    }
                )


                # -------------------------------------------------
                # Result Card
                # -------------------------------------------------

                st.markdown(
                    f"""
                    <div class="result-card">

                        <div class="result-icon">
                            {waste["icon"]}
                        </div>

                        <div class="result-label">
                            AI Prediction
                        </div>

                        <div class="result-name">
                            {predicted_class}
                        </div>

                        <div class="result-category">
                            {waste["category"]}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # -------------------------------------------------
                # Confidence
                # -------------------------------------------------

                confidence_percentage = (
                    confidence * 100
                )

                st.markdown(
                    f"""
                    <div class="confidence-card">

                        <div class="confidence-number">
                            {confidence_percentage:.1f}%
                        </div>

                        <div class="confidence-label">
                            Model Confidence
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(
                    confidence
                )


                # -------------------------------------------------
                # Confidence Message
                # -------------------------------------------------

                if confidence >= CONFIDENCE_THRESHOLD:

                    st.success(
                        "High-confidence prediction. "
                        "For best results, use a clear image "
                        "with the waste item centered."
                    )

                else:

                    st.warning(
                        "The model is less confident about this "
                        "prediction. Try another clear image "
                        "with better lighting."
                    )


                # -------------------------------------------------
                # Smart Guidance
                # -------------------------------------------------

                st.markdown(
                    """
                    <div class="section-heading">
                        ♻️ Smart Waste Guidance
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="info-card">

                        <div class="info-icon">
                            {waste["icon"]}
                        </div>

                        <div class="info-title">
                            {waste["category"]}
                        </div>

                        <div class="info-text">
                            {waste["description"]}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # -------------------------------------------------
                # Tips
                # -------------------------------------------------

                if waste["tips"]:

                    st.markdown(
                        """
                        <div class="section-description">
                            Recommended handling:
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    for tip in waste["tips"]:

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


                # -------------------------------------------------
                # Prediction Breakdown
                # -------------------------------------------------

                st.markdown(
                    """
                    <div class="section-heading">
                        📊 Prediction Breakdown
                    </div>

                    <div class="section-description">
                        Probability assigned by the AI model
                        to each supported category.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                for index, class_name in enumerate(
                    CLASS_NAMES
                ):

                    probability = float(
                        predictions[index]
                    )

                    st.markdown(
                        f"""
                        <div class="prediction-card">

                            <div class="prediction-title">
                                {class_name}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(
                        probability
                    )

                    st.caption(
                        f"{probability * 100:.2f}%"
                    )


                # -------------------------------------------------
                # Educational Notice
                # -------------------------------------------------

                st.info(
                    "⚠️ Educational demonstration only. "
                    "The AI prediction is based on visual patterns "
                    "learned from the training dataset. It does not "
                    "guarantee material composition, recyclability, "
                    "toxicity, or microbiological safety. Always "
                    "follow local waste-management guidelines."
                )


    except Exception as error:

        st.error(
            f"Image processing error: {error}"
        )


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        ⚙️ How WasteWise AI Works
    </div>

    <div class="section-description">
        A simple computer-vision pipeline turns an uploaded
        image into an AI classification.
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
                Upload a clear image of the waste item.
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
                The image is resized to 224×224 pixels
                before inference.
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
                AI Prediction
            </div>

            <div class="step-text">
                EfficientNetB0 analyzes visual patterns
                learned during training.
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
                WasteWise AI displays the predicted
                category and confidence.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SUPPORTED CATEGORIES
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        ♻️ Supported Waste Categories
    </div>

    <div class="section-description">
        The current trained model recognizes four categories.
    </div>

    <div class="chips">

        <div class="chip">
            ☣️ Hazardous
        </div>

        <div class="chip">
            🚫 Non-Recyclable
        </div>

        <div class="chip">
            🌱 Organic
        </div>

        <div class="chip">
            ♻️ Recyclable
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        🧠 Model Information
    </div>
    """,
    unsafe_allow_html=True
)

model_col1, model_col2, model_col3 = st.columns(3)


with model_col1:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                🧠
            </div>

            <div class="info-title">
                Architecture
            </div>

            <div class="info-text">
                EfficientNetB0 transfer learning model.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with model_col2:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                📐
            </div>

            <div class="info-title">
                Input Size
            </div>

            <div class="info-text">
                Images are processed at 224 × 224 pixels.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with model_col3:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                📊
            </div>

            <div class="info-title">
                Validation Accuracy
            </div>

            <div class="info-text">
                Approximately 74.14% on the validation set
                used during model development.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

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

        <div class="footer-note">
            Built with Python • TensorFlow • EfficientNetB0 •
            Streamlit
        </div>

    </div>
    """,
    unsafe_allow_html=True
)
