```python
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
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "wastewise_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"

CSS_PATH = BASE_DIR / "style.css"

ICODEGURU_LOGO = BASE_DIR / "assets" / "icodeguru_logo.png"
SCHOOL_LOGO = BASE_DIR / "assets" / "school_logo.png"

IMAGE_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# LOAD CSS FROM style.css
# ============================================================

def load_css():

    if not CSS_PATH.exists():

        st.warning(
            "style.css not found. "
            "Please make sure style.css is in the project root."
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
            f"Unable to load style.css: {error}"
        )


load_css()


# ============================================================
# IMAGE TO BASE64
# ============================================================

def image_to_base64(image_path):

    if not image_path.exists():
        return None

    try:

        with open(
            image_path,
            "rb"
        ) as image_file:

            encoded = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        return encoded

    except Exception:

        return None


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

        if not isinstance(classes, list):

            return default_classes

        return classes

    except Exception:

        return default_classes


CLASS_NAMES = load_class_names()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        return (
            None,
            f"Model file not found: {MODEL_PATH}"
        )

    try:

        loaded_model = tf.keras.models.load_model(
            MODEL_PATH
        )

        return loaded_model, None

    except Exception as error:

        return (
            None,
            str(error)
        )


model, model_error = load_model()


# ============================================================
# BRAND HEADER
# ============================================================

header_left, header_right = st.columns(
    [2.2, 1.8],
    vertical_alignment="center"
)


with header_left:

    st.markdown(
        """
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
        """,
        unsafe_allow_html=True
    )


with header_right:

    logo_left, logo_right = st.columns(
        2,
        vertical_alignment="center"
    )


    # iCodeGuru Logo
    with logo_left:

        if icodeguru_logo:

            st.markdown(
                f"""
                <div class="logo-box">

                    <img
                        src="data:image/png;base64,{icodeguru_logo}"
                        style="
                            width:100px;
                            max-height:70px;
                            object-fit:contain;
                        "
                    />

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="logo-box">

                    <div class="brand-name">
                        iCodeGuru
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # School Logo
    with logo_right:

        if school_logo:

            st.markdown(
                f"""
                <div class="logo-box">

                    <img
                        src="data:image/png;base64,{school_logo}"
                        style="
                            width:100px;
                            max-height:70px;
                            object-fit:contain;
                        "
                    />

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="logo-box">

                    <div class="brand-name">
                        🏫 School
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


st.divider()


# ============================================================
# HERO SECTION
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
                recycling awareness assistant designed to
                demonstrate how computer vision can support
                smarter waste management.

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
# STATISTICS
# ============================================================

st.markdown(
    """
    <div class="stats-grid">

        <div class="stat-card">

            <div class="stat-icon">
                🤖
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
                📷
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

    st.markdown(
        "## ♻️ WasteWise AI"
    )

    st.markdown(
        """
        ### About

        WasteWise AI uses computer vision to classify
        waste images into four broad categories.

        ### Supported Categories

        ☣️ Hazardous

        🚫 Non-Recyclable

        🌱 Organic

        ♻️ Recyclable

        ---

        **Model:** EfficientNetB0

        **Input:** 224 × 224

        **AI Type:** Image Classification
        """
    )

    st.divider()

    if model is not None:

        st.success(
            "AI Model Loaded"
        )

    else:

        st.error(
            "AI Model Not Loaded"
        )


# ============================================================
# MODEL ERROR / DEBUG
# ============================================================

if model is None:

    st.error(
        "⚠️ WasteWise AI model could not be loaded."
    )

    st.code(
        str(model_error)
    )

    st.markdown(
        "### Expected Project Structure"
    )

    st.code(
        """
WasteWise-AI/
│
├── app.py
├── style.css
├── requirements.txt
│
├── assets/
│   ├── icodeguru_logo.png
│   └── school_logo.png
│
└── model/
    ├── wastewise_model.keras
    └── class_names.json
        """,
        language="text"
    )

    if MODEL_PATH.parent.exists():

        model_files = [
            file.name
            for file in MODEL_PATH.parent.iterdir()
        ]

        st.write(
            "Files currently inside model folder:"
        )

        st.write(
            model_files
        )

    else:

        st.warning(
            "The model folder does not exist."
        )

    st.stop()


# ============================================================
# ANALYZE SECTION
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        🔍 Analyze Your Waste
    </div>

    <div class="section-description">

        Upload a clear image of a waste item or use your
        camera to capture one for AI analysis.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# IMAGE INPUT OPTIONS
# ============================================================

st.markdown(
    "### 📸 Choose Image Source"
)


upload_tab, camera_tab = st.tabs(
    [
        "📁 Upload Image",
        "📷 Use Camera"
    ]
)


uploaded_file = None


# ============================================================
# UPLOAD IMAGE
# ============================================================

with upload_tab:

    st.markdown(
        """
        <div class="preview-card">

            <div class="preview-title">
                📁 Upload Waste Image
            </div>

            <div class="preview-text">

                Select a JPG, JPEG, PNG or WEBP image
                from your device.

            </div>

        </div>
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
        help="Upload a clear image of a waste item.",
        key="waste_upload"
    )


# ============================================================
# CAMERA INPUT
# ============================================================

with camera_tab:

    st.markdown(
        """
        <div class="preview-card">

            <div class="preview-title">
                📷 Capture Waste Image
            </div>

            <div class="preview-text">

                Allow camera access and take a clear
                photo of the waste item.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    camera_file = st.camera_input(
        "Take a picture of the waste item",
        key="waste_camera"
    )


# ============================================================
# SELECT IMAGE SOURCE
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
# IMAGE ANALYSIS
# ============================================================

if selected_file is not None:

    try:

        image = Image.open(
            selected_file
        ).convert("RGB")

    except Exception:

        st.error(
            "Unable to open this image. "
            "Please upload or capture a valid image."
        )

        st.stop()


    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    st.markdown(
        "### 📸 Image Preview"
    )

    preview_left, preview_right = st.columns(
        [1, 1],
        vertical_alignment="center"
    )


    with preview_left:

        st.image(
            image,
            caption=f"{selected_source}",
            use_container_width=True
        )


    with preview_right:

        st.markdown(
            f"""
            <div class="preview-card">

                <div class="preview-title">
                    ✅ Image Ready
                </div>

                <div class="preview-text">

                    Source: <strong>{selected_source}</strong>

                    <br><br>

                    Your image is ready for AI analysis.

                    <br><br>

                    Click the button below to classify
                    the waste category.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    analyze_button = st.button(
        "♻️ Analyze Waste",
        use_container_width=True,
        key="analyze_waste"
    )


    if analyze_button:

        with st.spinner(
            "🤖 AI is analyzing the image..."
        ):

            try:

                # ------------------------------------------------
                # RESIZE IMAGE
                # ------------------------------------------------

                resized_image = image.resize(
                    IMAGE_SIZE
                )


                # ------------------------------------------------
                # CONVERT IMAGE TO NUMPY
                # ------------------------------------------------

                image_array = np.array(
                    resized_image
                ).astype(
                    np.float32
                )


                # ------------------------------------------------
                # ADD BATCH DIMENSION
                # ------------------------------------------------

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )


                # ------------------------------------------------
                # MODEL PREDICTION
                # ------------------------------------------------

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]


                # ------------------------------------------------
                # PREDICTED CLASS
                # ------------------------------------------------

                predicted_index = int(
                    np.argmax(predictions)
                )


                if predicted_index >= len(CLASS_NAMES):

                    raise ValueError(
                        "Model output does not match "
                        "class_names.json."
                    )


                predicted_class = CLASS_NAMES[
                    predicted_index
                ]


                confidence = float(
                    predictions[predicted_index]
                )


                # ------------------------------------------------
                # WASTE INFORMATION
                # ------------------------------------------------

                info = WASTE_INFO.get(
                    predicted_class,

                    {
                        "icon": "♻️",

                        "category":
                            predicted_class,

                        "message":
                            "The AI classified this image.",

                        "tips": []
                    }
                )


                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

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

                            {info["category"]}

                        </div>

                        <div class="result-message">

                            {info["message"]}

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # ------------------------------------------------
                # CONFIDENCE MESSAGE
                # ------------------------------------------------

                if confidence < CONFIDENCE_THRESHOLD:

                    st.warning(
                        "⚠️ The model confidence is relatively "
                        "low. Please treat this result as an "
                        "AI demonstration rather than a definitive "
                        "waste-disposal decision."
                    )

                else:

                    st.success(
                        "✅ AI classification completed successfully."
                    )


                # ------------------------------------------------
                # WASTE MANAGEMENT TIPS
                # ------------------------------------------------

                st.markdown(
                    "### 💡 Waste Management Tips"
                )


                for tip in info["tips"]:

                    st.markdown(
                        f"""
                        <div class="tip-card">
                            ✅ {tip}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # ------------------------------------------------
                # TOP PREDICTIONS
                # ------------------------------------------------

                st.markdown(
                    "### 📊 AI Prediction Breakdown"
                )


                top_indices = np.argsort(
                    predictions
                )[::-1][:len(CLASS_NAMES)]


                for index in top_indices:

                    class_name = CLASS_NAMES[
                        index
                    ]

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


            except Exception as error:

                st.error(
                    "Unable to analyze the image."
                )

                st.exception(
                    error
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

        WasteWise AI currently recognizes four broad
        waste categories from the training dataset.

    </div>
    """,
    unsafe_allow_html=True
)


category_columns = st.columns(
    len(CLASS_NAMES)
)


for column, class_name in zip(
    category_columns,
    CLASS_NAMES
):

    info = WASTE_INFO.get(
        class_name,

        {
            "icon": "♻️",

            "category": class_name,

            "message": ""
        }
    )


    with column:

        st.markdown(
            f"""
            <div class="info-card">

                <div class="info-icon">
                    {info["icon"]}
                </div>

                <div class="info-title">
                    {class_name}
                </div>

                <div class="info-text">
                    {info["category"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# HOW WASTEWISE AI WORKS
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        📸 How WasteWise AI Works
    </div>
    """,
    unsafe_allow_html=True
)


steps = [

    (
        "01",
        "📤",
        "Upload / Capture",
        "Upload an image or capture one using your camera."
    ),

    (
        "02",
        "🖼️",
        "Preprocess",
        "The image is resized to 224 × 224 pixels."
    ),

    (
        "03",
        "🤖",
        "AI Analysis",
        "EfficientNetB0 analyzes visual features."
    ),

    (
        "04",
        "📊",
        "Result",
        "The model predicts the waste category."
    )

]


step_columns = st.columns(
    4
)


for column, step in zip(
    step_columns,
    steps
):

    number, icon, title, description = step


    with column:

        st.markdown(
            f"""
            <div class="step-card">

                <div class="step-number">
                    {number}
                </div>

                <div class="step-icon">
                    {icon}
                </div>

                <div class="step-title">
                    {title}
                </div>

                <div class="step-text">
                    {description}
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

        <strong>⚠️ Important:</strong>

        WasteWise AI is an educational AI demonstration.
        Image classification alone cannot determine the
        exact material composition, contamination level,
        microbiological safety, or official disposal method
        of an item. Always follow local waste-management
        guidance.

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

        <div class="footer-title">
            ♻️ WasteWise AI
        </div>

        <div class="footer-subtitle">

            AI-Powered Waste Classification &
            Recycling Awareness Assistant

        </div>

        <div class="footer-partners">

            <span>iCodeGuru</span>

            <span>•</span>

            <span>
                🏫 School Science Exhibition
            </span>

        </div>

        <div class="footer-small">

            Built with Python • TensorFlow •
            EfficientNetB0 • Streamlit

        </div>

    </div>
    """,
    unsafe_allow_html=True
)
```
