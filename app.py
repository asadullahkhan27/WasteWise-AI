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

ICODEGURU_LOGO = BASE_DIR / "assets" / "icodeguru_logo.png"
SCHOOL_LOGO = BASE_DIR / "assets" / "school_logo.png"

IMAGE_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# CUSTOM CSS
# ALL CSS IS INSIDE app.py
# ============================================================

CUSTOM_CSS = """
<style>

/* ========================================================
   GLOBAL
   ======================================================== */

html,
body,
[data-testid="stAppViewContainer"] {

    background:
        radial-gradient(
            circle at top left,
            rgba(34, 197, 94, 0.12),
            transparent 35%
        ),
        radial-gradient(
            circle at top right,
            rgba(16, 185, 129, 0.10),
            transparent 30%
        ),
        #06130e;

}

[data-testid="stAppViewContainer"] {

    color: #ecfdf5;

}

.main {

    background: transparent;

}


/* ========================================================
   REMOVE DEFAULT TOP SPACE
   ======================================================== */

.block-container {

    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;

}


/* ========================================================
   SIDEBAR
   ======================================================== */

[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #082219 0%,
            #061811 100%
        );

    border-right:
        1px solid
        rgba(74, 222, 128, 0.15);

}

[data-testid="stSidebar"] * {

    color: #ecfdf5;

}


/* ========================================================
   BRAND HEADER
   ======================================================== */

.brand-left {

    display: flex;
    align-items: center;
    gap: 15px;

    padding: 8px 0;

}

.brand-icon {

    width: 58px;
    height: 58px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 18px;

    background:
        linear-gradient(
            135deg,
            #16a34a,
            #22c55e
        );

    box-shadow:
        0 10px 35px
        rgba(34, 197, 94, 0.25);

    font-size: 30px;

}

.brand-name {

    font-size: 27px;
    font-weight: 800;

    color: #f0fdf4;

    letter-spacing: -0.5px;

}

.brand-subtitle {

    margin-top: 3px;

    font-size: 13px;

    color: #86efac;

}


/* ========================================================
   LOGOS
   ======================================================== */

.logo-box {

    min-height: 72px;

    display: flex;

    align-items: center;

    justify-content: center;

    padding: 8px;

    border-radius: 16px;

    background:
        rgba(255, 255, 255, 0.045);

    border:
        1px solid
        rgba(134, 239, 172, 0.12);

    backdrop-filter: blur(10px);

}


/* ========================================================
   HERO
   ======================================================== */

.hero-wrapper {

    margin-top: 28px;
    margin-bottom: 28px;

    padding: 55px 30px;

    border-radius: 30px;

    text-align: center;

    background:
        linear-gradient(
            135deg,
            rgba(20, 83, 45, 0.65),
            rgba(6, 78, 59, 0.45)
        );

    border:
        1px solid
        rgba(74, 222, 128, 0.18);

    box-shadow:
        0 25px 80px
        rgba(0, 0, 0, 0.25);

}

.hero-content {

    max-width: 900px;

    margin: auto;

}

.hero-badge {

    display: inline-block;

    padding: 9px 18px;

    border-radius: 999px;

    background:
        rgba(34, 197, 94, 0.12);

    border:
        1px solid
        rgba(74, 222, 128, 0.25);

    color: #bbf7d0;

    font-size: 14px;
    font-weight: 700;

    margin-bottom: 18px;

}

.hero-title {

    margin: 0;

    font-size: clamp(42px, 7vw, 72px);

    font-weight: 900;

    letter-spacing: -2px;

    color: #f0fdf4;

}

.hero-title span {

    color: #22c55e;

}

.hero-subtitle {

    max-width: 780px;

    margin: 20px auto 0;

    color: #bbf7d0;

    font-size: 17px;

    line-height: 1.7;

}

.hero-tagline {

    margin-top: 22px;

    color: #4ade80;

    font-size: 18px;

    font-weight: 800;

}


/* ========================================================
   STATISTICS
   ======================================================== */

.stats-grid {

    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 18px;

    margin: 25px 0 45px;

}

.stat-card {

    padding: 25px 18px;

    text-align: center;

    border-radius: 22px;

    background:
        rgba(255, 255, 255, 0.045);

    border:
        1px solid
        rgba(134, 239, 172, 0.12);

    box-shadow:
        0 12px 35px
        rgba(0, 0, 0, 0.18);

    transition:
        transform 0.25s ease,
        border-color 0.25s ease;

}

.stat-card:hover {

    transform: translateY(-5px);

    border-color:
        rgba(74, 222, 128, 0.35);

}

.stat-icon {

    font-size: 30px;

    margin-bottom: 10px;

}

.stat-number {

    font-size: 28px;

    font-weight: 900;

    color: #4ade80;

}

.stat-label {

    margin-top: 5px;

    color: #a7f3d0;

    font-size: 13px;

}


/* ========================================================
   SECTION HEADINGS
   ======================================================== */

.section-heading {

    margin-top: 45px;

    font-size: 30px;

    font-weight: 850;

    color: #f0fdf4;

}

.section-description {

    margin-top: 8px;
    margin-bottom: 22px;

    color: #86efac;

    font-size: 15px;

    line-height: 1.6;

}


/* ========================================================
   FILE UPLOADER
   ======================================================== */

[data-testid="stFileUploader"] {

    background:
        rgba(255, 255, 255, 0.035);

    border:
        1px solid
        rgba(74, 222, 128, 0.15);

    border-radius: 20px;

    padding: 12px;

}

[data-testid="stFileUploaderDropzone"] {

    background:
        rgba(6, 78, 59, 0.20);

    border:
        1px dashed
        rgba(74, 222, 128, 0.35);

    border-radius: 16px;

}

[data-testid="stFileUploaderDropzoneInstructions"] {

    color: #bbf7d0;

}


/* ========================================================
   BUTTON
   ======================================================== */

.stButton > button {

    width: 100%;

    min-height: 52px;

    border: none;

    border-radius: 15px;

    background:
        linear-gradient(
            135deg,
            #16a34a,
            #22c55e
        );

    color: white;

    font-size: 16px;

    font-weight: 800;

    box-shadow:
        0 12px 30px
        rgba(34, 197, 94, 0.20);

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;

}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 16px 35px
        rgba(34, 197, 94, 0.30);

}


/* ========================================================
   TABS
   ======================================================== */

.stTabs [data-baseweb="tab-list"] {

    gap: 10px;

    background:
        rgba(255, 255, 255, 0.025);

    padding: 8px;

    border-radius: 16px;

}

.stTabs [data-baseweb="tab"] {

    height: 48px;

    padding: 0 20px;

    border-radius: 12px;

    color: #a7f3d0;

    font-weight: 700;

}

.stTabs [aria-selected="true"] {

    background:
        rgba(34, 197, 94, 0.15);

    color: #4ade80;

}


/* ========================================================
   CAMERA
   ======================================================== */

[data-testid="stCameraInput"] {

    border-radius: 20px;

    overflow: hidden;

}


/* ========================================================
   PREVIEW CARD
   ======================================================== */

.preview-card {

    padding: 25px;

    border-radius: 20px;

    background:
        rgba(255, 255, 255, 0.045);

    border:
        1px solid
        rgba(134, 239, 172, 0.12);

    min-height: 150px;

}

.preview-title {

    font-size: 21px;

    font-weight: 800;

    color: #4ade80;

    margin-bottom: 12px;

}

.preview-text {

    color: #bbf7d0;

    line-height: 1.7;

}


/* ========================================================
   RESULT BOX
   ======================================================== */

.result-box {

    margin-top: 30px;

    padding: 35px;

    text-align: center;

    border-radius: 26px;

    background:
        linear-gradient(
            135deg,
            rgba(22, 101, 52, 0.50),
            rgba(6, 78, 59, 0.45)
        );

    border:
        1px solid
        rgba(74, 222, 128, 0.30);

    box-shadow:
        0 20px 60px
        rgba(0, 0, 0, 0.25);

}

.result-icon {

    font-size: 60px;

}

.result-title {

    margin-top: 8px;

    font-size: 38px;

    font-weight: 900;

    color: #f0fdf4;

}

.confidence {

    margin-top: 8px;

    color: #4ade80;

    font-size: 18px;

    font-weight: 800;

}

.result-category {

    margin-top: 10px;

    color: #bbf7d0;

    font-size: 16px;

    font-weight: 700;

}

.result-message {

    max-width: 750px;

    margin: 18px auto 0;

    color: #d1fae5;

    line-height: 1.7;

}


/* ========================================================
   TIP CARD
   ======================================================== */

.tip-card {

    margin: 8px 0;

    padding: 14px 18px;

    border-radius: 13px;

    background:
        rgba(34, 197, 94, 0.07);

    border-left:
        4px solid
        #22c55e;

    color: #d1fae5;

}


/* ========================================================
   CATEGORY CARDS
   ======================================================== */

.info-card {

    height: 100%;

    padding: 25px 18px;

    text-align: center;

    border-radius: 20px;

    background:
        rgba(255, 255, 255, 0.045);

    border:
        1px solid
        rgba(134, 239, 172, 0.12);

    transition:
        transform 0.25s ease;

}

.info-card:hover {

    transform: translateY(-5px);

}

.info-icon {

    font-size: 40px;

    margin-bottom: 12px;

}

.info-title {

    font-size: 19px;

    font-weight: 800;

    color: #f0fdf4;

}

.info-text {

    margin-top: 7px;

    color: #86efac;

    font-size: 13px;

}


/* ========================================================
   HOW IT WORKS
   ======================================================== */

.step-card {

    height: 100%;

    padding: 25px 18px;

    text-align: center;

    border-radius: 20px;

    background:
        rgba(255, 255, 255, 0.04);

    border:
        1px solid
        rgba(134, 239, 172, 0.11);

}

.step-number {

    display: inline-flex;

    width: 38px;
    height: 38px;

    align-items: center;
    justify-content: center;

    border-radius: 50%;

    background:
        rgba(34, 197, 94, 0.14);

    color: #4ade80;

    font-size: 13px;

    font-weight: 900;

}

.step-icon {

    margin-top: 12px;

    font-size: 35px;

}

.step-title {

    margin-top: 10px;

    font-size: 18px;

    font-weight: 800;

    color: #f0fdf4;

}

.step-text {

    margin-top: 8px;

    color: #86efac;

    font-size: 13px;

    line-height: 1.6;

}


/* ========================================================
   DISCLAIMER
   ======================================================== */

.disclaimer {

    margin-top: 45px;

    padding: 20px 22px;

    border-radius: 18px;

    background:
        rgba(234, 179, 8, 0.07);

    border:
        1px solid
        rgba(250, 204, 21, 0.20);

    color: #fef3c7;

    line-height: 1.7;

}


/* ========================================================
   FOOTER
   ======================================================== */

.footer-box {

    margin-top: 45px;

    padding: 35px 20px;

    text-align: center;

    border-top:
        1px solid
        rgba(134, 239, 172, 0.12);

}

.footer-title {

    font-size: 25px;

    font-weight: 900;

    color: #4ade80;

}

.footer-subtitle {

    margin-top: 8px;

    color: #a7f3d0;

}

.footer-partners {

    display: flex;

    justify-content: center;

    align-items: center;

    gap: 10px;

    margin-top: 18px;

    color: #d1fae5;

    font-weight: 700;

}

.footer-small {

    margin-top: 12px;

    color: #6ee7b7;

    font-size: 12px;

}


/* ========================================================
   ALERTS
   ======================================================== */

[data-testid="stAlert"] {

    border-radius: 15px;

}


/* ========================================================
   PROGRESS BAR
   ======================================================== */

[data-testid="stProgress"] > div > div {

    background:
        linear-gradient(
            90deg,
            #16a34a,
            #4ade80
        );

}


/* ========================================================
   IMAGE
   ======================================================== */

[data-testid="stImage"] {

    border-radius: 18px;

    overflow: hidden;

}


/* ========================================================
   RESPONSIVE DESIGN
   ======================================================== */

@media (max-width: 900px) {

    .stats-grid {

        grid-template-columns:
            repeat(2, 1fr);

    }

    .hero-title {

        font-size: 48px;

    }

}


@media (max-width: 600px) {

    .block-container {

        padding-left: 1rem;
        padding-right: 1rem;

    }

    .stats-grid {

        grid-template-columns:
            1fr;

    }

    .hero-wrapper {

        padding: 35px 18px;

        border-radius: 22px;

    }

    .hero-title {

        font-size: 40px;

    }

    .hero-subtitle {

        font-size: 14px;

    }

    .brand-name {

        font-size: 22px;

    }

    .brand-icon {

        width: 48px;
        height: 48px;

        font-size: 25px;

    }

}

</style>
"""


# ============================================================
# APPLY CUSTOM CSS
# ============================================================

st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True
)


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

        if not isinstance(
            classes,
            list
        ):

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


# ============================================================
# LEFT BRAND
# ============================================================

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


# ============================================================
# RIGHT LOGOS
# ============================================================

with header_right:

    logo_left, logo_right = st.columns(
        2,
        vertical_alignment="center"
    )


    # iCodeGuru
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


    # School
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
# MODEL ERROR
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
# IMAGE SOURCE
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
camera_file = None


# ============================================================
# UPLOAD TAB
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
# CAMERA TAB
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
# SELECT SOURCE
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

    except Exception:

        st.error(
            "Unable to open this image. "
            "Please upload or capture a valid image."
        )

        st.stop()


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
            caption=selected_source,
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

                    Source:
                    <strong>{selected_source}</strong>

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


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze_button = st.button(
        "♻️ Analyze Waste",
        use_container_width=True,
        key="analyze_waste"
    )


    # ========================================================
    # AI ANALYSIS
    # ========================================================

    if analyze_button:

        with st.spinner(
            "🤖 AI is analyzing the image..."
        ):

            try:

                # ------------------------------------------------
                # RESIZE
                # ------------------------------------------------

                resized_image = image.resize(
                    IMAGE_SIZE
                )


                # ------------------------------------------------
                # NUMPY
                # ------------------------------------------------

                image_array = np.array(
                    resized_image
                ).astype(
                    np.float32
                )


                # ------------------------------------------------
                # BATCH DIMENSION
                # ------------------------------------------------

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )


                # ------------------------------------------------
                # PREDICTION
                # ------------------------------------------------

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]


                # ------------------------------------------------
                # CLASS
                # ------------------------------------------------

                predicted_index = int(
                    np.argmax(predictions)
                )


                if predicted_index >= len(
                    CLASS_NAMES
                ):

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
                # INFORMATION
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
                # CONFIDENCE
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
                # TIPS
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
                # PREDICTION BREAKDOWN
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

            "category":
                class_name,

            "message":
                ""
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
