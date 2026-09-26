```python
# ============================================================
# WASTEWISE AI
# AI-Powered Waste Classification & Recycling Assistant
# Premium Science Exhibition Edition
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
    page_title="WasteWise AI | Smart Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "wastewise_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"

IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# WASTE INFORMATION
# ============================================================

WASTE_INFO = {

    "Hazardous": {
        "icon": "☣️",
        "color": "#dc2626",
        "short": "Handle Carefully",
        "category": "Special / Hazardous Disposal",
        "description": (
            "This category may require special handling. "
            "Keep potentially hazardous materials separate "
            "from ordinary household recycling."
        ),
        "tips": [
            "Keep hazardous items separated.",
            "Avoid direct contact with unknown substances.",
            "Never burn hazardous materials.",
            "Follow local hazardous-waste guidance."
        ]
    },

    "Non-Recyclable": {
        "icon": "🚫",
        "color": "#64748b",
        "short": "General Waste",
        "category": "Non-Recyclable / General Waste",
        "description": (
            "The AI classified this image as non-recyclable. "
            "Disposal should follow your local waste-management rules."
        ),
        "tips": [
            "Keep it separate from recyclable materials.",
            "Avoid contaminating recyclable waste.",
            "Use the appropriate general-waste bin.",
            "Reduce unnecessary single-use materials."
        ]
    },

    "Organic": {
        "icon": "🌱",
        "color": "#16a34a",
        "short": "Organic Material",
        "category": "Organic / Compostable",
        "description": (
            "Organic materials may be suitable for composting "
            "when an appropriate composting system is available."
        ),
        "tips": [
            "Separate organic waste from recyclables.",
            "Use a suitable composting system.",
            "Keep compostable material uncontaminated.",
            "Follow local composting guidelines."
        ]
    },

    "Recyclable": {
        "icon": "♻️",
        "color": "#059669",
        "short": "Recyclable Material",
        "category": "Recyclable Material",
        "description": (
            "The AI classified this image as recyclable. "
            "Actual recyclability depends on local recycling facilities "
            "and material-specific rules."
        ),
        "tips": [
            "Keep recyclable materials clean and dry.",
            "Separate materials according to local rules.",
            "Avoid mixing contaminated waste with recyclables.",
            "Check your local recycling requirements."
        ]
    }
}


# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       RESET / GLOBAL
       ===================================================== */

    #MainMenu {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 5% 5%,
                rgba(16, 185, 129, 0.14),
                transparent 25%
            ),
            radial-gradient(
                circle at 95% 10%,
                rgba(59, 130, 246, 0.12),
                transparent 25%
            ),
            radial-gradient(
                circle at 50% 90%,
                rgba(34, 197, 94, 0.08),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #f8fffc 0%,
                #f4f8f7 45%,
                #f8fbff 100%
            );
    }

    .block-container {
        max-width: 1320px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }


    /* =====================================================
       HERO
       ===================================================== */

    .hero-wrapper {
        position: relative;
        overflow: hidden;
        padding: 70px 30px 55px 30px;
        margin-bottom: 25px;
        border-radius: 35px;
        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.95),
                rgba(240,253,244,0.88)
            );
        border: 1px solid rgba(16,185,129,0.15);
        box-shadow:
            0 25px 80px rgba(15,23,42,0.08);
        text-align: center;
    }

    .hero-wrapper::before {
        content: "";
        position: absolute;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: rgba(16,185,129,0.10);
        top: -170px;
        left: -100px;
    }

    .hero-wrapper::after {
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        border-radius: 50%;
        background: rgba(59,130,246,0.08);
        right: -100px;
        bottom: -140px;
    }

    .hero-content {
        position: relative;
        z-index: 2;
    }

    .hero-badge {
        display: inline-block;
        padding: 9px 18px;
        border-radius: 999px;
        background: rgba(16,185,129,0.10);
        border: 1px solid rgba(16,185,129,0.20);
        color: #047857;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.8px;
        margin-bottom: 22px;
    }

    .hero-title {
        margin: 0;
        font-size: clamp(42px, 7vw, 82px);
        line-height: 0.98;
        font-weight: 950;
        letter-spacing: -4px;
        color: #0f172a;
    }

    .hero-title span {
        background:
            linear-gradient(
                90deg,
                #047857,
                #10b981,
                #059669
            );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        max-width: 820px;
        margin: 25px auto 0 auto;
        font-size: 18px;
        line-height: 1.75;
        color: #475569;
    }

    .hero-tagline {
        margin-top: 25px;
        font-size: 14px;
        font-weight: 700;
        color: #64748b;
    }


    /* =====================================================
       STAT CARDS
       ===================================================== */

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 25px 0 45px 0;
    }

    .stat-card {
        position: relative;
        overflow: hidden;
        padding: 25px 18px;
        border-radius: 24px;
        background: rgba(255,255,255,0.80);
        border: 1px solid rgba(15,23,42,0.07);
        box-shadow:
            0 12px 35px rgba(15,23,42,0.055);
        text-align: center;
        transition: transform 0.2s ease;
    }

    .stat-card:hover {
        transform: translateY(-4px);
    }

    .stat-icon {
        font-size: 25px;
        margin-bottom: 8px;
    }

    .stat-number {
        font-size: 29px;
        font-weight: 950;
        color: #0f172a;
    }

    .stat-label {
        margin-top: 5px;
        color: #64748b;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }


    /* =====================================================
       SECTION
       ===================================================== */

    .section-heading {
        margin-top: 45px;
        margin-bottom: 6px;
        font-size: 30px;
        font-weight: 900;
        letter-spacing: -0.8px;
        color: #0f172a;
    }

    .section-description {
        margin-bottom: 20px;
        color: #64748b;
        font-size: 15px;
        line-height: 1.7;
    }


    /* =====================================================
       UPLOAD AREA
       ===================================================== */

    .upload-container {
        padding: 8px;
        border-radius: 30px;
        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.95),
                rgba(240,253,250,0.88)
            );
        border: 1px solid rgba(16,185,129,0.15);
        box-shadow:
            0 20px 60px rgba(15,23,42,0.07);
    }

    [data-testid="stFileUploader"] {
        padding: 12px;
    }

    [data-testid="stFileUploaderDropzone"] {
        min-height: 230px !important;
        border-radius: 25px !important;
        border: 2px dashed rgba(16,185,129,0.35) !important;
        background:
            rgba(240,253,244,0.55) !important;
        transition: all 0.2s ease;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(16,185,129,0.75) !important;
        background:
            rgba(236,253,245,0.90) !important;
    }


    /* =====================================================
       BUTTON
       ===================================================== */

    .stButton > button {
        width: 100%;
        min-height: 56px;
        border-radius: 17px;
        font-size: 17px;
        font-weight: 850;
        border: none;
        box-shadow:
            0 10px 25px rgba(16,185,129,0.18);
    }


    /* =====================================================
       PREVIEW CARD
       ===================================================== */

    .preview-card {
        padding: 20px;
        border-radius: 28px;
        background: rgba(255,255,255,0.90);
        border: 1px solid rgba(15,23,42,0.07);
        box-shadow:
            0 15px 45px rgba(15,23,42,0.07);
    }


    /* =====================================================
       READY CARD
       ===================================================== */

    .ready-card {
        padding: 32px;
        border-radius: 28px;
        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.95),
                rgba(240,253,250,0.90)
            );
        border: 1px solid rgba(16,185,129,0.14);
        box-shadow:
            0 15px 45px rgba(15,23,42,0.06);
    }

    .ready-icon {
        font-size: 50px;
        margin-bottom: 10px;
    }

    .ready-title {
        font-size: 25px;
        font-weight: 900;
        color: #0f172a;
    }

    .ready-text {
        margin-top: 10px;
        color: #64748b;
        line-height: 1.7;
    }


    /* =====================================================
       RESULT
       ===================================================== */

    .result-card {
        position: relative;
        overflow: hidden;
        padding: 45px 25px;
        margin-top: 20px;
        border-radius: 32px;
        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.98),
                rgba(240,253,244,0.92)
            );
        border: 1px solid rgba(16,185,129,0.20);
        box-shadow:
            0 25px 80px rgba(15,23,42,0.10);
        text-align: center;
    }

    .result-card::before {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: rgba(16,185,129,0.09);
        left: -60px;
        top: -70px;
    }

    .result-icon {
        position: relative;
        font-size: 72px;
    }

    .result-label {
        margin-top: 10px;
        font-size: 12px;
        font-weight: 900;
        letter-spacing: 2px;
        color: #64748b;
        text-transform: uppercase;
    }

    .result-name {
        margin-top: 8px;
        font-size: clamp(32px, 5vw, 55px);
        font-weight: 950;
        letter-spacing: -2px;
        color: #0f172a;
    }

    .result-category {
        margin-top: 5px;
        font-size: 17px;
        color: #64748b;
    }


    /* =====================================================
       CONFIDENCE
       ===================================================== */

    .confidence-card {
        padding: 28px;
        border-radius: 25px;
        background: rgba(255,255,255,0.90);
        border: 1px solid rgba(15,23,42,0.07);
        box-shadow:
            0 12px 40px rgba(15,23,42,0.06);
    }

    .confidence-number {
        font-size: 44px;
        font-weight: 950;
        text-align: center;
        color: #047857;
    }

    .confidence-label {
        text-align: center;
        margin-top: 3px;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1px;
        color: #64748b;
    }


    /* =====================================================
       INFORMATION CARDS
       ===================================================== */

    .info-card {
        min-height: 210px;
        padding: 28px;
        border-radius: 26px;
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(15,23,42,0.07);
        box-shadow:
            0 12px 40px rgba(15,23,42,0.055);
    }

    .info-icon {
        font-size: 40px;
    }

    .info-title {
        margin-top: 12px;
        font-size: 21px;
        font-weight: 900;
        color: #0f172a;
    }

    .info-text {
        margin-top: 10px;
        color: #64748b;
        line-height: 1.75;
    }


    /* =====================================================
       PREDICTION BAR AREA
       ===================================================== */

    .prediction-card {
        padding: 25px;
        border-radius: 25px;
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(15,23,42,0.07);
        box-shadow:
            0 12px 40px rgba(15,23,42,0.05);
    }

    .prediction-title {
        font-size: 18px;
        font-weight: 850;
        color: #0f172a;
    }


    /* =====================================================
       CATEGORY CHIPS
       ===================================================== */

    .chips {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 15px;
    }

    .chip {
        padding: 11px 17px;
        border-radius: 999px;
        background: rgba(16,185,129,0.08);
        border: 1px solid rgba(16,185,129,0.17);
        color: #047857;
        font-size: 13px;
        font-weight: 800;
    }


    /* =====================================================
       HOW IT WORKS
       ===================================================== */

    .step-card {
        min-height: 250px;
        padding: 27px;
        border-radius: 26px;
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(15,23,42,0.07);
        box-shadow:
            0 12px 40px rgba(15,23,42,0.05);
    }

    .step-number {
        font-size: 12px;
        font-weight: 900;
        color: #10b981;
        letter-spacing: 1px;
    }

    .step-icon {
        font-size: 42px;
        margin-top: 18px;
    }

    .step-title {
        margin-top: 12px;
        font-size: 20px;
        font-weight: 900;
        color: #0f172a;
    }

    .step-text {
        margin-top: 8px;
        color: #64748b;
        line-height: 1.7;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer-box {
        margin-top: 60px;
        padding: 38px 20px 15px 20px;
        border-top: 1px solid rgba(15,23,42,0.08);
        text-align: center;
        color: #64748b;
    }

    .footer-logo {
        font-size: 38px;
    }

    .footer-title {
        margin-top: 8px;
        font-size: 20px;
        font-weight: 900;
        color: #0f172a;
    }

    .footer-subtitle {
        margin-top: 5px;
        font-size: 13px;
    }

    .footer-note {
        margin-top: 15px;
        font-size: 11px;
    }


    /* =====================================================
       MOBILE
       ===================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
            padding-top: 0.8rem;
        }

        .hero-wrapper {
            padding: 45px 18px 40px 18px;
            border-radius: 26px;
        }

        .hero-title {
            font-size: 47px;
            letter-spacing: -2px;
        }

        .hero-subtitle {
            font-size: 15px;
        }

        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }

        .stat-card {
            padding: 20px 10px;
        }

        .stat-number {
            font-size: 23px;
        }

        .section-heading {
            font-size: 25px;
        }

        .upload-container {
            border-radius: 22px;
        }

        .result-card {
            padding: 35px 15px;
            border-radius: 25px;
        }

        .result-icon {
            font-size: 60px;
        }

        .confidence-number {
            font-size: 38px;
        }

        .info-card,
        .step-card {
            min-height: auto;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


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
        with open(CLASS_NAMES_PATH, "r") as file:
            classes = json.load(file)

        if isinstance(classes, list) and len(classes) > 0:
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
        return tf.keras.models.load_model(MODEL_PATH)

    except Exception as error:
        st.error(f"Model loading error: {error}")
        return None


model = load_model()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-wrapper">

        <div class="hero-content">

            <div class="hero-badge">
                🌍 SCIENCE EXHIBITION • ARTIFICIAL INTELLIGENCE
            </div>

            <h1 class="hero-title">
                ♻️ <span>WasteWise AI</span>
            </h1>

            <div class="hero-subtitle">
                An intelligent computer-vision assistant that analyzes
                waste images and predicts their category using a trained
                deep-learning model.
            </div>

            <div class="hero-tagline">
                🌱 Learn • Classify • Dispose Responsibly
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
            <div class="stat-icon">🧠</div>
            <div class="stat-number">AI</div>
            <div class="stat-label">
                Computer Vision
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
            <div class="stat-icon">📐</div>
            <div class="stat-number">224²</div>
            <div class="stat-label">
                Image Input
            </div>
        </div>

        <div class="stat-card">
            <div class="stat-icon">🌱</div>
            <div class="stat-number">AI +</div>
            <div class="stat-label">
                Sustainability
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL STATUS
# ============================================================

if model is None:

    st.error(
        "⚠️ WasteWise AI model is not available."
    )

    st.info(
        "Make sure these files exist inside your project:"
    )

    st.code(
        """
model/
├── wastewise_model.keras
└── class_names.json
        """
    )

    st.stop()


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-heading">📸 Analyze Your Waste</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-description">
        Upload a clear photo of a waste item. WasteWise AI will
        process the image and generate a predicted waste category
        with confidence information.
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="upload-container">',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Drag & drop your waste image here",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    help="Use a clear, well-lit image where the waste item is easy to see."
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# IMAGE ANALYSIS
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
        '<div class="section-heading">🖼️ Image Analysis</div>',
        unsafe_allow_html=True
    )

    preview_col, action_col = st.columns(
        [1.15, 1],
        gap="large"
    )


    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    with preview_col:

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
            '</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    with action_col:

        st.markdown(
            """
            <div class="ready-card">

                <div class="ready-icon">
                    🤖
                </div>

                <div class="ready-title">
                    Image Ready
                </div>

                <div class="ready-text">
                    Your image has been successfully loaded.
                    Click the button below to run the trained
                    WasteWise AI classification model.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        analyze_button = st.button(
            "🔍  ANALYZE WASTE",
            type="primary",
            use_container_width=True
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    if analyze_button:

        with st.spinner(
            "🤖 WasteWise AI is analyzing the image..."
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
            predictions[predicted_index]
        )

        info = WASTE_INFO.get(
            predicted_class,
            {
                "icon": "♻️",
                "short": predicted_class,
                "category": predicted_class,
                "description": (
                    "No additional information is available "
                    "for this category."
                ),
                "tips": []
            }
        )


        # ====================================================
        # AI RESULT
        # ====================================================

        st.markdown(
            '<div class="section-heading">🤖 AI Classification Result</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-icon">
                    {info["icon"]}
                </div>

                <div class="result-label">
                    WasteWise AI Prediction
                </div>

                <div class="result-name">
                    {predicted_class}
                </div>

                <div class="result-category">
                    {info["category"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        st.markdown(
            '<div class="section-heading">📊 Prediction Confidence</div>',
            unsafe_allow_html=True
        )

        confidence_col1, confidence_col2 = st.columns(
            [1, 2],
            gap="large"
        )


        with confidence_col1:

            st.markdown(
                f"""
                <div class="confidence-card">

                    <div class="confidence-number">
                        {confidence * 100:.1f}%
                    </div>

                    <div class="confidence-label">
                        MODEL CONFIDENCE
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with confidence_col2:

            st.markdown(
                '<div class="prediction-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="prediction-title">Confidence Level</div>',
                unsafe_allow_html=True
            )

            st.write("")

            st.progress(
                min(confidence, 1.0)
            )

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

                st.caption(
                    "Try a clearer image with the waste item "
                    "centered, visible, and well illuminated."
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # SMART GUIDANCE
        # ====================================================

        st.markdown(
            '<div class="section-heading">💡 Smart Waste Guidance</div>',
            unsafe_allow_html=True
        )

        guidance_col1, guidance_col2 = st.columns(
            2,
            gap="large"
        )


        with guidance_col1:

            st.markdown(
                f"""
                <div class="info-card">

                    <div class="info-icon">
                        {info["icon"]}
                    </div>

                    <div class="info-title">
                        What Does This Mean?
                    </div>

                    <div class="info-text">
                        {info["description"]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with guidance_col2:

            tips_html = ""

            for tip in info["tips"]:

                tips_html += (
                    f"<div style='margin:10px 0;'>"
                    f"✅ {tip}"
                    f"</div>"
                )

            st.markdown(
                f"""
                <div class="info-card">

                    <div class="info-icon">
                        🌱
                    </div>

                    <div class="info-title">
                        Recommended Practices
                    </div>

                    <div class="info-text">
                        {tips_html}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # CONFIDENCE BREAKDOWN
        # ====================================================

        st.markdown(
            '<div class="section-heading">📈 AI Confidence Breakdown</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="section-description">
                These values show how strongly the model scored
                each supported waste category.
            </div>
            """,
            unsafe_allow_html=True
        )


        sorted_indices = np.argsort(
            predictions
        )[::-1]


        for rank, index in enumerate(
            sorted_indices,
            start=1
        ):

            if index >= len(CLASS_NAMES):
                continue

            class_name = CLASS_NAMES[index]

            probability = float(
                predictions[index]
            )

            class_info = WASTE_INFO.get(
                class_name,
                {}
            )

            icon = class_info.get(
                "icon",
                "♻️"
            )

            st.markdown(
                f"""
                <div style="
                    margin-top:18px;
                    margin-bottom:5px;
                    font-weight:850;
                    color:#0f172a;
                ">
                    {rank}. {icon} {class_name}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                min(probability, 1.0)
            )

            st.caption(
                f"{probability * 100:.2f}% confidence"
            )


        # ====================================================
        # EDUCATIONAL NOTICE
        # ====================================================

        st.markdown("---")

        st.warning(
            """
            ⚠️ **Educational AI Notice**

            WasteWise AI is a science-exhibition machine-learning
            project. Its prediction is based on visual patterns
            learned from the training dataset.

            The prediction should not be treated as a definitive
            determination of material composition, recyclability,
            hazardousness, or safety.

            Always follow local waste-management rules and seek
            appropriate professional guidance when dealing with
            potentially hazardous materials.
            """
        )


# ============================================================
# HOW WASTEWISE AI WORKS
# ============================================================

st.markdown(
    '<div class="section-heading">🧠 How WasteWise AI Works</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-description">
        Four simple steps transform a waste image into an
        AI-powered classification.
    </div>
    """,
    unsafe_allow_html=True
)


how_col1, how_col2, how_col3, how_col4 = st.columns(
    4,
    gap="medium"
)


steps = [

    (
        "STEP 01",
        "📸",
        "Capture",
        "Upload a clear image of the waste item."
    ),

    (
        "STEP 02",
        "🧠",
        "Analyze",
        "EfficientNetB0 extracts important visual features."
    ),

    (
        "STEP 03",
        "🤖",
        "Classify",
        "The trained neural network predicts the waste category."
    ),

    (
        "STEP 04",
        "🌱",
        "Act",
        "Use the result to learn about responsible waste handling."
    )

]


for column, step in zip(
    [
        how_col1,
        how_col2,
        how_col3,
        how_col4
    ],
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
# SUPPORTED CATEGORIES
# ============================================================

st.markdown(
    '<div class="section-heading">♻️ Supported Waste Categories</div>',
    unsafe_allow_html=True
)

chips_html = '<div class="chips">'

for class_name in CLASS_NAMES:

    info = WASTE_INFO.get(
        class_name,
        {}
    )

    chips_html += (
        f'<div class="chip">'
        f'{info.get("icon", "♻️")} '
        f'{class_name}'
        f'</div>'
    )

chips_html += "</div>"

st.markdown(
    chips_html,
    unsafe_allow_html=True
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-heading">🔬 AI Model Information</div>',
    unsafe_allow_html=True
)

with st.expander(
    "Open technical details"
):

    model_col1, model_col2 = st.columns(
        2,
        gap="large"
    )

    with model_col1:

        st.write(
            "**Architecture:** EfficientNetB0"
        )

        st.write(
            "**Input:** 224 × 224 × 3"
        )

        st.write(
            "**Learning Method:** Transfer Learning"
        )

        st.write(
            "**Task:** Image Classification"
        )


    with model_col2:

        st.write(
            "**Output Classes:** 4"
        )

        st.write(
            "**Model Format:** Keras `.keras`"
        )

        st.write(
            "**Model File:** `wastewise_model.keras`"
        )

        st.write(
            "**Class File:** `class_names.json`"
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
            AI for Smarter Waste Classification & Awareness
        </div>

        <div class="footer-note">
            Built for Science Exhibition • Deep Learning •
            Computer Vision • Sustainability
        </div>

    </div>
    """,
    unsafe_allow_html=True
)
```
