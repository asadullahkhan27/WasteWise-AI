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
# PAGE CONFIG
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

# Logo paths
ICODEGURU_LOGO = BASE_DIR / "assets" / "icodeguru_logo.png"
SCHOOL_LOGO = BASE_DIR / "assets" / "school_logo.png"

IMAGE_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# CSS
# ============================================================

CUSTOM_CSS = """
<style>

.stApp {
    background: #07130f;
    color: #ffffff;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */

.brand-wrapper {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0 20px 0;
}

.brand-left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-icon {
    width: 55px;
    height: 55px;
    border-radius: 15px;
    background: linear-gradient(
        135deg,
        #16a34a,
        #059669
    );
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
}

.brand-name {
    font-size: 25px;
    font-weight: 800;
}

.brand-subtitle {
    color: #91b6a1;
    font-size: 13px;
}

.logo-box {
    display: flex;
    align-items: center;
    gap: 15px;
}

.logo-box img {
    height: 50px;
    width: auto;
    object-fit: contain;
    border-radius: 8px;
}


/* Hero */

.hero-wrapper {
    margin-top: 20px;
    padding: 50px 30px;
    border-radius: 25px;

    background:
        linear-gradient(
            135deg,
            rgba(22,163,74,0.20),
            rgba(5,150,105,0.08)
        );

    border: 1px solid rgba(74,222,128,0.18);

    text-align: center;
}

.hero-badge {
    display: inline-block;
    padding: 8px 15px;
    border-radius: 50px;

    background: rgba(34,197,94,0.12);

    color: #86efac;

    font-size: 13px;
    font-weight: 700;

    margin-bottom: 18px;
}

.hero-title {
    font-size: 60px;
    font-weight: 900;
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
    max-width: 750px;
    margin: 20px auto;

    color: #b9d8c5;

    font-size: 17px;
    line-height: 1.7;
}

.hero-tagline {
    color: #86efac;
    font-weight: 700;
}


/* Stats */

.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
    margin-top: 25px;
}

.stat-card {
    padding: 20px;

    background: rgba(13,37,28,0.8);

    border: 1px solid rgba(255,255,255,0.07);

    border-radius: 18px;

    text-align: center;
}

.stat-icon {
    font-size: 27px;
}

.stat-number {
    font-size: 25px;
    font-weight: 800;
    color: #86efac;
}

.stat-label {
    color: #91b6a1;
    font-size: 12px;
}


/* Section */

.section-heading {
    margin-top: 40px;
    margin-bottom: 8px;

    font-size: 28px;
    font-weight: 800;
}

.section-description {
    color: #91b6a1;
    margin-bottom: 20px;
}


/* Upload */

section[data-testid="stFileUploaderDropzone"] {
    background: rgba(13,37,28,0.65);
    border: 1px dashed rgba(74,222,128,0.30);
    border-radius: 18px;
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

    padding: 14px 20px;

    background: linear-gradient(
        135deg,
        #16a34a,
        #059669
    );

    color: white;

    font-weight: 800;
    font-size: 16px;
}

.stButton > button:hover {
    transform: translateY(-2px);
}


/* Preview */

.preview-card {
    padding: 20px;

    margin-top: 20px;

    border-radius: 18px;

    background: rgba(13,37,28,0.75);

    border: 1px solid rgba(255,255,255,0.07);
}

.preview-title {
    font-size: 19px;
    font-weight: 800;
}

.preview-text {
    color: #91b6a1;
    font-size: 13px;
}


/* Result */

.result-box {
    margin-top: 25px;

    padding: 30px;

    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            rgba(22,163,74,0.14),
            rgba(5,150,105,0.06)
        );

    border: 1px solid rgba(74,222,128,0.20);
}

.result-icon {
    font-size: 55px;
}

.result-title {
    font-size: 32px;
    font-weight: 900;
}

.confidence {
    display: inline-block;

    margin-top: 10px;

    padding: 7px 14px;

    border-radius: 50px;

    background: rgba(34,197,94,0.13);

    color: #86efac;

    font-weight: 800;
}

.result-category {
    margin-top: 15px;

    color: #86efac;

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

    padding: 20px;

    border-radius: 18px;

    background: rgba(8,27,20,0.85);

    border: 1px solid rgba(255,255,255,0.06);
}

.tip-title {
    font-size: 18px;
    font-weight: 800;
}

.tip-item {
    padding: 7px 0;
    color: #b9d8c5;
}


/* Information Cards */

.info-card {
    padding: 25px;

    border-radius: 20px;

    background: rgba(13,37,28,0.75);

    border: 1px solid rgba(255,255,255,0.07);

    height: 100%;
}

.info-icon {
    font-size: 35px;
}

.info-title {
    font-size: 20px;
    font-weight: 800;
}

.info-text {
    margin-top: 8px;

    color: #94b9a3;

    line-height: 1.7;
}


/* Steps */

.step-card {
    padding: 25px;

    border-radius: 20px;

    background: rgba(13,37,28,0.70);

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

    font-weight: 900;
}

.step-icon {
    font-size: 28px;
    margin-top: 12px;
}

.step-title {
    font-size: 18px;
    font-weight: 800;
}

.step-text {
    margin-top: 8px;

    color: #91b6a1;

    line-height: 1.6;
}


/* Disclaimer */

.disclaimer {
    margin-top: 35px;

    padding: 18px;

    border-radius: 15px;

    background: rgba(120,53,15,0.16);

    border: 1px solid rgba(251,191,36,0.18);

    color: #d9c58c;

    font-size: 13px;

    line-height: 1.7;
}


/* Footer */

.footer-box {
    margin-top: 50px;

    padding: 30px;

    border-top: 1px solid rgba(255,255,255,0.08);

    text-align: center;
}

.footer-title {
    font-size: 22px;
    font-weight: 800;
}

.footer-subtitle {
    color: #91b6a1;
}

.footer-small {
    margin-top: 15px;

    color: #617f6e;

    font-size: 12px;
}


/* Progress */

div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(
        90deg,
        #16a34a,
        #34d399
    );
}


/* Responsive */

@media (max-width: 800px) {

    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .hero-title {
        font-size: 42px;
    }

    .brand-wrapper {
        flex-direction: column;
        align-items: flex-start;
        gap: 20px;
    }
}

@media (max-width: 500px) {

    .stats-grid {
        grid-template-columns: 1fr;
    }

    .hero-title {
        font-size: 36px;
    }

}

</style>
"""


# Apply CSS
st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTION FOR LOGOS
# ============================================================

def image_to_base64(image_path):

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
# HEADER WITH LOGOS
# ============================================================

icodeguru_html = ""

if icodeguru_logo:

    icodeguru_html = f"""
    <img
        src="data:image/png;base64,{icodeguru_logo}"
        alt="iCodeGuru Logo"
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
        alt="School Logo"
    >
    """

else:

    school_html = """
    <span>School</span>
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
# HERO
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

            An AI-powered waste classification and
            recycling assistant that helps identify
            waste categories from images.

        </div>

        <div class="hero-tagline">

            Turn Waste into Better Decisions.

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
                Input Methods
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
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ============================================================
# CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

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
# IMAGE INPUT
# ============================================================

st.markdown(
    '<div class="section-heading">🔍 Analyze Your Waste</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-description">

        Upload a waste image or use your camera
        to analyze the item with AI.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD + CAMERA TABS
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
# UPLOAD
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
        help="Upload a clear image of a waste item."
    )


# ============================================================
# CAMERA
# ============================================================

with camera_tab:

    camera_file = st.camera_input(
        "Take a picture of the waste item"
    )


# ============================================================
# SELECT IMAGE
# ============================================================

if camera_file is not None:

    selected_file = camera_file

elif uploaded_file is not None:

    selected_file = uploaded_file

else:

    selected_file = None


# ============================================================
# ANALYSIS
# ============================================================

if selected_file is not None:

    image = Image.open(
        selected_file
    ).convert("RGB")


    # ========================================================
    # PREVIEW
    # ========================================================

    st.markdown(
        """
        <div class="preview-card">

            <div class="preview-title">
                🖼️ Selected Waste Image
            </div>

            <div class="preview-text">
                Image ready for AI analysis.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.image(
        image,
        use_container_width=True
    )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "♻️ Analyze Waste",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "🤖 AI is analyzing your waste..."
            ):

                # Load model
                model = load_model()

                # Load class names
                class_names = load_class_names()


                # ================================================
                # PREPROCESS IMAGE
                # ================================================

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


                # ================================================
                # PREDICTION
                # ================================================

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]


                predicted_index = int(
                    np.argmax(predictions)
                )


                if predicted_index >= len(class_names):

                    raise ValueError(
                        "Model output does not match "
                        "class_names.json."
                    )


                predicted_class = class_names[
                    predicted_index
                ]


                confidence = float(
                    predictions[predicted_index]
                )


                # ================================================
                # RESULT INFORMATION
                # ================================================

                info = WASTE_INFO.get(
                    predicted_class
                )


                if info is None:

                    info = {

                        "icon": "♻️",

                        "category":
                        predicted_class,

                        "message":
                        "Prediction completed.",

                        "tips": []
                    }


                # ================================================
                # RESULT
                # ================================================

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


                # ================================================
                # CONFIDENCE WARNING
                # ================================================

                if confidence < CONFIDENCE_THRESHOLD:

                    st.warning(
                        "⚠️ The model confidence is below "
                        f"{CONFIDENCE_THRESHOLD * 100:.0f}%. "
                        "Please verify the prediction manually."
                    )

                else:

                    st.success(
                        "✅ The model has sufficient confidence "
                        "for this demonstration."
                    )


                # ================================================
                # TIPS
                # ================================================

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


                # ================================================
                # PREDICTION BREAKDOWN
                # ================================================

                st.markdown(
                    '<div class="section-heading">📊 Prediction Breakdown</div>',
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
                Make sure your model exists here:

                model/wastewise_model.keras

                And class names exist here:

                model/class_names.json
                """
            )


        except Exception as error:

            st.error(
                "❌ An error occurred while "
                "analyzing the image."
            )

            st.code(
                str(error)
            )


else:

    st.info(
        "📷 Upload an image or use the camera to start."
    )


# ============================================================
# SUPPORTED CATEGORIES
# ============================================================

st.markdown(
    '<div class="section-heading">♻️ Supported Waste Categories</div>',
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
                Materials that may require special
                handling or disposal procedures.
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
                Biodegradable material such as food
                and other organic waste.
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
                Waste that may not be accepted by
                common recycling systems.
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
                on local facilities and rules.
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


step1, step2, step3, step4 = st.columns(4)


with step1:

    st.markdown(
        """
        <div class="step-card">

            <div class="step-number">1</div>

            <div class="step-icon">
                📷
            </div>

            <div class="step-title">
                Capture
            </div>

            <div class="step-text">
                Upload an image or take a photo
                using the camera.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with step2:

    st.markdown(
        """
        <div class="step-card">

            <div class="step-number">2</div>

            <div class="step-icon">
                🧠
            </div>

            <div class="step-title">
                AI Processing
            </div>

            <div class="step-text">
                The trained EfficientNetB0 model
                processes the image.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with step3:

    st.markdown(
        """
        <div class="step-card">

            <div class="step-number">3</div>

            <div class="step-icon">
                🔍
            </div>

            <div class="step-title">
                Classification
            </div>

            <div class="step-text">
                AI predicts one of the four
                waste categories.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with step4:

    st.markdown(
        """
        <div class="step-card">

            <div class="step-number">4</div>

            <div class="step-icon">
                🌱
            </div>

            <div class="step-title">
                Guidance
            </div>

            <div class="step-text">
                Get practical waste-management
                guidance based on the result.
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
        Predictions should not be treated as definitive
        determinations of material safety or recyclability.

        Always follow official local waste-management
        and recycling guidelines.

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
            AI-Powered Waste Classification & Recycling Assistant
        </div>

        <div class="footer-small">

            Built with Python • TensorFlow • Streamlit
            • EfficientNetB0

            <br><br>

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
