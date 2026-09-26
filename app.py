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
    layout="centered",
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

# Logo paths
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
# SIMPLE CUSTOM CSS
# ============================================================
# CSS is inside a Python string.
# It will be applied to Streamlit and will NOT appear as code.
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 20px;
        opacity: 0.85;
    }

    .logo-row {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 25px;
        margin-bottom: 15px;
    }

    .logo-row img {
        height: 55px;
        width: auto;
        object-fit: contain;
    }

    .result-box {
        padding: 25px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .result-title {
        font-size: 28px;
        font-weight: 700;
        text-align: center;
    }

    .confidence {
        font-size: 22px;
        font-weight: 600;
        text-align: center;
        margin-top: 10px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 25px;
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        opacity: 0.75;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOGO HELPER
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
# LOGO DISPLAY
# ============================================================

logo_html = ""


if icodeguru_logo:

    logo_html += f"""
    <img
        src="data:image/png;base64,{icodeguru_logo}"
        alt="iCodeGuru Logo"
    >
    """


if school_logo:

    logo_html += f"""
    <img
        src="data:image/png;base64,{school_logo}"
        alt="School Logo"
    >
    """


if logo_html:

    st.markdown(
        f"""
        <div class="logo-row">

            {logo_html}

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    if not CLASS_NAMES_PATH.exists():

        return [
            "Hazardous",
            "Non-Recyclable",
            "Organic",
            "Recyclable"
        ]

    try:

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return [
            "Hazardous",
            "Non-Recyclable",
            "Organic",
            "Recyclable"
        ]


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
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        ♻️ WasteWise AI
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="subtitle">
        AI-Powered Waste Classification & Recycling Assistant
    </div>
    """,
    unsafe_allow_html=True
)


st.info(
    "Upload a waste image or use your camera and "
    "WasteWise AI will classify it into one of four trained categories."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("♻️ WasteWise AI")

    st.markdown(
        "### About the Project"
    )

    st.write(
        """
        WasteWise AI uses a trained EfficientNetB0
        image-classification model to identify waste
        categories from images.
        """
    )

    st.markdown(
        "### Supported Classes"
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
        "Model: EfficientNetB0"
    )

    st.caption(
        "Image Size: 224 × 224"
    )

    st.caption(
        "Classes: 4"
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
        Please make sure your project has:

        `model/wastewise_model.keras`

        and

        `model/class_names.json`
        """
    )

    st.stop()


# ============================================================
# IMAGE INPUT
# ============================================================

st.markdown(
    "### 📤 Select Waste Image"
)


# ============================================================
# UPLOAD / CAMERA TABS
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

        "Choose an image",

        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],

        help="Upload a clear image of the waste item.",

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
# SELECT INPUT
# ============================================================

if camera_file is not None:

    selected_file = camera_file

elif uploaded_file is not None:

    selected_file = uploaded_file

else:

    selected_file = None


# ============================================================
# IMAGE PROCESSING
# ============================================================

if selected_file is not None:

    try:

        image = Image.open(
            selected_file
        ).convert("RGB")

    except Exception as error:

        st.error(
            f"Could not open image: {error}"
        )

        st.stop()


    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    st.markdown(
        "### 🖼️ Selected Image"
    )

    st.image(
        image,
        caption="Waste Image",
        width="stretch"
    )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    analyze_button = st.button(

        "🔍 Analyze Waste",

        type="primary",

        width="stretch"
    )


    # ========================================================
    # ANALYZE
    # ========================================================

    if analyze_button:

        with st.spinner(
            "🤖 WasteWise AI is analyzing the image..."
        ):

            # ------------------------------------------------
            # RESIZE IMAGE
            # ------------------------------------------------

            processed_image = image.resize(
                IMAGE_SIZE
            )


            # ------------------------------------------------
            # CONVERT TO NUMPY
            # ------------------------------------------------

            image_array = np.array(
                processed_image
            ).astype("float32")


            # ------------------------------------------------
            # ADD BATCH DIMENSION
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
            # TOP PREDICTION
            # ------------------------------------------------

            predicted_index = int(
                np.argmax(predictions)
            )


            if predicted_index >= len(CLASS_NAMES):

                st.error(
                    "Model output does not match "
                    "class_names.json."
                )

                st.stop()


            predicted_class = CLASS_NAMES[
                predicted_index
            ]


            confidence = float(
                predictions[predicted_index]
            )


        # ====================================================
        # RESULT
        # ====================================================

        info = WASTE_INFO.get(

            predicted_class,

            {
                "icon": "♻️",

                "category":
                    predicted_class,

                "message":
                    "No additional information available.",

                "tips": []
            }
        )


        icon = info["icon"]

        category = info["category"]


        st.markdown(
            """
            <div class="section-title">
                🤖 AI Prediction
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="result-box">

                <div class="result-title">

                    {icon} {predicted_class}

                </div>

                <div class="confidence">

                    Confidence:
                    {confidence * 100:.2f}%

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # CONFIDENCE STATUS
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
                "Try uploading a clearer image with "
                "the waste item more visible."
            )


        # ====================================================
        # CATEGORY
        # ====================================================

        st.markdown(
            "### 🏷️ Waste Category"
        )

        st.write(
            f"**{category}**"
        )


        # ====================================================
        # GUIDANCE
        # ====================================================

        st.markdown(
            "### 💡 Waste Management Guidance"
        )

        st.info(
            info["message"]
        )


        # ====================================================
        # TIPS
        # ====================================================

        if info["tips"]:

            st.markdown(
                "### ✅ Recommended Practices"
            )

            for tip in info["tips"]:

                st.write(
                    f"• {tip}"
                )


        # ====================================================
        # TOP 3 PREDICTIONS
        # ====================================================

        st.markdown(
            "### 📊 Top Predictions"
        )


        sorted_indices = np.argsort(
            predictions
        )[::-1]


        for rank, index in enumerate(

            sorted_indices[:3],

            start=1

        ):

            if index >= len(CLASS_NAMES):
                continue


            class_name = CLASS_NAMES[
                index
            ]


            probability = float(
                predictions[index]
            )


            st.write(
                f"**{rank}. {class_name}**"
            )


            st.progress(
                min(probability, 1.0)
            )


            st.caption(
                f"{probability * 100:.2f}%"
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.markdown("---")

        st.warning(
            """
            ⚠️ **Important:** WasteWise AI is an educational
            machine-learning project. Its prediction should
            not be treated as a definitive determination of
            material composition, recyclability, hazardousness,
            or microbiological safety. Always follow local
            waste-management rules and use appropriate
            professional guidance for hazardous materials.
            """
        )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.markdown(
        "### 📸 How to Use"
    )

    st.write(
        """
        1. Choose **Upload Image** or **Use Camera**.
        2. Provide a clear image of the waste item.
        3. Click **Analyze Waste**.
        4. WasteWise AI predicts the waste category.
        5. Review the confidence score.
        6. Follow the displayed waste-management guidance.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        ♻️ <b>WasteWise AI</b>
        <br>

        AI for Smarter Waste Classification & Awareness

    </div>
    """,
    unsafe_allow_html=True
)
