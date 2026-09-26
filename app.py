# # ============================================================
# # WASTEWISE AI
# # AI-Powered Waste Classification & Recycling Assistant
# # ============================================================

# import json
# from pathlib import Path

# import numpy as np
# import streamlit as st
# import tensorflow as tf
# from PIL import Image


# # ============================================================
# # PAGE CONFIGURATION
# # ============================================================

# st.set_page_config(
#     page_title="WasteWise AI",
#     page_icon="♻️",
#     layout="centered",
#     initial_sidebar_state="expanded"
# )


# # ============================================================
# # PROJECT PATHS
# # ============================================================

# BASE_DIR = Path(__file__).resolve().parent

# MODEL_PATH = (
#     BASE_DIR
#     / "model"
#     / "wastewise_model.keras"
# )

# CLASS_NAMES_PATH = (
#     BASE_DIR
#     / "model"
#     / "class_names.json"
# )

# # Logo paths
# ICODEGURU_LOGO = (
#     BASE_DIR
#     / "assets"
#     / "icodeguru_logo.png"
# )

# SCHOOL_LOGO = (
#     BASE_DIR
#     / "assets"
#     / "school_logo.png"
# )


# # ============================================================
# # MODEL SETTINGS
# # ============================================================

# IMAGE_SIZE = (224, 224)

# CONFIDENCE_THRESHOLD = 0.60


# # ============================================================
# # WASTE INFORMATION
# # ============================================================

# WASTE_INFO = {

#     "Hazardous": {

#         "icon": "☣️",

#         "category":
#             "Handle Carefully / Special Disposal",

#         "message":
#             "Hazardous waste may require special handling. "
#             "Do not mix potentially hazardous materials with "
#             "ordinary household recycling.",

#         "tips": [
#             "Keep hazardous items separate.",
#             "Avoid direct contact with unknown substances.",
#             "Follow local hazardous-waste disposal guidance.",
#             "Do not burn or improperly dump hazardous materials."
#         ]
#     },


#     "Non-Recyclable": {

#         "icon": "🚫",

#         "category":
#             "Non-Recyclable",

#         "message":
#             "This item was classified as non-recyclable "
#             "by the AI model. Disposal options depend on "
#             "local waste-management rules.",

#         "tips": [
#             "Keep it separate from recyclable materials.",
#             "Check local waste-disposal guidelines.",
#             "Avoid contaminating recyclable waste.",
#             "Reduce single-use materials where possible."
#         ]
#     },


#     "Organic": {

#         "icon": "🌱",

#         "category":
#             "Compostable / Organic",

#         "message":
#             "Organic waste can potentially be composted "
#             "when suitable facilities or composting systems "
#             "are available.",

#         "tips": [
#             "Separate organic waste from recyclables.",
#             "Use a suitable composting system where available.",
#             "Keep compostable material free from contamination.",
#             "Follow local composting guidelines."
#         ]
#     },


#     "Recyclable": {

#         "icon": "♻️",

#         "category":
#             "Recyclable",

#         "message":
#             "This item was classified as recyclable. "
#             "Actual recyclability depends on your local "
#             "recycling system and material requirements.",

#         "tips": [
#             "Keep recyclable materials clean and dry.",
#             "Separate materials according to local rules.",
#             "Avoid mixing contaminated waste with recyclables.",
#             "Check your local recycling guidelines."
#         ]
#     }
# }


# # ============================================================
# # SIMPLE CSS
# # ============================================================

# st.markdown(
#     """
#     <style>

#     .main-title {
#         text-align: center;
#         font-size: 42px;
#         font-weight: 800;
#         margin-bottom: 5px;
#     }

#     .subtitle {
#         text-align: center;
#         font-size: 18px;
#         margin-bottom: 20px;
#         opacity: 0.85;
#     }

#     .logo-title {
#         text-align: center;
#         font-size: 16px;
#         font-weight: 600;
#         margin-bottom: 8px;
#     }

#     .result-box {
#         padding: 25px;
#         border-radius: 18px;
#         border: 1px solid rgba(128, 128, 128, 0.25);
#         margin-top: 20px;
#         margin-bottom: 20px;
#     }

#     .result-title {
#         font-size: 28px;
#         font-weight: 700;
#         text-align: center;
#     }

#     .confidence {
#         font-size: 22px;
#         font-weight: 600;
#         text-align: center;
#         margin-top: 10px;
#     }

#     .section-title {
#         font-size: 22px;
#         font-weight: 700;
#         margin-top: 25px;
#     }

#     .footer {
#         text-align: center;
#         margin-top: 40px;
#         padding: 20px;
#         opacity: 0.75;
#     }

#     </style>
#     """,
#     unsafe_allow_html=True
# )


# # ============================================================
# # LOAD CLASS NAMES
# # ============================================================

# @st.cache_data
# def load_class_names():

#     if not CLASS_NAMES_PATH.exists():

#         return [
#             "Hazardous",
#             "Non-Recyclable",
#             "Organic",
#             "Recyclable"
#         ]

#     try:

#         with open(
#             CLASS_NAMES_PATH,
#             "r"
#         ) as file:

#             return json.load(file)

#     except Exception:

#         return [
#             "Hazardous",
#             "Non-Recyclable",
#             "Organic",
#             "Recyclable"
#         ]


# CLASS_NAMES = load_class_names()


# # ============================================================
# # LOAD MODEL
# # ============================================================

# @st.cache_resource
# def load_model():

#     if not MODEL_PATH.exists():
#         return None

#     try:

#         return tf.keras.models.load_model(
#             MODEL_PATH
#         )

#     except Exception as error:

#         st.error(
#             f"Unable to load model: {error}"
#         )

#         return None


# model = load_model()


# # ============================================================
# # HEADER LOGOS
# # ============================================================

# logo_col1, logo_col2 = st.columns(2)

# with logo_col1:

#     if ICODEGURU_LOGO.exists():

#         st.image(
#             str(ICODEGURU_LOGO),
#             width=170
#         )

#     else:

#         st.warning(
#             "iCodeGuru logo not found."
#         )


# with logo_col2:

#     if SCHOOL_LOGO.exists():

#         st.image(
#             str(SCHOOL_LOGO),
#             width=170
#         )

#     else:

#         st.warning(
#             "School logo not found."
#         )


# # ============================================================
# # MAIN HEADER
# # ============================================================

# st.markdown(
#     '<div class="main-title">♻️ WasteWise AI</div>',
#     unsafe_allow_html=True
# )

# st.markdown(
#     """
#     <div class="subtitle">
#         AI-Powered Waste Classification & Recycling Assistant
#     </div>
#     """,
#     unsafe_allow_html=True
# )


# st.info(
#     "Upload a waste image or use your camera. "
#     "WasteWise AI will classify it into one of four "
#     "trained waste categories."
# )


# # ============================================================
# # SIDEBAR
# # ============================================================

# with st.sidebar:

#     st.header("♻️ WasteWise AI")

#     st.markdown("### About the Project")

#     st.write(
#         """
#         WasteWise AI uses a trained EfficientNetB0
#         image-classification model to identify waste
#         categories from images.
#         """
#     )

#     st.markdown("### Supported Classes")

#     for class_name in CLASS_NAMES:

#         info = WASTE_INFO.get(
#             class_name,
#             {}
#         )

#         icon = info.get(
#             "icon",
#             "♻️"
#         )

#         st.write(
#             f"{icon} {class_name}"
#         )

#     st.markdown("---")

#     st.caption(
#         "Model: EfficientNetB0"
#     )

#     st.caption(
#         "Image Size: 224 × 224"
#     )

#     st.caption(
#         "Classes: 4"
#     )


# # ============================================================
# # MODEL CHECK
# # ============================================================

# if model is None:

#     st.error(
#         "⚠️ Trained model not found."
#     )

#     st.markdown(
#         """
#         Please make sure your project has:

#         `model/wastewise_model.keras`

#         and

#         `model/class_names.json`
#         """
#     )

#     st.stop()


# # ============================================================
# # IMAGE INPUT
# # ============================================================

# st.markdown(
#     "### 📤 Select Waste Image"
# )


# upload_tab, camera_tab = st.tabs(
#     [
#         "📁 Upload Image",
#         "📷 Use Camera"
#     ]
# )


# uploaded_file = None
# camera_file = None


# # ============================================================
# # UPLOAD IMAGE
# # ============================================================

# with upload_tab:

#     uploaded_file = st.file_uploader(
#         "Choose an image",
#         type=[
#             "jpg",
#             "jpeg",
#             "png",
#             "webp"
#         ],
#         help="Upload a clear image of the waste item.",
#         key="waste_upload"
#     )


# # ============================================================
# # CAMERA
# # ============================================================

# with camera_tab:

#     camera_file = st.camera_input(
#         "Take a picture of the waste item",
#         key="waste_camera"
#     )


# # ============================================================
# # SELECT INPUT
# # ============================================================

# if camera_file is not None:

#     selected_file = camera_file

# elif uploaded_file is not None:

#     selected_file = uploaded_file

# else:

#     selected_file = None


# # ============================================================
# # IMAGE PROCESSING
# # ============================================================

# if selected_file is not None:

#     try:

#         image = Image.open(
#             selected_file
#         ).convert("RGB")

#     except Exception as error:

#         st.error(
#             f"Could not open image: {error}"
#         )

#         st.stop()


#     # --------------------------------------------------------
#     # DISPLAY IMAGE
#     # --------------------------------------------------------

#     st.markdown(
#         "### 🖼️ Selected Waste Image"
#     )

#     st.image(
#         image,
#         caption="Waste Image",
#         width="stretch"
#     )


#     # --------------------------------------------------------
#     # ANALYZE BUTTON
#     # --------------------------------------------------------

#     analyze_button = st.button(
#         "🔍 Analyze Waste",
#         type="primary",
#         width="stretch"
#     )


#     # ========================================================
#     # PREDICTION
#     # ========================================================

#     if analyze_button:

#         with st.spinner(
#             "🤖 WasteWise AI is analyzing the image..."
#         ):

#             try:

#                 # Resize image
#                 processed_image = image.resize(
#                     IMAGE_SIZE
#                 )

#                 # Convert to NumPy
#                 image_array = np.array(
#                     processed_image
#                 ).astype("float32")

#                 # Add batch dimension
#                 image_array = np.expand_dims(
#                     image_array,
#                     axis=0
#                 )

#                 # Prediction
#                 predictions = model.predict(
#                     image_array,
#                     verbose=0
#                 )[0]

#                 # Highest probability
#                 predicted_index = int(
#                     np.argmax(predictions)
#                 )

#                 predicted_class = (
#                     CLASS_NAMES[predicted_index]
#                 )

#                 confidence = float(
#                     predictions[predicted_index]
#                 )


#             except Exception as error:

#                 st.error(
#                     f"Prediction failed: {error}"
#                 )

#                 st.stop()


#         # ====================================================
#         # RESULT INFORMATION
#         # ====================================================

#         info = WASTE_INFO.get(
#             predicted_class,
#             {
#                 "icon": "♻️",
#                 "category": predicted_class,
#                 "message":
#                     "No additional information available.",
#                 "tips": []
#             }
#         )


#         icon = info["icon"]

#         category = info["category"]


#         # ====================================================
#         # AI PREDICTION
#         # ====================================================

#         st.markdown(
#             '<div class="section-title">'
#             '🤖 AI Prediction'
#             '</div>',
#             unsafe_allow_html=True
#         )


#         st.markdown(
#             f"""
#             <div class="result-box">

#                 <div class="result-title">
#                     {icon} {predicted_class}
#                 </div>

#                 <div class="confidence">
#                     Confidence:
#                     {confidence * 100:.2f}%
#                 </div>

#             </div>
#             """,
#             unsafe_allow_html=True
#         )


#         # ====================================================
#         # CONFIDENCE STATUS
#         # ====================================================

#         if confidence >= 0.80:

#             st.success(
#                 "🟢 High-confidence prediction"
#             )

#         elif confidence >= CONFIDENCE_THRESHOLD:

#             st.warning(
#                 "🟡 Moderate-confidence prediction"
#             )

#         else:

#             st.error(
#                 "🔴 Low-confidence prediction"
#             )

#             st.warning(
#                 "The AI is not sufficiently confident. "
#                 "Try uploading a clearer image with the "
#                 "waste item more visible."
#             )


#         # ====================================================
#         # WASTE CATEGORY
#         # ====================================================

#         st.markdown(
#             "### 🏷️ Waste Category"
#         )

#         st.write(
#             f"**{category}**"
#         )


#         # ====================================================
#         # GUIDANCE
#         # ====================================================

#         st.markdown(
#             "### 💡 Waste Management Guidance"
#         )

#         st.info(
#             info["message"]
#         )


#         # ====================================================
#         # RECOMMENDED PRACTICES
#         # ====================================================

#         if info["tips"]:

#             st.markdown(
#                 "### ✅ Recommended Practices"
#             )

#             for tip in info["tips"]:

#                 st.write(
#                     f"• {tip}"
#                 )


#         # ====================================================
#         # TOP PREDICTIONS
#         # ====================================================

#         st.markdown(
#             "### 📊 Top Predictions"
#         )


#         sorted_indices = np.argsort(
#             predictions
#         )[::-1]


#         for rank, index in enumerate(
#             sorted_indices[:3],
#             start=1
#         ):

#             class_name = CLASS_NAMES[index]

#             probability = float(
#                 predictions[index]
#             )


#             st.write(
#                 f"**{rank}. {class_name}**"
#             )


#             st.progress(
#                 min(
#                     probability,
#                     1.0
#                 )
#             )


#             st.caption(
#                 f"{probability * 100:.2f}%"
#             )


#         # ====================================================
#         # IMPORTANT NOTICE
#         # ====================================================

#         st.markdown("---")

#         st.warning(
#             """
#             ⚠️ **Important:** WasteWise AI is an educational
#             machine-learning project. Its prediction should
#             not be treated as a definitive determination of
#             material composition, recyclability, hazardousness,
#             or microbiological safety. Always follow local
#             waste-management rules and use appropriate
#             professional guidance for hazardous materials.
#             """
#         )


# # ============================================================
# # EMPTY STATE
# # ============================================================

# else:

#     st.markdown(
#         "### 📸 How to Use"
#     )

#     st.write(
#         """
#         1. Upload a clear waste image, or use your camera.
#         2. Click **Analyze Waste**.
#         3. WasteWise AI predicts the waste category.
#         4. Review the confidence score.
#         5. Follow the displayed waste-management guidance.
#         """
#     )


# # ============================================================
# # FOOTER
# # ============================================================

# st.markdown(
#     """
#     <div class="footer">
#         ♻️ <b>WasteWise AI</b><br>
#         AI for Smarter Waste Classification & Awareness
#     </div>
#     """,
#     unsafe_allow_html=True
# )


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
    page_title="WasteWise AI",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
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
# MODEL SETTINGS
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
# SIMPLE CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Top-right logos */

    .logo-container {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 12px;
        margin-bottom: 5px;
    }

    .logo-circle {
        width: 65px;
        height: 65px;
        border-radius: 100%;
        object-fit: cover;
        border: 2px solid rgba(128, 128, 128, 0.25);
    }


    /* Main title */

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }


    /* Subtitle */

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 20px;
        opacity: 0.85;
    }


    /* Prediction result */

    .result-box {
        padding: 25px;
        border-radius: 18px;
        border: 1px solid rgba(128, 128, 128, 0.25);
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


    /* Section title */

    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 25px;
    }


    /* Footer */

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
# LOGO DISPLAY
# ============================================================
#
# NOTE:
# Streamlit's st.image() is used instead of Base64 HTML.
# Therefore <img src="data:image/png;base64,..."> will
# NOT appear as text in the application.
#
# ============================================================

logo_col1, logo_col2, logo_col3 = st.columns(
    [7, 1, 1]
)


with logo_col2:

    if ICODEGURU_LOGO.exists():

        st.image(
            str(ICODEGURU_LOGO),
            width=65;
            height=100
        )

    else:

        st.caption(
            "iCodeGuru logo missing"
        )


with logo_col3:

    if SCHOOL_LOGO.exists():

        st.image(
            str(SCHOOL_LOGO),
            width=65;
            height=100
        )

    else:

        st.caption(
            "School logo missing"
        )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">♻️ WasteWise AI</div>',
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
    "Upload a waste image or use your camera. "
    "WasteWise AI will classify it into one of four "
    "trained waste categories."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("♻️ WasteWise AI")

    st.markdown("### About the Project")

    st.write(
        """
        WasteWise AI uses a trained EfficientNetB0
        image-classification model to identify waste
        categories from images.
        """
    )

    st.markdown("### Supported Classes")

    for class_name in CLASS_NAMES if 'CLASS_NAMES' in locals() else [
        "Hazardous",
        "Non-Recyclable",
        "Organic",
        "Recyclable"
    ]:

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
            "r"
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
# SELECT IMAGE
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
        "### 🖼️ Selected Waste Image"
    )

    st.image(
        image,
        caption="Waste Image",
        width="stretch"
    )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze_button = st.button(
        "🔍 Analyze Waste",
        type="primary",
        width="stretch"
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if analyze_button:

        with st.spinner(
            "🤖 WasteWise AI is analyzing the image..."
        ):

            try:

                # Resize image
                processed_image = image.resize(
                    IMAGE_SIZE
                )

                # Convert image to NumPy
                image_array = np.array(
                    processed_image
                ).astype("float32")

                # Add batch dimension
                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                # Model prediction
                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

                # Find highest prediction
                predicted_index = int(
                    np.argmax(predictions)
                )

                predicted_class = (
                    CLASS_NAMES[predicted_index]
                )

                confidence = float(
                    predictions[predicted_index]
                )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )

                st.stop()


        # ====================================================
        # GET WASTE INFORMATION
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


        icon = info["icon"]

        category = info["category"]


        # ====================================================
        # AI PREDICTION
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🤖 AI Prediction'
            '</div>',
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
                "Try uploading a clearer image with the "
                "waste item more visible."
            )


        # ====================================================
        # WASTE CATEGORY
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
        # RECOMMENDED PRACTICES
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
        # TOP PREDICTIONS
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

            class_name = CLASS_NAMES[index]

            probability = float(
                predictions[index]
            )


            st.write(
                f"**{rank}. {class_name}**"
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
        # IMPORTANT NOTICE
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
        1. Upload a clear waste image, or use your camera.
        2. Click **Analyze Waste**.
        3. WasteWise AI predicts the waste category.
        4. Review the confidence score.
        5. Follow the displayed waste-management guidance.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ♻️ <b>WasteWise AI</b><br>
        AI for Smarter Waste Classification & Awareness
    </div>
    """,
    unsafe_allow_html=True
)
