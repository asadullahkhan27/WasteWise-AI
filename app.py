```python
# ============================================================
# WASTEWISE AI
# AI-Powered Waste Classification & Recycling Assistant
# Science Exhibition Edition
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
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "wastewise_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"


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
        "short": "Handle Carefully",
        "category": "Special / Hazardous Disposal",
        "description": (
            "This waste may require special handling and "
            "should not be mixed with ordinary recycling."
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
        "short": "Non-Recyclable",
        "category": "General Waste",
        "description": (
            "The AI classified this item as non-recyclable. "
            "Disposal should follow local waste-management rules."
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
        "short": "Organic",
        "category": "Compostable / Organic",
        "description": (
            "Organic waste can potentially be composted "
            "when suitable composting facilities are available."
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
        "short": "Recyclable",
        "category": "Recyclable Material",
        "description": (
            "The AI classified this item as recyclable. "
            "Actual recyclability depends on local recycling rules."
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
# CUSTOM CSS — MODERN EXHIBITION UI
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       GLOBAL
    ------------------------------------------------------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(46, 204, 113, 0.10),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(52, 152, 219, 0.08),
                transparent 30%
            ),
            linear-gradient(
                180deg,
                #f8fffb 0%,
                #f4f8f7 100%
            );
    }


    /* -------------------------------------------------------
       MAIN CONTAINER
    ------------------------------------------------------- */

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* -------------------------------------------------------
       HERO
    ------------------------------------------------------- */

    .hero {
        text-align: center;
        padding: 45px 20px 30px 20px;
    }

    .hero-badge {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 50px;
        background: rgba(39, 174, 96, 0.12);
        border: 1px solid rgba(39, 174, 96, 0.25);
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: clamp(42px, 7vw, 78px);
        line-height: 1;
        font-weight: 900;
        letter-spacing: -3px;
        margin: 0;
    }

    .hero-title span {
        background: linear-gradient(
            90deg,
            #159957,
            #30b56d,
            #0d8b55
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        max-width: 760px;
        margin: 20px auto 0 auto;
        font-size: 20px;
        line-height: 1.6;
        opacity: 0.75;
    }


    /* -------------------------------------------------------
       STATS
    ------------------------------------------------------- */

    .stats-grid {
        display: grid;
        grid-template-columns:
            repeat(4, 1fr);
        gap: 15px;
        margin: 25px 0 35px 0;
    }

    .stat-card {
        padding: 22px 15px;
        border-radius: 20px;
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(0,0,0,0.07);
        text-align: center;
        box-shadow:
            0 8px 30px rgba(0,0,0,0.05);
    }

    .stat-number {
        font-size: 30px;
        font-weight: 900;
    }

    .stat-label {
        font-size: 13px;
        opacity: 0.65;
        margin-top: 4px;
    }


    /* -------------------------------------------------------
       SECTION HEADERS
    ------------------------------------------------------- */

    .section-heading {
        font-size: 28px;
        font-weight: 850;
        margin-top: 35px;
        margin-bottom: 8px;
    }

    .section-description {
        opacity: 0.68;
        margin-bottom: 20px;
    }


    /* -------------------------------------------------------
       UPLOAD CARD
    ------------------------------------------------------- */

    .upload-card {
        padding: 28px;
        border-radius: 26px;
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(0,0,0,0.07);
        box-shadow:
            0 15px 50px rgba(0,0,0,0.07);
    }


    /* -------------------------------------------------------
       RESULT CARD
    ------------------------------------------------------- */

    .result-card {
        padding: 32px;
        border-radius: 28px;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(39,174,96,0.18);
        box-shadow:
            0 18px 55px rgba(0,0,0,0.08);
        text-align: center;
        margin-top: 25px;
    }

    .result-icon {
        font-size: 65px;
        margin-bottom: 5px;
    }

    .result-label {
        font-size: 14px;
        font-weight: 700;
        opacity: 0.55;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .result-name {
        font-size: clamp(30px, 5vw, 48px);
        font-weight: 900;
        margin: 8px 0;
    }

    .result-category {
        font-size: 17px;
        opacity: 0.7;
    }


    /* -------------------------------------------------------
       CONFIDENCE
    ------------------------------------------------------- */

    .confidence-card {
        padding: 24px;
        border-radius: 22px;
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(0,0,0,0.07);
        margin-top: 18px;
    }

    .confidence-number {
        font-size: 38px;
        font-weight: 900;
        text-align: center;
    }

    .confidence-label {
        text-align: center;
        font-size: 13px;
        opacity: 0.6;
    }


    /* -------------------------------------------------------
       INFO CARDS
    ------------------------------------------------------- */

    .info-card {
        padding: 24px;
        border-radius: 22px;
        background: rgba(255,255,255,0.86);
        border: 1px solid rgba(0,0,0,0.06);
        height: 100%;
        box-shadow:
            0 8px 30px rgba(0,0,0,0.04);
    }

    .info-icon {
        font-size: 38px;
    }

    .info-title {
        font-size: 20px;
        font-weight: 800;
        margin-top: 10px;
    }

    .info-text {
        opacity: 0.68;
        line-height: 1.6;
        margin-top: 8px;
    }


    /* -------------------------------------------------------
       CLASS CHIPS
    ------------------------------------------------------- */

    .chips {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 15px;
    }

    .chip {
        padding: 9px 15px;
        border-radius: 50px;
        background: rgba(39,174,96,0.09);
        border: 1px solid rgba(39,174,96,0.18);
        font-size: 13px;
        font-weight: 700;
    }


    /* -------------------------------------------------------
       FOOTER
    ------------------------------------------------------- */

    .footer-box {
        text-align: center;
        padding: 35px 20px 10px 20px;
        margin-top: 45px;
        border-top: 1px solid rgba(0,0,0,0.08);
        opacity: 0.7;
    }


    /* -------------------------------------------------------
       STREAMLIT INPUTS
    ------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        padding: 10px;
    }

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 20px !important;
        border: 2px dashed rgba(39,174,96,0.35) !important;
        padding: 28px !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: 15px;
        min-height: 52px;
        font-size: 17px;
        font-weight: 800;
    }


    /* -------------------------------------------------------
       MOBILE RESPONSIVE
    ------------------------------------------------------- */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }

        .hero {
            padding-top: 25px;
        }

        .hero-title {
            letter-spacing: -1.5px;
        }

        .hero-subtitle {
            font-size: 16px;
        }

        .stats-grid {
            grid-template-columns:
                repeat(2, 1fr);
        }

        .stat-number {
            font-size: 24px;
        }

        .upload-card {
            padding: 18px;
        }

        .result-card {
            padding: 24px 15px;
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
            f"Model loading error: {error}"
        )

        return None


model = load_model()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            🌍 SCIENCE EXHIBITION • AI PROJECT
        </div>

        <h1 class="hero-title">
            ♻️ <span>WasteWise AI</span>
        </h1>

        <div class="hero-subtitle">
            An AI-powered waste classification assistant
            designed to promote smarter waste awareness,
            responsible disposal, and recycling.
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
            <div class="stat-number">🤖 AI</div>
            <div class="stat-label">
                Image Classification
            </div>
        </div>

        <div class="stat-card">
            <div class="stat-number">4</div>
            <div class="stat-label">
                Waste Categories
            </div>
        </div>

        <div class="stat-card">
            <div class="stat-number">224²</div>
            <div class="stat-label">
                Image Input
            </div>
        </div>

        <div class="stat-card">
            <div class="stat-number">♻️</div>
            <div class="stat-label">
                Sustainability Focus
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL ERROR CHECK
# ============================================================

if model is None:

    st.error(
        "⚠️ WasteWise AI model is not available."
    )

    st.markdown(
        """
        Please check that your project contains:

        `model/wastewise_model.keras`

        and

        `model/class_names.json`
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
        Upload a clear image of a waste item and let
        WasteWise AI analyze it.
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="upload-card">',
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
    help="For better results, use a clear image with the waste item visible."
)


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# IMAGE + PREDICTION
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
        '<div class="section-heading">🖼️ Image Preview</div>',
        unsafe_allow_html=True
    )


    preview_col, details_col = st.columns(
        [1.2, 1],
        gap="large"
    )


    with preview_col:

        st.image(
            image,
            caption="Uploaded Waste Image",
            width="stretch"
        )


    with details_col:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-icon">
                    📷
                </div>

                <div class="info-title">
                    Ready for Analysis
                </div>

                <div class="info-text">
                    Your image has been loaded successfully.
                    Click the button below to send it to the
                    WasteWise AI classification model.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        analyze_button = st.button(
            "🔍 Analyze Waste",
            type="primary"
        )


    # ========================================================
    # PREDICTION
    # ========================================================

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
            predictions[predicted_index]
        )

        info = WASTE_INFO.get(
            predicted_class,
            {
                "icon": "♻️",
                "short": predicted_class,
                "category": predicted_class,
                "description":
                    "No additional information available.",
                "tips": []
            }
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            '<div class="section-heading">🤖 AI Result</div>',
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
                "### Confidence Level"
            )

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
                    "centered and well illuminated."
                )


        # ====================================================
        # GUIDANCE
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
                        What does this mean?
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
                    f"<div style='margin:8px 0;'>"
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
        # TOP PREDICTIONS
        # ====================================================

        st.markdown(
            '<div class="section-heading">📊 AI Confidence Breakdown</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="section-description">
                See how the model scored each supported waste category.
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

            class_name = CLASS_NAMES[
                index
            ]

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


            st.write(
                f"**{rank}. {icon} {class_name}**"
            )

            st.progress(
                min(probability, 1.0)
            )

            st.caption(
                f"{probability * 100:.2f}% confidence"
            )


        # ====================================================
        # EDUCATIONAL NOTE
        # ====================================================

        st.markdown(
            "---"
        )

        st.warning(
            """
            ⚠️ **Educational AI Notice**

            WasteWise AI is a science-exhibition machine-learning
            project. Predictions are based on the visual patterns
            learned from its training dataset and should not be
            treated as a definitive determination of material
            composition, recyclability, hazardousness, or safety.

            Always follow local waste-management rules and seek
            appropriate professional guidance for hazardous materials.
            """
        )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    '<div class="section-heading">🧠 How WasteWise AI Works</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-description">
        From image to intelligent waste classification.
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
        "01",
        "📸",
        "Capture",
        "Upload a clear image of the waste item."
    ),

    (
        "02",
        "🧠",
        "Analyze",
        "EfficientNetB0 extracts visual features."
    ),

    (
        "03",
        "🤖",
        "Classify",
        "The trained AI predicts the waste category."
    ),

    (
        "04",
        "🌱",
        "Act",
        "Get awareness and waste-management guidance."
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
            <div class="info-card">

                <div style="
                    font-size:13px;
                    font-weight:800;
                    opacity:0.45;
                ">
                    STEP {number}
                </div>

                <div class="info-icon">
                    {icon}
                </div>

                <div class="info-title">
                    {title}
                </div>

                <div class="info-text">
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

with st.expander(
    "🔬 View AI Model Information"
):

    model_col1, model_col2 = st.columns(
        2
    )

    with model_col1:

        st.write(
            "**Architecture:** EfficientNetB0"
        )

        st.write(
            "**Input Size:** 224 × 224 × 3"
        )

        st.write(
            "**Output Classes:** 4"
        )


    with model_col2:

        st.write(
            "**Training:** Transfer Learning"
        )

        st.write(
            "**Classification:** Image Classification"
        )

        st.write(
            "**Model File:** wastewise_model.keras"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-box">

        <div style="font-size:28px;">
            ♻️
        </div>

        <div style="
            font-size:18px;
            font-weight:800;
            margin-top:8px;
        ">
            WasteWise AI
        </div>

        <div style="
            margin-top:5px;
            font-size:13px;
        ">
            AI for Smarter Waste Classification & Awareness
        </div>

        <div style="
            margin-top:12px;
            font-size:12px;
        ">
            Built for Science Exhibition • Machine Learning • Sustainability
        </div>

    </div>
    """,
    unsafe_allow_html=True
)
```

### `requirements.txt`

Is UI ke liye yeh rakho:

```txt
streamlit
tensorflow
numpy
pillow
```

### 📁 Final folder

```text
WasteWise-AI/
│
├── app.py
├── requirements.txt
│
└── model/
    ├── wastewise_model.keras
    └── class_names.json
```

### 🔥 UI mein ab kya hai?

* **Exhibition-style Hero Section**
* **Responsive desktop + mobile layout**
* **Modern upload/drop-zone**
* **Large AI prediction card**
* **Confidence percentage + progress bar**
* **Top 3/4 prediction breakdown**
* **Waste-management guidance**
* **Recommended practices**
* **4 supported-category chips**
* **“How WasteWise AI Works” 4-step section**
* **AI model information**
* **Educational safety notice**
* **Responsive cards and typography**
* **Clean sustainability-focused visual design**

Run karne ke liye:

```bash
streamlit run app.py
```

**Ek important point:** tumhari current model accuracy **74.14%** hai, isliye UI ko powerful banana possible hai, lekin UI accuracy ko artificially improve nahi karta. Ab app visually exhibition-ready ho jayegi; actual AI quality tumhare trained model ki performance par depend karegi.
