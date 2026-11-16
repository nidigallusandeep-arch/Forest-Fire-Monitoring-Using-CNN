
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import os

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Forest Fire Detection Using CNN",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. FOREST BACKGROUND
# ==========================================

def set_background(image_path):

    if os.path.exists(image_path):

        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(
                image_file.read()
            ).decode()

        background_css = f"""
        background-image:
            linear-gradient(
                rgba(0, 25, 12, 0.65),
                rgba(0, 25, 12, 0.78)
            ),
            url("data:image/jpeg;base64,{encoded}");
        """

    else:
        background_css = """
        background: linear-gradient(
            135deg,
            #052e16,
            #064e3b,
            #022c22
        );
        """

    st.markdown(
        f"""
        <style>

        /* Main background */

        [data-testid="stAppViewContainer"] {{
            {background_css}
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        /* Main title */

        .main-title {{
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #ffffff;
            padding: 22px;
            background: rgba(2, 44, 34, 0.88);
            border: 1px solid #22c55e;
            border-radius: 15px;
            margin-bottom: 12px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        }}

        .main-title span {{
            color: #4ade80;
        }}

        /* Subtitle */

        .sub-title {{
            text-align: center;
            font-size: 17px;
            color: #dcfce7;
            margin-bottom: 30px;
        }}

        /* Headings and labels */

        h1, h2, h3, p, label {{
            color: #ffffff !important;
        }}

        /* Upload section */

        [data-testid="stFileUploader"] {{
            background: rgba(2, 44, 34, 0.85);
            border: 2px solid #16a34a;
            border-radius: 14px;
            padding: 18px;
        }}

        [data-testid="stFileUploader"] section {{
            background: rgba(6, 78, 59, 0.85) !important;
            border: 1px dashed #4ade80 !important;
            border-radius: 10px !important;
        }}

        [data-testid="stFileUploader"] button {{
            background: #15803d !important;
            color: white !important;
            border: 1px solid #4ade80 !important;
        }}

        [data-testid="stFileUploader"] small {{
            color: #dcfce7 !important;
        }}

        [data-testid="stFileUploader"] span {{
            color: white !important;
        }}

        /* Predict button */

        .stButton > button {{
            width: 100%;
            height: 55px;
            background: linear-gradient(
                90deg,
                #15803d,
                #22c55e
            );
            color: white;
            font-size: 20px;
            font-weight: bold;
            border-radius: 12px;
            border: 1px solid #86efac;
            transition: 0.3s;
        }}

        .stButton > button:hover {{
            background: #16a34a;
            color: white;
            border: 2px solid white;
        }}

        /* Image border */

        [data-testid="stImage"] img {{
            border: 3px solid #22c55e;
            border-radius: 14px;
        }}

        /* Result box */

        .result-box {{
            padding: 22px;
            border-radius: 14px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: white;
            background: rgba(2, 44, 34, 0.95);
            border: 2px solid #4ade80;
            margin-top: 20px;
        }}

        /* Footer */

        .footer {{
            text-align: center;
            color: #dcfce7;
            font-size: 15px;
            padding: 20px;
            background: rgba(2, 44, 34, 0.88);
            border: 1px solid #16a34a;
            border-radius: 12px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


set_background("forest_background.jpg")


# ==========================================
# 3. LOAD CNN MODEL
# ==========================================

@st.cache_resource
def load_forest_model():

    return tf.keras.models.load_model(
        "forest_fire_model.keras",
        compile=False
    )


try:
    model = load_forest_model()
    model_status = True

except Exception as e:
    model_status = False
    st.error(f"Model loading failed: {e}")


# ==========================================
# 4. PROJECT TITLE
# ==========================================

st.markdown(
    """
    <div class="main-title">
        🔥 Forest Fire Detection Using
        <span>CNN</span>
    </div>

    <div class="sub-title">
        Upload a forest image to detect whether it contains fire or no fire.
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# 5. IMAGE UPLOAD
# ==========================================

st.markdown("### 📤 Upload Forest Image")

uploaded_file = st.file_uploader(
    "Choose a forest image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# 6. IMAGE PREVIEW AND PREDICTION
# ==========================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(
            [1, 1],
            gap="large"
        )

        with col1:

            st.image(
                image,
                caption="Uploaded Forest Image",
                width="stretch"
            )

        with col2:

            st.markdown("### 🔍 Prediction")

            if st.button("🔥 Predict"):

                if model_status:

                    with st.spinner("Analyzing forest image..."):

                        # Resize image

                        resized_image = image.resize((160, 160))

                        # Convert image to array

                        image_array = np.array(
                            resized_image,
                            dtype=np.float32
                        )

                        # Add batch dimension

                        image_array = np.expand_dims(
                            image_array,
                            axis=0
                        )

                        # MobileNetV2 preprocessing is already
                        # included inside the saved model.

                        prediction = model.predict(
                            image_array,
                            verbose=0
                        )

                        probability = float(
                            np.asarray(prediction).reshape(-1)[0]
                        )

                        # Training mapping:
                        # FIRE = 0
                        # NON_FIRE = 1

                        if probability >= 0.5:

                            result = "NON_FIRE"
                            confidence = probability * 100
                            icon = "🌳"

                        else:

                            result = "FIRE"
                            confidence = (1 - probability) * 100
                            icon = "🔥"

                    # Display prediction

                    if result == "FIRE":

                        st.error("🔥 FIRE DETECTED")

                    else:

                        st.success("🌳 NO FIRE DETECTED")

                    st.markdown(
                        f"""
                        <div class="result-box">
                            Prediction: {icon} {result}
                            <br><br>
                            Confidence: {confidence:.2f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(
                        min(max(confidence / 100, 0.0), 1.0)
                    )

                else:

                    st.error(
                        "Model is not loaded. Check the model file."
                    )

    except Exception as e:

        st.error(f"Image processing error: {e}")


# ==========================================
# 7. FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        🔥 Smart Forest Fire Monitoring System
        <br><br>
        Developed using Python | CNN | TensorFlow | Streamlit
    </div>
    """,
    unsafe_allow_html=True
)



