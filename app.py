
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
# 2. FOREST BACKGROUND FUNCTION
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
                rgba(0, 0, 0, 0.35),
                rgba(0, 0, 0, 0.50)
            ),
            url("data:image/jpeg;base64,{encoded}");
        """

    else:

        background_css = """
        background: linear-gradient(
            135deg,
            #18392b,
            #263d32,
            #172b22
        );
        """

        st.warning(
            "Forest background image not found. "
            "Please add forest_background.jpg to the project folder."
        )

    st.markdown(
        f"""
        <style>

        /* Natural forest background */

        [data-testid="stAppViewContainer"] {{
            {background_css}
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Transparent header */

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        /* Main content */

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        /* Main title */

        .main-title {{
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: white;
            padding: 22px;
            background: rgba(0, 0, 0, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.35);
            border-radius: 15px;
            margin-bottom: 12px;
            text-shadow: 2px 2px 5px black;
        }}

        .main-title span {{
            color: #86efac;
        }}

        /* Subtitle */

        .sub-title {{
            text-align: center;
            font-size: 17px;
            color: white;
            margin-bottom: 30px;
            text-shadow: 1px 1px 4px black;
        }}

        /* Headings and labels */

        h1, h2, h3, p, label {{
            color: white !important;
        }}

        /* Upload section */

        [data-testid="stFileUploader"] {{
            background: rgba(0, 0, 0, 0.48);
            border: 1px solid rgba(255, 255, 255, 0.4);
            border-radius: 14px;
            padding: 16px;
        }}

        [data-testid="stFileUploader"] section {{
            background: rgba(255, 255, 255, 0.12);
            border: 1px dashed rgba(255, 255, 255, 0.7);
            border-radius: 10px;
        }}

        [data-testid="stFileUploader"] button {{
            background: #277a48 !important;
            color: white !important;
            border: 1px solid #86efac !important;
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
                #237a43,
                #39a85e
            );
            color: white;
            font-size: 20px;
            font-weight: bold;
            border-radius: 12px;
            border: 1px solid #a7f3c0;
            transition: 0.3s;
        }}

        .stButton > button:hover {{
            background: #166534;
            color: white;
            border: 1px solid white;
        }}

        /* Uploaded image */

        [data-testid="stImage"] img {{
            border: 2px solid rgba(255, 255, 255, 0.7);
            border-radius: 14px;
        }}

        /* Prediction result */

        .result-box {{
            padding: 22px;
            border-radius: 14px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: white;
            background: rgba(0, 0, 0, 0.70);
            border: 2px solid #86efac;
            margin-top: 20px;
            text-shadow: 1px 1px 3px black;
        }}

        /* Footer */

        .footer {{
            text-align: center;
            color: white;
            font-size: 15px;
            padding: 20px;
            background: rgba(0, 0, 0, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            text-shadow: 1px 1px 3px black;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# Apply background

set_background("forest_background.jpg")


# ==========================================
# 3. LOAD TRAINED CNN MODEL
# ==========================================

@st.cache_resource
def load_forest_model():

    model = tf.keras.models.load_model(
        "forest_fire_model.keras",
        compile=False
    )

    return model


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

                        resized_image = image.resize(
                            (160, 160)
                        )

                        # Convert image into NumPy array

                        image_array = np.array(
                            resized_image,
                            dtype=np.float32
                        )

                        # Add batch dimension

                        image_array = np.expand_dims(
                            image_array,
                            axis=0
                        )

                        # Preprocessing is already included
                        # inside the saved model.

                        prediction = model.predict(
                            image_array,
                            verbose=0
                        )

                        probability = float(
                            np.asarray(prediction).reshape(-1)[0]
                        )

                        # Training label mapping:
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

                    # Display confidence

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
                        "Model is not loaded. Please check the model file."
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




