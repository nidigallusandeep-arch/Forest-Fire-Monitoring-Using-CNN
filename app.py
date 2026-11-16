import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Forest Fire Detection",
    page_icon="🔥",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #14231b;
    color: white;
}
h1 {
    color: #ff7043;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "forest_fire_model.keras"
    )

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Heading
st.title("🔥 Forest Fire Detection Using MobileNetV2")

st.write(
    "Upload a forest image to detect whether it contains fire or no fire."
)

# Image upload
uploaded_file = st.file_uploader(
    "Upload Forest Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Forest Image",
            use_container_width=True
        )

    with col2:

        if st.button("Predict", use_container_width=True):

            # Resize image
            resized_image = image.resize((160, 160))

            # Convert to array
            img_array = np.array(
                resized_image,
                dtype=np.float32
            )

            # Add batch dimension
            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # Model prediction
            prediction = model.predict(
                img_array,
                verbose=0
            )[0][0]

            # FIRE = 0, NON_FIRE = 1
            if prediction >= 0.5:

                result = "NON_FIRE"
                confidence = float(prediction) * 100

                st.success("🌳 No Fire Detected")

            else:

                result = "FIRE"
                confidence = (1 - float(prediction)) * 100

                st.error("🔥 Fire Detected")

            st.subheader("Prediction Result")

            st.write("Predicted Class:", result)

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                float(confidence / 100)
            )

st.markdown("---")

st.caption(
    "Developed using Python, TensorFlow, Keras and MobileNetV2"
)

st.warning(
    "This is an image classification demonstration, "
    "not a certified fire alarm or emergency detection system."
)
