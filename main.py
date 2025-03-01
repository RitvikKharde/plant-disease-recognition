import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image
import os

# Function to download and load the model
def download_model():
    model_path = "trained_model.keras"
    if not os.path.exists(model_path):  # Check if model exists
        url = "https://drive.google.com/file/d/1Z87vAZK_77oHON3Yn1RoP0YS5-zaPoiw/view?usp=drive_link"  # Google Drive Model Link
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

# Cache model correctly
@st.cache_resource
def load_model():
    return download_model()

# Load model
model = load_model()

# Image Preprocessing & Prediction Function
def model_prediction(test_image):
    try:
        # Ensure image is in RGB format
        image = Image.open(test_image).convert("RGB")
        image = image.resize((128, 128))  # Resize to match model input size
        input_arr = np.array(image) / 255.0  # Normalize between 0-1
        input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch

        # Run prediction
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)  # Get highest probability class index
        confidence_score = float(np.max(prediction))  # Get confidence score

        return result_index, confidence_score

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
        return None, None

# Sidebar
st.sidebar.title("🌿 Dashboard")
app_mode = st.sidebar.selectbox("📌 Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("🌱 PLANT DISEASE RECOGNITION SYSTEM")
    
    # Ensure the image path exists
    image_path = "/Users/ritvikkharde/Downloads/Plant_disease.jpg"
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True)
    else:
        st.warning("⚠️ Home page image not found. Please check the file path.")

    st.markdown("""
    Welcome to the **Plant Disease Recognition System!** 🌿🔍

    Our **AI-powered** model helps identify plant diseases **quickly and accurately**.

    ### **How It Works**
    1️⃣ **Upload an Image** → Go to the **Disease Recognition** page.  
    2️⃣ **AI Analysis** → The system processes the image.  
    3️⃣ **Get Results** → The predicted disease and confidence score will be displayed.

    🔹 **Accurate** 🔹 **User-Friendly** 🔹 **Fast & Efficient**
    
    👉 Click **Disease Recognition** in the sidebar to start!
""")

# About Page
elif app_mode == "About":
    st.header("📚 About the Project")
    st.markdown("""
    **Dataset Information:**  
    - The dataset consists of **87,000+ images** of **healthy & diseased** crop leaves.  
    - It is divided into **38 different plant disease classes**.  
    - **80% training data** and **20% validation data** for improved accuracy.

    **Contents:**  
    🔹 **Train**: 70,295 images  
    🔹 **Validation**: 17,572 images  
    🔹 **Test**: 33 images  
""")

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("📸 Disease Recognition")

    test_image = st.file_uploader("📂 Upload a plant image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        st.image(test_image, use_column_width=True, caption="Uploaded Image")

        if st.button("🔍 Predict"):
            with st.spinner("🧠 AI Analyzing..."):
                result_index, confidence_score = model_prediction(test_image)

                if result_index is not None:
                    # Define Class Labels
                    class_names = [
                        "Apple___Apple_scab",
                        "Apple___Cedar_apple_rust",
                        "Apple___healthy",
                        "Corn_(maize)___Northern_Leaf_Blight",
                        "Corn_(maize)___healthy",
                        "Grape___Esca_(Black_Measles)",
                        "Strawberry___Leaf_scorch",
                        "Tomato___Early_blight"
                    ]

                    if result_index < len(class_names):  # Ensure index is valid
                        predicted_label = class_names[result_index]
                        st.success(f"🌱 **Prediction: {predicted_label}**")
                        st.write(f"🔍 **Confidence Score:** {confidence_score:.2f}")
                    else:
                        st.error("⚠️ Error: Model returned an invalid class index.")
