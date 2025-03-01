import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image
import os

# Function to download model from Google Drive
def download_model():
    model_path = "trained_model2.keras"
    if not os.path.exists(model_path):  # Check if model already exists
        url = "https://drive.google.com/uc?id=13LrTSaRiWgFKJ2ZGV-Bbzz8tMeaDir6q"  # Corrected Google Drive Link
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

# Load Model Once
@st.cache_resource
def load_model():
    return download_model()

# Load model
model = load_model()


# Image Preprocessing & Prediction Function
def model_prediction(test_image):
    image = Image.open(test_image).convert("RGB")  # Ensure RGB format
    image = image.resize((128, 128))  # Resize to match model input size
    input_arr = np.array(image) / 255.0  # Normalize between 0-1
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch

    prediction = model.predict(input_arr)  # Get prediction
    result_index = np.argmax(prediction)  # Get highest probability class index
    confidence_score = np.max(prediction)  # Get confidence score
    return result_index, confidence_score


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("🌱 PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "/Users/ritvikkharde/Downloads/Plant_disease.jpg"
    
    # Check if image exists before displaying
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True)
    else:
        st.warning("⚠️ Warning: Home page image not found. Please check the path.")

    st.markdown("""
    Welcome to the **Plant Disease Recognition System!** 🌿🔍

    Our AI-powered system helps identify plant diseases quickly. Just upload an image of a plant, and our system will analyze it.

    ### **How It Works**
    1️⃣ **Upload an Image** → Go to the **Disease Recognition** page and upload a plant image.  
    2️⃣ **AI Analysis** → The model processes the image using deep learning.  
    3️⃣ **Get Results** → The system will display the predicted disease with confidence.

    🔹 **Accurate** 🔹 **User-Friendly** 🔹 **Fast & Efficient**
    
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of AI!
""")

# About Page
elif app_mode == "About":
    st.header("📚 About the Project")
    st.markdown("""
    **Dataset Information:**  
    - This dataset consists of **87,000+ images** of **healthy & diseased** crop leaves.  
    - It is divided into **38 different classes** for plant disease detection.  
    - The dataset is split **80% for training** and **20% for validation**.

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
                
                predicted_label = class_names[result_index]
                
                st.success(f"🌱 **Prediction: {predicted_label}**")
                st.write(f"🔍 **Confidence Score:** {confidence_score:.2f}")
