import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image
import os

# Function to download model from Google Drive
def download_model():
    model_path = "trained_model.keras"
    if not os.path.exists(model_path):  # Check if model already exists
        url = "https://drive.google.com/file/d/13LrTSaRiWgFKJ2ZGV-Bbzz8tMeaDir6q/view?usp=drive_link"  # Replace with your file ID
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)
# Load Model Once
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('trained_model2.keras')  # Ensure correct file name

model = load_model()

# Image Preprocessing & Prediction Function
def model_prediction(test_image):
    image = Image.open(test_image)  # Open the uploaded image
    image = image.resize((128, 128))  # Resize to model input size
    input_arr = np.array(image) / 255.0  # Normalize
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert to batch format

    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)  # Get highest probability class index
    return result_index

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "/Users/ritvikkharde/Downloads/Plant_disease.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
""")

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio for training and validation, preserving the directory structure. A new directory containing 33 test images was created later for prediction purposes.
    #### Content
    1. Train (70,295 images)
    2. Valid (17,572 images)
    3. Test (33 images)
""")

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        st.image(test_image, use_column_width=True)
    
        if st.button("Predict"):
            with st.spinner("Please Wait..."):
                result_index = model_prediction(test_image)
                
                # Define Class Labels
                class_names=[
                    'Apple___Apple_scab',
                    'Apple___Cedar_apple_rust',
                    'Apple___healthy',
                    'Corn_(maize)___Northern_Leaf_Blight',
                    'Corn_(maize)___healthy',
                    'Grape___Esca_(Black_Measles)',
                    'Strawberry___Leaf_scorch',
                    'Tomato___Early_blight'
                ]

                predicted_label = class_names[result_index]

                st.success(f"🌱 **Model Prediction:** {predicted_label}")

