import streamlit as st
import tensorflow as tf
import numpy as np

# 🌿 Disease Information (Easy-to-Understand Version)
disease_info = {
    "Apple___Apple_scab": {
        "Causes": "Caused by the fungus *Venturia inaequalis*. Spreads in cool, wet weather through infected leaves.",
        "Symptoms": "Dark spots on leaves and fruit, causing reduced yield.",
        "How to Save Your Crops": [
            "✅ Remove and destroy infected leaves.",
            "✅ Apply fungicide (e.g., *Mancozeb, Captan*) in early spring.",
            "✅ Use resistant apple varieties."
        ],
        "Best Fertilizers": [
            "✔ Potassium-rich fertilizer for stronger plants.",
            "✔ Organic compost to enhance soil health."
        ]
    },
    "Apple___Cedar_apple_rust": {
        "Causes": "Fungus *Gymnosporangium juniperi-virginianae*. Requires both cedar and apple trees to complete its lifecycle.",
        "Symptoms": "Orange rust spots on leaves, reducing fruit quality.",
        "How to Save Your Crops": [
            "✅ Remove nearby cedar/juniper trees.",
            "✅ Apply fungicide (e.g., *Myclobutanil, Mancozeb*) before bud break.",
            "✅ Use resistant apple varieties."
        ],
        "Best Fertilizers": [
            "✔ Balanced NPK (10-10-10) fertilizer for plant immunity.",
            "✔ Calcium nitrate for stronger leaves."
        ]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "Causes": "Fungus *Exserohilum turcicum*. Spreads in humid, rainy conditions.",
        "Symptoms": "Long gray-green lesions on corn leaves.",
        "How to Save Your Crops": [
            "✅ Use resistant corn hybrids.",
            "✅ Apply fungicides (e.g., *Propiconazole, Azoxystrobin*).",
            "✅ Rotate crops every 2-3 years to prevent soil infection."
        ],
        "Best Fertilizers": [
            "✔ Nitrogen-based fertilizer to promote healthy leaf growth.",
            "✔ Phosphorus-rich fertilizer for strong roots."
        ]
    },
    "Grape___Esca_(Black_Measles)": {
        "Causes": "Fungi *Phaeomoniella chlamydospora* and *Phaeoacremonium aleophilum*. Spreads through pruning wounds.",
        "Symptoms": "Brown streaking inside wood, vine collapse.",
        "How to Save Your Crops": [
            "✅ Avoid pruning in wet weather.",
            "✅ Seal pruning cuts with fungicidal paste.",
            "✅ Remove and destroy infected vines."
        ],
        "Best Fertilizers": [
            "✔ Organic compost to improve soil health.",
            "✔ Balanced NPK fertilizer to strengthen vines."
        ]
    },
    "Strawberry___Leaf_scorch": {
        "Causes": "Fungus *Diplocarpon earlianum*. Thrives in warm, humid conditions.",
        "Symptoms": "Brown spots with purple edges on strawberry leaves.",
        "How to Save Your Crops": [
            "✅ Remove infected leaves before winter.",
            "✅ Apply fungicide (e.g., *Captan, Chlorothalonil*).",
            "✅ Ensure good air circulation around plants."
        ],
        "Best Fertilizers": [
            "✔ High phosphorus fertilizer to promote fruiting.",
            "✔ Organic manure for better soil structure."
        ]
    },
    "Tomato___Early_blight": {
        "Causes": "Fungus *Alternaria solani*. Spreads through infected seeds, soil, and splashing water.",
        "Symptoms": "Brown concentric rings on tomato leaves.",
        "How to Save Your Crops": [
            "✅ Rotate crops to prevent soil contamination.",
            "✅ Use resistant tomato varieties.",
            "✅ Apply fungicides (e.g., *Copper Fungicide, Mancozeb*)."
        ],
        "Best Fertilizers": [
            "✔ Potassium-based fertilizers for plant immunity.",
            "✔ Calcium-rich fertilizers to reduce fruit rot."
        ]
    }
}

# 🌱 Load the trained model
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model2.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# 🌾 Sidebar Navigation
st.sidebar.title("👨‍🌾 Farmer’s Dashboard")
app_mode = st.sidebar.selectbox("📌 Choose Option", ["🏡 Home", "📖 Info", "🌿 Detect Disease"])

# 🏡 Home Page
if app_mode == "🏡 Home":
    st.header("🌾 Crop Disease Detection System")
    st.image("/Users/ritvikkharde/Downloads/Plant_disease.jpg", use_column_width=True)
    st.markdown("""
    **Welcome Farmers! 🙏**  
    Use this system to **detect crop diseases and get prevention tips.**  

    **How to Use?**  
    1️⃣ **Upload an image of your crop** under "🌿 Detect Disease"  
    2️⃣ **Our AI will analyze and detect the disease**  
    3️⃣ **Get treatment & fertilizer recommendations**  

    ✅ **Start by clicking "🌿 Detect Disease" now!**  
    """)

# 📖 Info Page
elif app_mode == "📖 Info":
    st.header("📚 Learn About Crop Diseases")
    st.markdown("""
    🌾 This system is trained on **87,000+ crop images** using **AI-powered deep learning**.  
    📊 It can detect **7+ crop diseases** accurately.  

    ✅ **Simple, Fast, and Accurate!**  
    """)

# 🌿 Disease Detection Page
elif app_mode == "🌿 Detect Disease":
    st.header("🌱 Upload Crop Image for Disease Detection")
    test_image = st.file_uploader("📸 Upload an image:")

    if st.button("🖼️ Show Image"):
        st.image(test_image, use_column_width=True)

    # 🔍 Predict Disease
    if st.button("🔍 Detect Disease"):
        with st.spinner("⏳ Please wait..."):
            st.write("🔎 AI is analyzing your image...")
            result_index = model_prediction(test_image)
            
            # Crop Disease Classes
            class_name = [
                'Apple___Apple_scab',
                'Apple___Cedar_apple_rust',
                'Apple___healthy',
                'Corn_(maize)___Northern_Leaf_Blight',
                'Corn_(maize)___healthy',
                'Grape___Esca_(Black_Measles)',
                'Strawberry___Leaf_scorch',
                'Tomato___Early_blight'
            ]

            predicted_disease = class_name[result_index]
            st.success(f"🌾 Your crop has `{predicted_disease}` disease.")

            # ✅ Show Prevention & Fertilizer Advice
            if predicted_disease in disease_info:
                st.subheader("🦠 What is causing this?")
                st.write(f"👉 {disease_info[predicted_disease]['Causes']}")

                st.subheader("🛡️ How to Protect Your Crop?")
                for prevention in disease_info[predicted_disease]['How to Save Your Crops']:
                    st.write(f"✔ {prevention}")

                st.subheader("🌱 Best Fertilizers to Use")
                for fertilizer in disease_info[predicted_disease]['Best Fertilizers']:
                    st.write(f"✅ {fertilizer}")
