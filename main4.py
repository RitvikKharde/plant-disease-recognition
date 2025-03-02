import streamlit as st
import tensorflow as tf
import numpy as np
from reportlab.pdfgen import canvas
from googletrans import Translator

# 🌍 Initialize Translator for Multiple Languages
translator = Translator()

# 🏡 Website Title
st.sidebar.title("🚜 Non-Stop Farmers")

# 🌍 Language Selection
language = st.sidebar.selectbox("🌐 Select Language", ["English", "Hindi", "Gujarati", "Marathi"])

# 📌 Sidebar Navigation
app_mode = st.sidebar.radio(translator.translate("📌 Choose Option", dest=language).text, ["🏡 Home", "📖 Info", "🌿 Detect Disease"])

# 🌿 Disease Information (Multilingual Support)
disease_info = {
    "Apple___Apple_scab": {
        "causes": "Caused by the fungus *Venturia inaequalis*. Spreads in cool, wet weather.",
        "symptoms": "Dark spots on leaves and fruit, reducing yield.",
        "prevention": ["Remove infected leaves.", "Apply fungicide (Mancozeb, Captan).", "Use resistant varieties."],
        "fertilizer": ["Potassium-rich fertilizer", "Organic compost"]
    },
    "Apple___Cedar_apple_rust": {
        "causes": "Caused by *Gymnosporangium juniperi-virginianae*, requires both cedar and apple trees.",
        "symptoms": "Orange rust spots on leaves, reducing fruit quality.",
        "prevention": ["Remove nearby cedar trees.", "Apply fungicide before bud break.", "Use resistant varieties."],
        "fertilizer": ["Balanced NPK (10-10-10)", "Calcium Nitrate"]
    },
    "Apple___healthy": {
        "causes": "All good for your plant.",
        "symptoms": "No symptoms.",
        "prevention": ["Maintain proper watering.", "Use organic pesticides.", "Regularly check for early disease signs."],
        "fertilizer": ["General NPK Fertilizer (10-10-10)", "Cow Manure"]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "causes": "Caused by the fungus *Exserohilum turcicum*. Spreads in humid, rainy conditions.",
        "symptoms": "Long gray-green lesions on corn leaves, causing reduced photosynthesis.",
        "prevention": ["Use resistant corn hybrids.", "Apply fungicides (Propiconazole, Azoxystrobin).", "Rotate crops every 2-3 years."],
        "fertilizer": ["Nitrogen-based fertilizer", "Phosphorus-rich fertilizer"]
    },
    "Corn_(maize)___healthy": {
        "causes": "All good for your plant.",
        "symptoms": "No symptoms.",
        "prevention": ["Rotate crops to prevent soil depletion.", "Use disease-resistant seeds.", "Avoid overwatering."],
        "fertilizer": ["Balanced NPK Fertilizer (10-10-10)", "Organic compost"]
    },
    "Grape___Esca_(Black_Measles)": {
        "causes": "Caused by fungi *Phaeomoniella chlamydospora* and *Phaeoacremonium aleophilum*. Spreads through pruning wounds.",
        "symptoms": "Brown streaking inside wood, vine collapse.",
        "prevention": ["Avoid pruning in wet weather.", "Seal pruning cuts with fungicidal paste.", "Remove and destroy infected vines."],
        "fertilizer": ["Organic compost", "Balanced NPK fertilizer"]
    },
    "Strawberry___Leaf_scorch": {
        "causes": "Caused by the fungus *Diplocarpon earlianum*. Thrives in warm, humid conditions.",
        "symptoms": "Brown spots with purple edges on strawberry leaves.",
        "prevention": ["Remove infected leaves before winter.", "Apply fungicide (Captan, Chlorothalonil).", "Ensure good air circulation around plants."],
        "fertilizer": ["High phosphorus fertilizer", "Organic manure"]
    },
    "Tomato___Early_blight": {
        "causes": "Caused by the fungus *Alternaria solani*. Spreads through infected seeds, soil, and splashing water.",
        "symptoms": "Brown concentric rings on tomato leaves.",
        "prevention": ["Rotate crops to prevent soil contamination.", "Use resistant tomato varieties.", "Apply fungicides (Copper Fungicide, Mancozeb)."],
        "fertilizer": ["Potassium-based fertilizers", "Calcium-rich fertilizers"]
    }
}

# 🌾 Load ML Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model2.keras")

model = load_model()

# 📜 Generate PDF Report
def generate_report(disease, translated_data):
    filename = "Crop_Report.pdf"
    c = canvas.Canvas(filename)
    c.drawString(100, 800, "Crop Disease Report")
    c.drawString(100, 780, f"Disease: {disease}")
    c.drawString(100, 740, "Details:")
    
    y_position = 720
    for key, value in translated_data.items():
        c.drawString(100, y_position, f"{key}: {value}")
        y_position -= 20
    
    c.save()
    return filename

def model_prediction(test_image):
    model  = tf.keras.models.load_model('trained_model2.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# 📸 Image Prediction
def predict_disease(image):
    image = tf.keras.preprocessing.image.load_img(image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) / 255.0

    prediction = model.predict(input_arr)
    return np.argmax(prediction), max(prediction[0])

# 🏡 Home Page
if app_mode == "🏡 Home":
    st.header(translator.translate("🌾 Crop Disease Detection System", dest=language).text)
    st.image("/Users/ritvikkharde/Downloads/Plant_disease.jpg", use_column_width=True)
    st.markdown(translator.translate("""
    **Welcome Farmers! 🙏**  
    Use this system to **detect crop diseases and get prevention tips.**  

    **How to Use?**  
    1️⃣ **Upload an image of your crop** under "🌿 Detect Disease"  
    2️⃣ **Our AI will analyze and detect the disease**  
    3️⃣ **Get treatment & fertilizer recommendations**  

    ✅ **Start by clicking "🌿 Detect Disease" now!**  
    """, dest=language).text)

# 📖 Info Page
elif app_mode == "📖 Info":
    st.header(translator.translate("📚 Learn About Crop Diseases", dest=language).text)
    st.markdown(translator.translate("""
    🌾 This system is trained on **15,000+ crop images** using **AI-powered deep learning**.  
    📊 It can detect **multiple crop diseases accurately**.  

    ✅ **Simple, Fast, and Accurate!**  
    """, dest=language).text)

# 🌿 Disease Detection Page
elif app_mode == "🌿 Detect Disease":
    st.header(translator.translate("🌱 Upload Crop Image for Disease Detection", dest=language).text)
    test_image = st.file_uploader(translator.translate("📸 Upload an image:", dest=language).text)

    if st.button(translator.translate("🖼️ Show Image", dest=language).text):
        st.image(test_image, use_column_width=True)

    # 🔍 Predict Disease
    if st.button(translator.translate("🔍 Detect Disease", dest=language).text):
        with st.spinner(translator.translate("⏳ Please wait...", dest=language).text):
            result_index = model_prediction(test_image)

            class_name = ['Apple___Apple_scab',
                'Apple___Cedar_apple_rust',
                'Apple___healthy',
                'Corn_(maize)___Northern_Leaf_Blight',
                'Corn_(maize)___healthy',
                'Grape___Esca_(Black_Measles)',
                'Strawberry___Leaf_scorch',
                'Tomato___Early_blight']

            predicted_disease = class_name[result_index]
            st.success(f"🌾 {translator.translate('Your crop has:', dest=language).text} `{predicted_disease}`")

            if predicted_disease in disease_info:
                # 🌍 Translate Disease Information
                translated_data = {
                "Disease": translator.translate(predicted_disease, dest=language).text,
                "Causes": translator.translate(disease_info[predicted_disease]["causes"], dest=language).text,
                "Symptoms": translator.translate(disease_info[predicted_disease]["symptoms"], dest=language).text,
                "Prevention": [translator.translate(prevention, dest=language).text for prevention in disease_info[predicted_disease]["prevention"]],
                "Fertilizer": [translator.translate(fertilizer, dest=language).text for fertilizer in disease_info[predicted_disease]["fertilizer"]]
            }

                # 🛡️ Display Results
                st.success(f"🌾 {translated_data['Disease']} detected!")
                st.subheader(translator.translate("🦠 Causes:", dest=language).text)
                st.write(f"👉 {translated_data['Causes']}")
                st.subheader(translator.translate("🛑 Symptoms:", dest=language).text)
                st.write(f"🔹 {translated_data['Symptoms']}")

                st.subheader(translator.translate("🛡️ Prevention Methods:", dest=language).text)
                for prevention in translated_data["Prevention"]:
                    st.write(f"✔ {prevention}")

                st.subheader(translator.translate("🌱 Recommended Fertilizers:", dest=language).text)
                for fertilizer in translated_data["Fertilizer"]:
                    st.write(f"✅ {fertilizer}")

                # 📜 Generate & Download Report
                report_file = generate_report(predicted_disease,  translated_data)
                with open(report_file, "rb") as file:
                    st.download_button(label=translator.translate("📄 Download Report", dest=language).text, data=file, file_name="Crop_Report.pdf")
