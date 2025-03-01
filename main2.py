import streamlit as st
import tensorflow as tf
import numpy as np

# 🌍 Language Dictionary
translations = {
    "en": {
        "title": "🌾 Crop Disease Detection System",
        "welcome": "👨‍🌾 Welcome, Farmers! Use this system to detect crop diseases and get prevention tips.",
        "upload": "📸 Upload an Image",
        "detect": "🔍 Detect Disease",
        "disease_detected": "🌾 Your crop has",
        "causes": "🦠 What is causing this?",
        "symptoms": "🛑 Symptoms",
        "prevention": "🛡️ How to Protect Your Crop?",
        "fertilizer": "🌱 Best Fertilizers to Use",
        "healthy_crop": "🎉 Your crop is healthy! Keep following these best practices."
    },
    "hi": {
        "title": "🌾 फसल रोग पहचान प्रणाली",
        "welcome": "👨‍🌾 नमस्ते किसान भाइयों! इस प्रणाली का उपयोग करें और अपनी फसल की बीमारियों का पता लगाएं।",
        "upload": "📸 अपनी फसल की तस्वीर अपलोड करें",
        "detect": "🔍 रोग पहचान करें",
        "disease_detected": "🌾 आपकी फसल को यह रोग हुआ है:",
        "causes": "🦠 इस बीमारी का कारण क्या है?",
        "symptoms": "🛑 लक्षण",
        "prevention": "🛡️ फसल को कैसे बचाएं?",
        "fertilizer": "🌱 कौन-सा खाद उपयोग करें?",
        "healthy_crop": "🎉 आपकी फसल स्वस्थ है! इन सुझावों का पालन करें।"
    }
}

# 🌱 Disease Information (English & Hindi) for all 8 conditions
disease_info = {
    "Apple___Apple_scab": {
        "causes": {
            "en": "Caused by the fungus *Venturia inaequalis*. Spreads in cool, wet weather.",
            "hi": "यह *Venturia inaequalis* नामक फफूंद से होता है, जो ठंडे और गीले मौसम में फैलता है।"
        },
        "symptoms": {
            "en": "Dark spots on leaves and fruit, reducing yield.",
            "hi": "पत्तियों और फलों पर काले धब्बे, जिससे पैदावार कम होती है।"
        },
        "prevention": {
            "en": ["Remove infected leaves.", "Apply fungicide (Mancozeb, Captan).", "Use resistant varieties."],
            "hi": ["संक्रमित पत्तियां हटाएं।", "फफूंदनाशक (Mancozeb, Captan) का छिड़काव करें।", "रोग प्रतिरोधी किस्में लगाएं।"]
        },
        "fertilizer": {
            "en": ["Potassium-rich fertilizer", "Organic compost"],
            "hi": ["पोटैशियम युक्त खाद", "जैविक खाद"]
        }
    },
    "Apple___Cedar_apple_rust": {
        "causes": {
            "en": "Caused by *Gymnosporangium juniperi-virginianae*, requires both cedar and apple trees.",
            "hi": "यह *Gymnosporangium juniperi-virginianae* से होता है और इसे देवदार और सेब के पेड़ों की जरूरत होती है।"
        },
        "symptoms": {
            "en": "Orange rust spots on leaves, reducing fruit quality.",
            "hi": "पत्तियों पर नारंगी धब्बे, जिससे फल की गुणवत्ता कम होती है।"
        },
        "prevention": {
            "en": ["Remove nearby cedar trees.", "Apply fungicide before bud break.", "Use resistant varieties."],
            "hi": ["आसपास के देवदार के पेड़ हटाएं।", "बड ब्रेक से पहले फफूंदनाशक छिड़कें।", "रोग प्रतिरोधी किस्में लगाएं।"]
        },
        "fertilizer": {
            "en": ["Balanced NPK (10-10-10)", "Calcium Nitrate"],
            "hi": ["संतुलित एनपीके (10-10-10)", "कैल्शियम नाइट्रेट"]
        }
    },
    "Apple___healthy": {
        "causes": {
            "en": "All good for you plant",
            "hi": "आपके पौधे के लिए सब कुछ अच्छा है"
        },
        "symptoms": {
            "en": "No Symptoms",
            "hi": "कोई लक्षण नहीं"
        },
        "prevention": {
            "en": ["Maintain proper watering.", "Use organic pesticides.", "Regularly check for early disease signs."],
            "hi": ["सही मात्रा में पानी दें।", "जैविक कीटनाशकों का उपयोग करें।", "रोग के लक्षणों की नियमित जांच करें।"]
        },
        "fertilizer": {
            "en": ["General NPK Fertilizer (10-10-10)", "Cow Manure"],
            "hi": ["सामान्य एनपीके खाद (10-10-10)", "गाय का गोबर"]
        }
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "causes": {
            "en": "Caused by the fungus *Exserohilum turcicum*. Spreads in humid, rainy conditions.",
            "hi": "यह *Exserohilum turcicum* नामक फफूंद के कारण होता है, जो आर्द्र और बरसाती मौसम में फैलता है।"
        },
        "symptoms": {
            "en": "Long gray-green lesions on corn leaves, causing reduced photosynthesis.",
            "hi": "मक्का की पत्तियों पर लंबे भूरे-हरे धब्बे, जिससे प्रकाश संश्लेषण कम हो जाता है।"
        },
        "prevention": {
            "en": ["Use resistant corn hybrids.", "Apply fungicides (Propiconazole, Azoxystrobin).", "Rotate crops every 2-3 years."],
            "hi": ["रोग प्रतिरोधी मक्का किस्में लगाएं।", "फफूंदनाशक (Propiconazole, Azoxystrobin) छिड़कें।", "हर 2-3 साल में फसल चक्र अपनाएं।"]
        },
        "fertilizer": {
            "en": ["Nitrogen-based fertilizer", "Phosphorus-rich fertilizer"],
            "hi": ["नाइट्रोजन युक्त खाद", "फास्फोरस युक्त खाद"]
        }
    },
    "Corn_(maize)___healthy": {
        "causes": {
            "en": "All good for you plant",
            "hi": "आपके पौधे के लिए सब कुछ अच्छा है" 
        },
        "symptoms": {
            "en": "No Symptoms",
            "hi": "कोई लक्षण नहीं"
        },
        "prevention": {
            "en": ["Rotate crops to prevent soil depletion.", "Use disease-resistant seeds.", "Avoid overwatering."],
            "hi": ["फसल चक्र अपनाएं।", "रोग प्रतिरोधी बीजों का उपयोग करें।", "अधिक पानी न दें।"]
        },
        "fertilizer": {
            "en": ["Balanced NPK Fertilizer (10-10-10)", "Organic compost"],
            "hi": ["संतुलित एनपीके खाद (10-10-10)", "जैविक खाद"]
        }
    },
    "Grape___Esca_(Black_Measles)": {
        "causes": {
            "en": "Caused by fungi *Phaeomoniella chlamydospora* and *Phaeoacremonium aleophilum*. Spreads through pruning wounds.",
            "hi": "यह *Phaeomoniella chlamydospora* और *Phaeoacremonium aleophilum* फफूंदों के कारण होता है, जो कटाई के घावों से फैलता है।"
        },
        "symptoms": {
            "en": "Brown streaking inside wood, vine collapse.",
            "hi": "लकड़ी के अंदर भूरे धब्बे, जिससे बेलें सूख जाती हैं।"
        },
        "prevention": {
            "en": ["Avoid pruning in wet weather.", "Seal pruning cuts with fungicidal paste.", "Remove and destroy infected vines."],
            "hi": ["गीले मौसम में कटाई से बचें।", "कटाई के घावों पर फफूंदनाशक लेप लगाएं।", "संक्रमित बेलों को हटा कर नष्ट करें।"]
        },
        "fertilizer": {
            "en": ["Organic compost", "Balanced NPK fertilizer"],
            "hi": ["जैविक खाद", "संतुलित एनपीके खाद"]
        }
    },
    "Strawberry___Leaf_scorch": {
        "causes": {
            "en": "Caused by the fungus *Diplocarpon earlianum*. Thrives in warm, humid conditions.",
            "hi": "यह *Diplocarpon earlianum* नामक फफूंद के कारण होता है, जो गर्म और नम स्थितियों में पनपता है।"
        },
        "symptoms": {
            "en": "Brown spots with purple edges on strawberry leaves.",
            "hi": "स्ट्रॉबेरी की पत्तियों पर भूरे धब्बे और बैंगनी किनारे।"
        },
        "prevention": {
            "en": ["Remove infected leaves before winter.", "Apply fungicide (Captan, Chlorothalonil).", "Ensure good air circulation around plants."],
            "hi": ["सर्दियों से पहले संक्रमित पत्तियों को हटा दें।", "फफूंदनाशक (Captan, Chlorothalonil) छिड़कें।", "पौधों के चारों ओर अच्छी हवा संचार बनाए रखें।"]
        },
        "fertilizer": {
            "en": ["High phosphorus fertilizer", "Organic manure"],
            "hi": ["उच्च फास्फोरस युक्त खाद", "जैविक खाद"]
        }
    },
    "Tomato___Early_blight": {
        "causes": {
            "en": "Caused by the fungus *Alternaria solani*. Spreads through infected seeds, soil, and splashing water.",
            "hi": "यह *Alternaria solani* फफूंद के कारण होता है, जो संक्रमित बीज, मिट्टी और पानी के छींटों से फैलता है।"
        },
        "symptoms": {
            "en": "Brown concentric rings on tomato leaves.",
            "hi": "टमाटर की पत्तियों पर भूरे रंग के छल्ले।"
        },
        "prevention": {
            "en": ["Rotate crops to prevent soil contamination.", "Use resistant tomato varieties.", "Apply fungicides (Copper Fungicide, Mancozeb)."],
            "hi": ["मिट्टी को संक्रमित होने से बचाने के लिए फसल चक्र अपनाएं।", "रोग प्रतिरोधी टमाटर की किस्में लगाएं।", "फफूंदनाशक (Copper Fungicide, Mancozeb) का छिड़काव करें।"]
        },
        "fertilizer": {
            "en": ["Potassium-based fertilizers", "Calcium-rich fertilizers"],
            "hi": ["पोटैशियम युक्त खाद", "कैल्शियम युक्त खाद"]
        }
    }
}

# 🌾 Load ML Model
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model2.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# 🌍 Language Selection
st.sidebar.title("🌐 Select Language / भाषा चुनें")
language = st.sidebar.radio("", ["English", "हिन्दी"])
lang_code = "en" if language == "English" else "hi"

# 🏡 Home Page
st.title(translations[lang_code]["title"])
st.markdown(translations[lang_code]["welcome"])

# 📸 Image Upload
test_image = st.file_uploader(translations[lang_code]["upload"])

if st.button("🖼️ Show Image"):
    st.image(test_image, use_column_width=True)

# 🔍 Predict Disease
if st.button(translations[lang_code]["detect"]):
    with st.spinner("⏳ Please wait..."):
        result_index = model_prediction(test_image)

        # 🌱 Disease Classes
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
        st.success(f"{translations[lang_code]['disease_detected']} `{predicted_disease}`")

        # 🛡️ Show Disease Information
        if predicted_disease in disease_info:
            st.subheader(translations[lang_code]["causes"])
            st.write(f"👉 {disease_info[predicted_disease]['causes'][lang_code]}")

            st.subheader(translations[lang_code]["symptoms"])
            st.write(f"🔹 {disease_info[predicted_disease]['symptoms'][lang_code]}")

            st.subheader(translations[lang_code]["prevention"])
            for prevention in disease_info[predicted_disease]["prevention"][lang_code]:
                st.write(f"✔ {prevention}")

            st.subheader(translations[lang_code]["fertilizer"])
            for fertilizer in disease_info[predicted_disease]["fertilizer"][lang_code]:
                st.write(f"✅ {fertilizer}")
