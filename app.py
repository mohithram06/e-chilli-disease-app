import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import gdown

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = 224

class_labels = ['cp0','cp1','cp3','cp5','cp7','cp9']

def get_severity(label):
    if label == 'cp0':
        return "Healthy"
    elif label == 'cp1':
        return "Mild"
    elif label == 'cp3':
        return "Moderate"
    elif label == 'cp5':
        return "Moderate-Severe"
    elif label == 'cp7':
        return "Severe"
    elif label == 'cp9':
        return "Very Severe"
    else:
        return "Unknown"

# -------------------------------
# DOWNLOAD MODEL FROM DRIVE
# -------------------------------
MODEL_PATH = "chilli_disease_model.h5"

# 🔴 IMPORTANT: Replace with your file ID
FILE_ID = "PASTE_YOUR_FILE_ID_HERE"

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1eq1AwA9fzxyE-t-EE1cudio95gFwFFhX"
        gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("🌶️ Chilli Disease Detection")
st.info("Upload a clear, close-up image of a chilli leaf")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    pred = model.predict(img_array)
    pred_index = np.argmax(pred)
    confidence = float(np.max(pred))

    pred_label = class_labels[pred_index]
    severity = get_severity(pred_label)

    # -------------------------------
    # OUTPUT
    # -------------------------------
    st.subheader("🌿 Prediction Result")

    st.write(f"**Class:** {pred_label}")
    st.write(f"**Severity:** {severity}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Confidence warning
    if confidence < 0.5:
        st.error("⚠️ Low confidence. Try a clearer leaf image.")

    # Severity indicator
    if severity == "Healthy":
        st.success("Plant is healthy ✅")
    elif severity in ["Mild", "Moderate"]:
        st.warning("Early stage disease ⚠️")
    else:
        st.error("Severe disease detected 🚨")

    # Show probabilities
    st.subheader("All Class Probabilities")
    for i, prob in enumerate(pred[0]):
        st.write(f"{class_labels[i]} : {prob:.2f}")