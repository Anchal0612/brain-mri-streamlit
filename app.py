import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model (must be in same folder as this script)
model = load_model("brain_mri_model.h5")  # or 'brain_mri_model.keras' if using new format

# Define class labels
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Streamlit App UI
st.set_page_config(page_title="Brain MRI Tumor Detection", layout="centered")
st.title("🧠 Brain MRI Tumor Classifier")
st.markdown("Upload a brain MRI scan image, and this app will classify the tumor type.")

# Image uploader
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded MRI Scan', use_column_width=True)

    # Preprocess the image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display result
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence * 100:.2f}%")

    # Optional: Show probabilities
    st.subheader("Prediction Probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {prediction[0][i]*100:.2f}%")

