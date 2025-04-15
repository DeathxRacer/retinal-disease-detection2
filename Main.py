import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# Disease labels
disease_labels = [
    "Diabetic Retinopathy",
    "Macular Edema",
    "Glaucoma",
    "Macular Degeneration",
    "Retinal Vascular Occlusion",
    "Opacity",
    "Normal"
]

# Load the model
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(disease_labels))
    model.load_state_dict(torch.load("models/ResNet50_v1.0.ipynb.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- 💠 Custom CSS for Pinned Uploader ---
st.markdown("""
    <style>
    .fixed-uploader {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        background-color: #262730;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
        width: 300px;
    }
    </style>
    <div class="fixed-uploader">
        <p style='color: white; font-weight: bold;'>📁 Upload Retinal Image</p>
        <div id="uploader-box"></div>
    </div>
""", unsafe_allow_html=True)

# 🧠 Title & Intro
st.markdown('<h1 style="text-align:center; color:#00bcd4;">👁️ Retinal Disease Diagnosis</h1>', unsafe_allow_html=True)
st.image("https://chromaviso.com/hubfs/Blog/shutterstock_1962443701_Lille.jpeg", caption="Retinal Scan", use_column_width=True)
st.markdown('<p style="text-align:center; color:white;">Upload a retinal image to detect common eye diseases using AI.</p>', unsafe_allow_html=True)

# 👇 Regular file uploader (to be moved using JS)
uploaded_file = st.file_uploader("Upload retinal image", type=["jpg", "jpeg", "png"], key="file-upload")

# 📦 Move uploader into floating box using JavaScript
st.markdown("""
    <script>
    const fileUploader = window.parent.document.querySelector('section[data-testid="stFileUploader"]');
    const targetBox = window.parent.document.getElementById("uploader-box");
    if (fileUploader && targetBox && !targetBox.contains(fileUploader)) {
        targetBox.appendChild(fileUploader);
    }
    </script>
""", unsafe_allow_html=True)

# 🧪 Classification Logic
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Classify"):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze().numpy()

        st.subheader("📊 Prediction Confidence")
        for i, disease in enumerate(disease_labels):
            st.progress(int(probs[i] * 100))
            st.write(f"**{disease}**: {probs[i] * 100:.2f}%")

        top_idx = np.argmax(probs)
        st.success(f"🧾 Most likely diagnosis: **{disease_labels[top_idx]}**")

# 👣 Footer
st.markdown("""
    <div style="text-align:center; margin-top:50px; color:#bbb;">
        Made by Keerti Vardhan, Sathwik & Sujith · © 2025
    </div>
""", unsafe_allow_html=True)
