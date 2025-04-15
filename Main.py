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

# Load trained model
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(disease_labels))
    model.load_state_dict(torch.load("models/ResNet50_v1.0.ipynb.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Preprocess uploaded image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Custom CSS for pinning the uploader
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        color: #00bcd4;
        font-weight: bold;
    }
    .description {
        text-align: center;
        font-size: 18px;
        color: #ffffff;
        margin-bottom: 100px;
    }
    .footer {
        text-align: center;
        font-size: 18px;
        color: #f0f0f0;
        margin-top: 50px;
        opacity: 0.8;
    }
    .fixed-uploader {
        position: fixed;
        top: 100px;
        right: 30px;
        width: 300px;
        z-index: 9999;
        background-color: #1c1c1c;
        padding: 10px;
        border-radius: 10px;
        border: 2px solid #00bcd4;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="title">👁️ Retinal Disease Diagnosis🩺</h1>', unsafe_allow_html=True)
st.image("https://chromaviso.com/hubfs/Blog/shutterstock_1962443701_Lille.jpeg", caption="Human Eye - Retinal Analysis", use_column_width=True)
st.markdown('<p class="description">Upload a retinal image to analyze for common diseases using deep learning models. See results instantly!</p>', unsafe_allow_html=True)

# Floating uploader box
with st.container():
    st.markdown('<div class="fixed-uploader">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "jpeg", "png"], key="uploader")
    st.markdown('</div>', unsafe_allow_html=True)

# Only process if uploaded
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Retinal Image", use_column_width=True, channels="RGB")

    if st.button("🔍 Classify", key="classify_button"):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze().numpy()

        st.subheader("📊 Prediction Confidence")
        for i, disease in enumerate(disease_labels):
            st.progress(int(probs[i] * 100))
            st.write(f"{disease}: {probs[i] * 100:.2f}%")

        top_idx = np.argmax(probs)
        st.success(f"🧾 Most likely diagnosis: **{disease_labels[top_idx]}**")

# Footer
st.markdown("""
    <div class="footer">
        Made By Keerthi Vardhan, Sathwik & Sujith · © 2025
    </div>
""", unsafe_allow_html=True)
