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

# Styling the page
st.markdown("""
    <style>
    .main {
        background-color: #1c1c1c;
        color: #f0f0f0;
        font-family: 'Segoe UI', sans-serif;
    }
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
        margin-bottom: 30px;
    }
    .footer {
        text-align: center;
        font-size: 18px;
        color: #f0f0f0;
        margin-top: 50px;
        opacity: 0.8;
    }
    .progress-bar {
        height: 25px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Sticky uploader (top-right corner)
st.markdown("""
    <div id="sticky-uploader" style="position:fixed; top:20px; right:20px; z-index:9999; background-color:#1c1c1c; padding:10px; border:2px solid #00bcd4; border-radius:10px; box-shadow:0 0 10px #00bcd4;">
        <label style="color:white;">📁 Upload Retinal Image</label><br/>
        <input type="file" id="custom-upload" accept=".jpg,.jpeg,.png" style="margin-top:8px;" />
    </div>
    <script>
        const customUploader = document.getElementById("custom-upload");
        const realUploader = window.parent.document.querySelector('input[type="file"]');

        customUploader.addEventListener("change", (e) => {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(e.target.files[0]);
            realUploader.files = dataTransfer.files;
            realUploader.dispatchEvent(new Event("change", { bubbles: true }));
        });
    </script>
""", unsafe_allow_html=True)

# Title section with human eye image
st.markdown('<h1 class="title">👁️ Retinal Disease Diagnosis🩺</h1>', unsafe_allow_html=True)
st.image("https://chromaviso.com/hubfs/Blog/shutterstock_1962443701_Lille.jpeg", caption="Human Eye - Retinal Analysis", use_column_width=True)

# Description
st.markdown('<p class="description">Upload a retinal image to analyze for common diseases using deep learning models. See results instantly!</p>', unsafe_allow_html=True)

# Real file uploader (linked with the sticky one)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

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
