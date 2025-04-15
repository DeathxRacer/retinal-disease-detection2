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

# Style the page with custom background and design
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
    .image-container {
        border-radius: 10px;
        border: 2px solid #00bcd4;
        padding: 10px;
    }
    .button {
        background-color: #00bcd4;
        color: #fff;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
        border: none;
    }
    .button:hover {
        background-color: #008c99;
    }
    .progress-bar {
        height: 25px;
        border-radius: 10px;
    }
    
    /* CSS for floating file uploader box */
    .stFileUploader {
        position: fixed;
        top: 50%;  /* Position vertically centered */
        left: 20px;
        transform: translateY(-50%);  /* Exact vertical centering */
        z-index: 1000;
        width: 250px;  /* Slightly rectangular width */
        height: 150px;  /* Slightly rectangular height */
        border-radius: 15px;  /* Rounded corners */
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.9);  /* Semi-transparent background */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        display: flex;
        justify-content: center;
        align-items: center;
        border: 2px solid #00bcd4;  /* Border similar to page style */
    }

    /* Style for the file uploader text */
    .stFileUploader .css-1k76s6d {
        background-color: #00bcd4;
        color: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        font-size: 14px;
        width: 100%;
        height: 100%;
    }

    .stFileUploader .css-1k76s6d:hover {
        background-color: #008c99;
    }
    </style>
""", unsafe_allow_html=True)

# Title section
st.markdown('<h1 class="title">👁️ Retinal Disease Diagnosis🩺</h1>', unsafe_allow_html=True)
st.image("https://chromaviso.com/hubfs/Blog/shutterstock_1962443701_Lille.jpeg", caption="Human Eye - Retinal Analysis", use_column_width=True)

# Description section
st.markdown('<p class="description">Upload a retinal image to analyze for common diseases using deep learning models. See results instantly!</p>', unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Retinal Image", use_column_width=True, channels="RGB")

    if st.button("🔍 Classify", key="class
