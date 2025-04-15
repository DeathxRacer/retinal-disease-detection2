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

# Style the page
st.markdown("""
    <style>
    .main {
        background-color: #0f0f0f;
        color: #f0f0f0;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 36px;
        color: #00f5d4;
    }
    .footer {
        text-align: center;
        font-size: 13px;
        color: #888;
        margin-top: 60px;
    }
    </style>
""", unsafe_allow_html=True)

# Title section
st.markdown('<h1 class="title">🧠 Retinal Disease Classifier</h1>', unsafe_allow_html=True)
st.write("Upload a retinal image to analyze for common diseases using deep learning.")

# Upload image
uploaded_file = st.file_uploader("Upload retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Retinal Image", use_column_width=True)

    if st.button("🔍 Classify"):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze().numpy()

        st.subheader("📊 Prediction Confidence")
        for i, disease in enumerate(disease_labels):
            st.progress(min(float(probs[i]), 1.0))  # progress bar
            st.write(f"**{disease}**: `{probs[i] * 100:.2f}%`")

        top_idx = np.argmax(probs)
        st.success(f"🧾 Most likely diagnosis: **{disease_labels[top_idx]}**")

# Footer
st.markdown('<div class="footer">Made with ❤️ using PyTorch & Streamlit · © 2025</div>', unsafe_allow_html=True)
