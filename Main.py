import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import os

# Define disease labels
disease_labels = [
    "Diabetic Retinopathy",
    "Macular Edema",
    "Glaucoma",
    "Macular Degeneration",
    "Retinal Vascular Occlusion",
    "Opacity",
    "Normal"  # 👈 Commonly used as the 7th class in retinal classification

]

# Load the trained model
def load_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(disease_labels))  # Output layer for the number of diseases
    model.load_state_dict(torch.load("models/ResNet50_v1.0.ipynb.pth", map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Retinal Disease Classification")
st.write("Upload a retinal image to classify diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    
    if st.button("Classify Image"):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)  # Get raw logits from the model
            probs = torch.softmax(output, dim=1).squeeze().numpy()  # Apply softmax to get probabilities
        
        st.subheader("Prediction Results:")
        for i, disease in enumerate(disease_labels):
            st.write(f"{disease}: {probs[i] * 100:.2f}% confidence")  # Show prediction probabilities

st.write("---")
st.write("Developed using PyTorch and Streamlit")
