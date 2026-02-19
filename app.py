import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# Set page configuration for a premium look
st.set_page_config(
    page_title="Brain Tumor Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stAlert {
        border-radius: 20px;
    }
    .header-text {
        color: #00d4ff;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='header-text'>🧠 Advanced Brain Tumor Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an MRI scan to detect potential brain tumors using our state-of-the-art AI model.</p>", unsafe_allow_html=True)

from torchvision import models

# Load the model
@st.cache_resource
def load_model():
    model_path = 'best_model_ft.pkl'
    try:
        # 1. Instantiate the architecture (ResNet18)
        model = models.resnet18(weights=None)
        
        # 2. Modify the final fully connected layer to match 4 classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 4)
        
        # 3. Load the state_dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Handle cases where the pkl might be the state_dict directly or wrapped
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            model.load_state_dict(state_dict)
            
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = image.convert('RGB')
    tensor = transform(image).unsqueeze(0)
    return tensor

# Sidebar information
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865801.png", width=100)
    st.title("About the System")
    st.info("This system uses a Deep Learning model trained on MRI datasets to identify brain rumors.")
    st.warning("Note: This is an AI-assisted tool and should not be used as a primary diagnostic tool.")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload MRI Scan")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Scan', use_container_width=True)

with col2:
    st.subheader("Prediction Analysis")
    if uploaded_file is not None:
        if model is not None:
            with st.spinner('Analyzing scan...'):
                processed_image = preprocess_image(image)
                with torch.no_grad():
                    output = model(processed_image)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                
                # Typical 4-class labels for brain tumor datasets
                class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']
                
                result_class = class_names[predicted.item()] if predicted.item() < len(class_names) else f"Class {predicted.item()}"
                conf_score = confidence.item() * 100
                
                if result_class == 'Tumor':
                    st.error(f"Prediction: {result_class}")
                else:
                    st.success(f"Prediction: {result_class}")
                
                st.write(f"Confidence Level: **{conf_score:.2f}%**")
                
                # Visualizing probabilities
                st.progress(conf_score / 100)
        else:
            st.warning("Please ensure the model 'best_model_ft.pkl' is correctly placed in the directory.")
    else:
        st.write("Results will appear here after you upload an image.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.8em;'>© 2026 NeuroScan AI. All rights reserved.</p>", unsafe_allow_html=True)
