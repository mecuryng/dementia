
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np
import os

# ==============================
# Load Models
# ==============================

# ==============================
# Model Classes and Loaders
# ==============================
class CNN_MRI(nn.Module):
    def __init__(self):
        super(CNN_MRI, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision', 'resnet18', weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)
    def forward(self, x):
        return self.resnet(x)

# Load MRI model
try:
    mri_model = CNN_MRI()
    state_dict = torch.load("mri_model.pth", map_location="cpu")
    if any(k.startswith("model.") for k in state_dict.keys()):
        new_state_dict = {k.replace("model.", "resnet.", 1): v for k, v in state_dict.items()}
        mri_model.load_state_dict(new_state_dict)
    else:
        mri_model.load_state_dict(state_dict)
    mri_model.eval()
except Exception as e:
    st.error(f"Error loading MRI model: {e}")
    mri_model = None

# Load XGBoost model
try:
    xgb_model = joblib.load("xgb_model.pkl")
except Exception as e:
    st.error(f"Error loading XGBoost model: {e}")
    xgb_model = None

# Load Scaler
try:
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.warning("Scaler not found. Please ensure 'scaler.pkl' is present.")
    scaler = None

# ==============================
# Image Preprocessing
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==============================
# Streamlit UI
# ==============================

# ==============================
# Streamlit UI
# ==============================
st.title("🧠 Multimodal Dementia Prediction")
st.markdown("""
<style>
.stButton>button {background-color: #4CAF50; color: white; font-weight: bold;}
</style>
""", unsafe_allow_html=True)
st.write("Upload an MRI image and enter clinical data for prediction.")

# MRI upload
uploaded_image = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])



# Clinical data input with descriptive labels
st.subheader("Clinical Data Input")
clinical_features = []
feature_labels = [
    "Age (years)",
    "Gender (0=Male, 1=Female)",
    "MMSE Score",
    "Education (years)",
    "APOE4 Status (0=No, 1=Yes)",
    "Hippocampal Volume",
    "Ventricular Volume",
    "Cortical Thickness",
    "CSF Tau Level",
    "CSF Amyloid-beta Level"
]
for label in feature_labels:
    val = st.number_input(label, value=0)
    clinical_features.append(val)


if st.button("Predict"):
    if mri_model is None or xgb_model is None or scaler is None:
        st.error("Model or scaler not loaded. Please check your files.")
    elif uploaded_image is None:
        st.error("Please upload an MRI image.")
    else:
        try:
            # Process MRI
            image = Image.open(uploaded_image).convert("RGB")
            image = transform(image).unsqueeze(0)  # [1,3,224,224]
            with torch.no_grad():
                mri_features = mri_model(image)

            # Process Tabular data
            clinical_array = np.array(clinical_features).reshape(1, -1)
            scaled_sample_data_point = scaler.transform(clinical_array)
            xgb_pred_prob = xgb_model.predict_proba(scaled_sample_data_point)[:, 1]  # Probability for class 1

            # Combine (simple weighted fusion example)
            alpha, beta = 0.5, 0.5
            combined = alpha * mri_features.mean().item() + beta * xgb_pred_prob[0]

            # Decision
            prediction = 1 if combined > 0.5 else 0
            st.success(f"Prediction: {'Dementia' if prediction==1 else 'No Dementia'}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
