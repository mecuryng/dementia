
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np

# ==============================
# Load Models
# ==============================
# Load PyTorch MRI model (ResNet18-based)
class CNN_MRI(nn.Module):
    def __init__(self):
        super(CNN_MRI, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)

    def forward(self, x):
        return self.resnet(x)


# Load model and fix key mismatch if needed
mri_model = CNN_MRI()
state_dict = torch.load("mri_model.pth", map_location="cpu")
# Remove 'model.' prefix if present
if any(k.startswith("model.") for k in state_dict.keys()):
    new_state_dict = {k.replace("model.", "resnet.", 1): v for k, v in state_dict.items()}
    mri_model.load_state_dict(new_state_dict)
else:
    mri_model.load_state_dict(state_dict)
mri_model.eval()

# Load XGBoost model
xgb_model = joblib.load("xgb_model.pkl")

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
st.title("Multimodal Dementia Prediction")
st.write("Upload an MRI image and enter clinical data for prediction.")

# MRI upload
uploaded_image = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

# Clinical data input
clinical_features = []
feature_names = [f"Feature_{i}" for i in range(10)]  # Replace with actual names
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0)
    clinical_features.append(val)

if st.button("Predict"):
    if uploaded_image is not None:
        # Process MRI
        image = Image.open(uploaded_image).convert("RGB")
        image = transform(image).unsqueeze(0)  # [1,3,224,224]
        with torch.no_grad():
            mri_features = mri_model(image)

        # Process Tabular data
        clinical_array = np.array(clinical_features).reshape(1, -1)
        xgb_pred_prob = xgb_model.predict_proba(clinical_array)[:, 1]  # Probability for class 1

        # Combine (simple weighted fusion example)
        alpha, beta = 0.5, 0.5
        combined = alpha * mri_features.mean().item() + beta * xgb_pred_prob[0]

        # Decision
        prediction = 1 if combined > 0.5 else 0
        st.success(f"Prediction: {'Dementia' if prediction==1 else 'No Dementia'}")
    else:
        st.error("Please upload an MRI image.")
