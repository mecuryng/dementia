
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np
import os

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
     # Display first few weights for verification
    """first_weight = next(mri_model.resnet.parameters()).flatten()[:10].detach().cpu().numpy()
    st.info(f"First 10 weights of loaded MRI model: {first_weight}")"""
except Exception as e:
    st.error(f"Error loading MRI model: {e}")
    mri_model = None 

# Load XGBoost model
try:
    xgb_model = joblib.load("xgb_model.pkl")
except Exception as e:
    st.error(f"Error loading XGBoost model: {e}")
    xgb_model = None

# Load Scaler (check both current and parent directory)
scaler = None
scaler_paths = [
    os.path.join(os.getcwd(), "scaler.pkl"),
    os.path.abspath(os.path.join(os.getcwd(), "..", "scaler.pkl"))
]
for path in scaler_paths:
    if os.path.exists(path):
        try:
            scaler = joblib.load(path)
            break
        except Exception as e:
            st.warning(f"Scaler found at {path} but could not be loaded: {e}")
if scaler is None:
    st.warning("Scaler not found. Please ensure 'scaler.pkl' is present in the app or parent directory.")

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
    "Visit",
    "MR Delay",
    "Age",
    "Education (years)",
    "SES (0=Low, 1=Medium, 2=High)",
    "MMSE Score",
    "Clinical Dementia Rating (CDR)",
    "eTIV (Estimated Total Intracranial Volume)",
    "nWBV (Normalized Whole Brain Volume)",
    "ASF (Atlas-based Segmentation Framework)",
    
]
for label in feature_labels:
    val = st.number_input(label, value=0.0, format="%0.2f")
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
            xgb_pred_class = xgb_model.predict(scaled_sample_data_point)[0]  # Predicted class (0 or 1)

            # Combine (simple weighted fusion example)
            import torch.nn.functional as F
            alpha, beta = 0.5, 0.5
            mri_score_raw = mri_features.mean().item()
            # Normalize MRI output to [0,1] using sigmoid
            mri_score = float(torch.sigmoid(torch.tensor(mri_score_raw)))
            combined = alpha * mri_score + beta * xgb_pred_class

            # Debug output
            """ st.info(f"MRI feature mean (raw): {mri_score_raw:.4f}")
            st.info(f"MRI feature mean (sigmoid): {mri_score:.4f}")
            st.info(f"XGBoost probability: {xgb_pred_prob[0]:.4f}")
            st.info(f"XGBoost predicted class: {xgb_pred_class}")
            st.info(f"Combined score: {combined:.4f}") """

            # Decision
            prediction = 1 if combined > 0.5 else 0
            st.success(f"Prediction: {'Dementia' if prediction==1 else 'No Dementia'}")
            st.success(f"Probability of prediction is: {combined:.4f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
