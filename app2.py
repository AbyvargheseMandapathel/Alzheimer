import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import tempfile
import matplotlib.pyplot as plt
from explainability.grad_cam import GradCAM, overlay_gradcam
from explainability.shap_explainer import generate_shap_explanation
from explainability.counterfactual_image import ImageCounterfactual, overlay_counterfactual
from src.counterfactual_explainer import CounterfactualExplainer
from src.model import RandomForestModel
from models.cnn_model import AlzheimerResNet18  # Using the Legacy ResNet18
import joblib
import pandas as pd
import io
import os
try:
    from fpdf import FPDF
    HAVE_FPDF = True
except ImportError:
    HAVE_FPDF = False

# Class map
CLASS_NAMES = {0: "Non Demented", 1: "Very Mild Dementia", 2: "Mild / Moderate Dementia"}


st.set_page_config(page_title="Alzheimer's Disease Detection (Legacy v1.0)", layout="wide")

@st.cache_resource
def load_model(model_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Map selection to file path
    path = "weights/best_centralized_model.pth" if model_type == "Centralized" else "weights/best_federated_model.pth"
    
    # Instantiate the Legacy ResNet18 model
    model = AlzheimerResNet18(num_classes=3).to(device)
    
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            st.sidebar.success(f"✅ Loaded Legacy {model_type} Model weights.")
        except RuntimeError as e:
            st.sidebar.error(f"❌ Error loading weights: {str(e)}")
            st.sidebar.info("This legacy app expects ResNet18 weights (~44MB).")
    else:
        st.sidebar.warning(f"⚠️ {model_type} weights not found at `{path}`. Using untrained model!")
        
    model.eval()
    return model, device

@st.cache_resource
def load_tabular_model():
    model_path = "results/global_model.joblib"
    data_path = "Datasets/Multimodal Dataset/phase1_processed.csv"
    
    if os.path.exists(model_path) and os.path.exists(data_path):
        rf_model = joblib.load(model_path)
        df = pd.read_csv(data_path)
        explainer = CounterfactualExplainer(rf_model, df)
        return rf_model, df, explainer
    return None, None, None

@st.cache_data
def load_performance_metrics():
    # EXCLUSIVELY use MRI metrics for the updated app
    mri_metrics_path = "results/mri_metrics.csv"
    
    if os.path.exists(mri_metrics_path):
        df = pd.read_csv(mri_metrics_path)
        df['accuracy_pct'] = df['accuracy'] * 100
        return df
    return None



st.title("🧠 Alzheimer's Diagnostic Center (Legacy ResNet18)")
st.markdown("### Using stable ResNet18 architecture for existing weight compatibility")

# Global Data Loading
metrics_df = load_performance_metrics()

# Sidebar selection
st.sidebar.title("Configuration")
model_selection = "Centralized"
model, device = load_model(model_selection)

# Display MRI Model Performance in Sidebar
st.sidebar.divider()
st.sidebar.subheader("📈 MRI Model Reliability")


if metrics_df is not None:

    latest = metrics_df.iloc[-1]
    best_acc = metrics_df['accuracy_pct'].max()
    
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        st.metric("Best Accuracy", f"{best_acc:.1f}%")
        st.metric("Latest Recall", f"{latest['recall']:.2f}")
    with col_b:
        st.metric("Latest Precision", f"{latest['precision']:.2f}")
        st.metric("Latest F1", f"{latest['f1']:.2f}")
else:
    st.sidebar.info("Performance stats not yet available.")

# --- MRI Scan Analysis ---
uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])


if uploaded_file is not None:
    # 1. Preprocessing & Prediction (Run this first)
    image = Image.open(uploaded_file).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_t = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[class_idx].item() * 100

    # --- 2. Diagnosis Header (Top of Page) ---
    st.markdown("---")
    st.markdown(f"### 🩺 **Diagnosis Prediction:** {CLASS_NAMES[class_idx]} ({confidence:.2f}% confidence)")
    
    # 3. MRI Model Reliability (Metrics)
    if metrics_df is not None:
        latest = metrics_df.iloc[-1]
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("MRI Model Accuracy", f"{latest['accuracy']*100:.1f}%")
        with mcol2:
            st.metric("Precision", f"{latest['precision']:.2f}")
        with mcol3:
            st.metric("Recall", f"{latest['recall']:.2f}")
        with mcol4:
            st.metric("F1 Score", f"{latest['f1']:.2f}")
    else:
        st.info("📊 **Note:** MRI-specific training metrics are not yet generated. Please run `mri_train.py` to see updated model reliability.")

    # 4. Confidence Bars
    for i, class_name in CLASS_NAMES.items():
        st.progress(probabilities[i].item(), text=f"{class_name}: {probabilities[i].item()*100:.2f}%")
        
    st.markdown("---")

    # --- 5. Visualization Columns ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original Scan")
        st.image(image, use_container_width=True)
    
    with st.spinner("Generating XAI Explanations..."):
        # 1. Grad CAM
        target_layer = model.get_last_conv_layer()
        cam_generator = GradCAM(model, target_layer)
        img_t.requires_grad = True # Required for hooks to trigger
        cam, _ = cam_generator(img_t, class_idx)
        
        # Prepare for overlay
        img_np = np.array(image.resize((224, 224))) / 255.0
        overlay = overlay_gradcam(img_np, cam)
        
        with col2:
            st.subheader("Grad-CAM Heatmap")
            st.image(overlay, use_container_width=True)
            
        # 2. SHAP
        # Better background: use "normalized black" instead of raw zeros
        # This prevents SHAP from being noisy in the black areas of the MRI
        m_mean = [0.485, 0.456, 0.406]
        m_std = [0.229, 0.224, 0.225]
        
        # Create a background of 10 samples (more samples = smoother results)
        bg_tensor = torch.zeros((10, 3, 224, 224)).to(device)
        for i in range(3):
            bg_tensor[:, i, :, :] = (0 - m_mean[i]) / m_std[i]
        
        try:
            shap_path = "results/shap_output_legacy.png"
            os.makedirs("results", exist_ok=True)
            generate_shap_explanation(
                model, bg_tensor, img_t, device, 
                save_path=shap_path,
                mean=m_mean,
                std=m_std
            )
            with col3:
                st.subheader("SHAP Attribution")
                st.image(shap_path, use_container_width=True)
        except Exception as e:
            with col3:
                st.error(f"SHAP failed: {str(e)}")

        st.markdown("---")
        st.subheader("💡 Counterfactual AI Analysis")
        cf_generator = ImageCounterfactual(model, device)
        cf_map = cf_generator.generate_counterfactual_map(img_t, 0)
        cf_overlay = overlay_counterfactual(img_np, cf_map)
        
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.image(cf_overlay, use_container_width=True)
        with c_col2:
            st.info("Visualizing the minimal changes needed in the MRI scan to shift the diagnosis to a healthier state.")


# End of MRI Analysis

