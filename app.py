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
from models.cnn_model import AlzheimerResNet
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

st.set_page_config(page_title="Alzheimer's Disease Detection", layout="wide")

@st.cache_resource
def load_model(model_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Map selection to file path
    path = "weights/best_centralized_model.pth" if model_type == "Centralized" else "weights/best_federated_model.pth"
    
    model = AlzheimerResNet(num_classes=3).to(device)
    
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            st.sidebar.success(f"✅ Loaded {model_type} Model weights.")
        except RuntimeError as e:
            st.sidebar.error(f"⚠️ Architecture Mismatch: Could not load weights from `{path}`.")
            st.sidebar.info("This App uses **ResNet50**, but the weights file appears to be from a different architecture (likely ResNet18).")
            st.sidebar.warning("Prediction results will be inaccurate until you retrain the model with `python mri_train.py`.")
    else:
        st.sidebar.warning(f"⚠️ {model_type} weights not found at `{path}`. Using untrained model!")

        
    model.eval()
    return model, device

@st.cache_resource
def load_tabular_model():
    model_path = "results/global_model.joblib"
    # Use the processed dataset that matches the 53-feature model
    data_path = "Datasets/Multimodal Dataset/phase1_processed.csv"
    
    if os.path.exists(model_path) and os.path.exists(data_path):
        # Load the raw sklearn model for DiCE
        rf_model = joblib.load(model_path)
        df = pd.read_csv(data_path)
        explainer = CounterfactualExplainer(rf_model, df)
        return rf_model, df, explainer
    return None, None, None

@st.cache_data
def load_performance_metrics():
    # Prioritize the updated MRI metrics
    mri_metrics_path = "results/mri_metrics.csv"
    legacy_metrics_path = "results/training_metrics.csv"
    
    path = mri_metrics_path if os.path.exists(mri_metrics_path) else legacy_metrics_path
    
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Convert to percentage for display where appropriate
        df['accuracy_pct'] = df['accuracy'] * 100
        return df
    return None


# Global Data Loading
metrics_df = load_performance_metrics()

st.title("🧠 Alzheimer's Disease Diagnostic Center")
st.markdown("### Advanced Explainable AI for Early Detection")

# Sidebar selection
st.sidebar.title("Configuration")
model_selection = "Centralized" 
model, device = load_model(model_selection)



# Display MRI Model Performance in Sidebar
st.sidebar.divider()
st.sidebar.subheader("📈 MRI Model Reliability")


if metrics_df is not None:
    # Get the latest stats from the last round
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
    # Read the file
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original Scan")
        # st.image(image, width='stretch')
        st.image(image, use_container_width=True)
        
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_t = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[class_idx].item() * 100
        
    st.markdown("---")
    st.markdown(f"### 🩺 **Diagnosis Prediction:** {CLASS_NAMES[class_idx]} ({confidence:.2f}% confidence)")
    
    # Show Global Metrics for context (Prominent display for each input)
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
    
    st.markdown("---")

    for i, class_name in CLASS_NAMES.items():
        st.progress(probabilities[i].item(), text=f"{class_name}: {probabilities[i].item()*100:.2f}%")
        
    st.markdown("---")
    
    # Extract Interpretability
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
            st.image(overlay, width='stretch', caption="Red indicates high focus regions")
            
        # 2. SHAP
        # Since SHAP deep explainer needs a background, we just use a small zero baseline
        # (In a real scenario, use actual dataset baseline, but this is a Streamlit demo hack)
        background = torch.zeros((5, 3, 224, 224)).to(device)
        try:
            shap_path = "shap_output.png"
            generate_shap_explanation(model, background, img_t, device, save_path=shap_path)
            with col3:
                st.subheader("SHAP Feature Attribution")
                st.image(shap_path, width='stretch', caption="Red = Supports, Blue = Opposes diagnosis")
        except Exception as e:
            with col3:
                st.error(f"SHAP Explanation failed: {str(e)}")
                st.info("Ensure the `shap` library is installed: `pip install shap`")

        # 3. Counterfactual Image Analysis
        st.markdown("---")
        st.subheader("💡 Counterfactual AI: What should change for a Healthy diagnosis?")
        
        # We want to see how to increase probability of 'Non Demented' (index 0)
        target_cf_class = 0 
        cf_generator = ImageCounterfactual(model, device)
        cf_map = cf_generator.generate_counterfactual_map(img_t, target_cf_class)
        cf_overlay = overlay_counterfactual(img_np, cf_map)
        
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.image(cf_overlay, use_container_width=True, caption="Counterfactual Heatmap (Green/Yellow indicates target areas for improvement)")
        with c_col2:
            st.info("""
            **Counterfactual Insight:**
            The highlighted regions above indicate the specific morphological areas of the brain that the AI identifies as most critical for a 'Non Demented' classification. 
            Reducing volume loss or intensity variations in these areas would shift the diagnosis towards a healthier state.
            """)

    # PDF Generator
    if HAVE_FPDF:
        with st.spinner("Preparing PDF Report..."):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("helvetica", "B", 16)
            pdf.cell(0, 10, "Alzheimer's Disease MRI Diagnostics Report", align="C", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "", 12)
            pdf.cell(0, 10, f"Predicted Diagnosis: {CLASS_NAMES[class_idx]}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 10, f"Confidence: {confidence:.2f}%", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 10, "Disclaimer: AI interpretation only. Not for clinical use.", new_x="LMARGIN", new_y="NEXT")
            
            # Write PDF to memory buffer
            pdf_bytes = pdf.output()
            
            st.download_button(
                label="📥 Download Diagnostic Report (PDF)",
                data=bytes(pdf_bytes),
                file_name="diagnosis_report.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("Install `fpdf` (`pip install fpdf`) to enable PDF report generation.")
# End of MRI Analysis

