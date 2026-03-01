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
from models.cnn_model import AlzheimerResNet
import io
import os
try:
    from fpdf import FPDF
    HAVE_FPDF = True
except ImportError:
    HAVE_FPDF = False

# Class map
CLASS_NAMES = {0: "Non Demented", 1: "Class 1", 2: "Class 2"}

st.set_page_config(page_title="Alzheimer's Disease Detection", layout="wide")

@st.cache_resource
def load_model(model_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Map selection to file path
    path = "weights/best_centralized_model.pth" if model_type == "Centralized" else "weights/best_federated_model.pth"
    
    model = AlzheimerResNet(num_classes=3).to(device)
    
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        st.sidebar.success(f"✅ Loaded {model_type} Model weights.")
    else:
        st.sidebar.warning(f"⚠️ {model_type} weights not found at `{path}`. Using untrained model!")
        
    model.eval()
    return model, device

st.title("🧠 Alzheimer's Disease Detection from MRI")

# Sidebar selection
st.sidebar.title("Configuration")
model_selection = st.sidebar.selectbox("Choose Trained Model", ["Centralized", "Federated"])
model, device = load_model(model_selection)

uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the file
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original Scan")
        st.image(image, width='stretch')
        
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
    
    # Progress bars for probabilities
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
