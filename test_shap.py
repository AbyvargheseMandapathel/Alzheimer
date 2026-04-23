import torch
import numpy as np
import shap
from models.cnn_model import AlzheimerResNet
from explainability.shap_explainer import generate_shap_explanation
import sys

# Setup dummy model and data
device = "cpu"
model = AlzheimerResNet(num_classes=3).to(device)
model.eval()

# Setup dummy data with proper normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Normalized black background
background = torch.zeros((5, 3, 224, 224)).to(device)
for i in range(3):
    background[:, i, :, :] = (0 - mean[i]) / std[i]

img_t = torch.rand((1, 3, 224, 224)).to(device)

print("Starting GradientExplainer...")

explainer = shap.GradientExplainer(model, background)
shap_results = explainer.shap_values(img_t, ranked_outputs=1)

print("shap_results type:", type(shap_results))

try:
    generate_shap_explanation(
        model, background, img_t, device, 
        save_path="test_shap_out.png",
        mean=mean,
        std=std
    )
    print("generate_shap_explanation ran successfully")
except Exception as e:
    print("Error in generate_shap_explanation:", e)
    import traceback
    traceback.print_exc()
