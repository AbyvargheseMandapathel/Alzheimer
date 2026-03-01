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

background = torch.zeros((5, 3, 224, 224)).to(device)
img_t = torch.rand((1, 3, 224, 224)).to(device)

print("Starting GradientExplainer...")

explainer = shap.GradientExplainer(model, background)
shap_results = explainer.shap_values(img_t, ranked_outputs=1)

print("shap_results type:", type(shap_results))

if isinstance(shap_results, tuple):
    print("shap_results is a tuple of length", len(shap_results))
    shap_values = shap_results[0]
    print("tuple[0] type:", type(shap_values))
    if isinstance(shap_values, list):
        print("shap_values list length:", len(shap_values))
        print("shap_values list[0] shape:", shap_values[0].shape)
    elif hasattr(shap_values, 'shape'):
        print("shap_values shape:", shap_values.shape)
else:
    shap_values = shap_results
    if isinstance(shap_values, list):
        print("shap_values list length:", len(shap_values))
        print("shap_values list[0] shape:", shap_values[0].shape)
    elif hasattr(shap_values, 'shape'):
        print("shap_values shape:", shap_values.shape)

try:
    generate_shap_explanation(model, background, img_t, device, save_path="test_shap_out.png")
    print("generate_shap_explanation ran successfully")
except Exception as e:
    print("Error in generate_shap_explanation:", e)
