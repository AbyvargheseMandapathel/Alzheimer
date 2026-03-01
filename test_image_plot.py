import torch
import numpy as np
import shap
from models.cnn_model import AlzheimerResNet
import matplotlib.pyplot as plt

device = "cpu"
model = AlzheimerResNet(num_classes=3).to(device)
model.eval()

background = torch.zeros((5, 3, 224, 224)).to(device)
img_t = torch.rand((1, 3, 224, 224)).to(device)

explainer = shap.GradientExplainer(model, background)
shap_results = explainer.shap_values(img_t, ranked_outputs=1)

shap_values = shap_results[0]
indexes = shap_results[1]

print("Original shap_values shape:", shap_values.shape)

# Test formatting to pass to image_plot
# shap_values is (N, C, H, W, ranked_outputs) = (1, 3, 224, 224, 1)
# shap_values[..., 0] will be (1, 3, 224, 224)
# Then transpose to (1, 224, 224, 3) 
# shap.image_plot takes a list of arrays for multiple classes, or a single array

shap_numpy = np.transpose(shap_values[..., 0], (0, 2, 3, 1))
test_numpy = np.transpose(img_t.detach().cpu().numpy(), (0, 2, 3, 1))

print("Formatted shap_numpy shape:", shap_numpy.shape)
print("Formatted test_numpy shape:", test_numpy.shape)

try:
    # `shap.image_plot` may want a list of shapes if we want it to work as if it's 1 class
    shap.image_plot([shap_numpy], test_numpy, show=False)
    plt.savefig("test_image_plot.png")
    print("SUCCESS: Image plot generated!")
except Exception as e:
    print("Error during image_plot:", e)

