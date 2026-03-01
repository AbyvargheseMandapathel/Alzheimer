import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_shap_explanation(model, background_images, test_images, device="cuda", save_path="shap_explanation.png"):
    """
    Generates and saves a SHAP explanation image.
    Args:
        model: Standard Pytorch Model
        background_images: A batch of images (e.g. 50-100) used to represent the background distribution.
        test_images: The images to explain.
    """
    model.eval()
    
    # Use GradientExplainer
    explainer = shap.GradientExplainer(model, background_images)
    
    # Compute SHAP values
    # Note: GradientExplainer returns (shap_values, indexes) when ranked_outputs is used
    shap_results = explainer.shap_values(test_images, ranked_outputs=1)
    
    if isinstance(shap_results, tuple):
        shap_values = shap_results[0]
    else:
        shap_values = shap_results

    # Prepare data for SHAP image_plot
    # PyTorch is (N, C, H, W) but SHAP expects (N, H, W, C)
    test_numpy = np.transpose(test_images.detach().cpu().numpy(), (0, 2, 3, 1))
    
    # Ensure shap_values are numpy arrays and transposed
    if isinstance(shap_values, list):
        shap_numpy = []
        for s in shap_values:
            if torch.is_tensor(s):
                s = s.detach().cpu().numpy()
            shap_numpy.append(np.transpose(s, (0, 2, 3, 1)))
    else:
        if torch.is_tensor(shap_values):
            shap_values = shap_values.detach().cpu().numpy()
        
        # If GradientExplainer returned an array of shape (N, C, H, W, ranked_outputs)
        if len(shap_values.shape) == 5:
            # We take the first ranked output
            s = shap_values[..., 0]
            shap_numpy = [np.transpose(s, (0, 2, 3, 1))]
        else:
            shap_numpy = [np.transpose(shap_values, (0, 2, 3, 1))]
    
    # Plot and save
    plt.clf() # Clear current figure
    shap.image_plot(shap_numpy, test_numpy, show=False)
    
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')
    
    return save_path
