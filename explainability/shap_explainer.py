import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def denormalize(x, mean, std):
    """
    Denormalize tensor for visualization
    """
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(x.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(x.device)
    return x * std + mean


def generate_shap_explanation(
    model,
    background_images,
    test_images,
    device="cuda",
    save_path="shap_explanation.png",
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
):
    """
    Generates and saves SHAP explanation image (fixed + improved)

    Args:
        model: PyTorch model
        background_images: Tensor (N, C, H, W)
        test_images: Tensor (N, C, H, W)
    """

    model.eval()
    model.to(device)

    # Move data to device
    background_images = background_images.to(device)
    test_images = test_images.to(device)

    # Create SHAP explainer
    # DeepExplainer is often better for PyTorch models with ReLUs
    # But GradientExplainer is a solid alternative
    explainer = shap.GradientExplainer(model, background_images)

    # Compute SHAP values for the top predicted class
    # GradientExplainer returns (shap_values, indexes) when ranked_outputs is used
    shap_results = explainer.shap_values(test_images, ranked_outputs=1)
    
    if isinstance(shap_results, tuple):
        shap_values = shap_results[0]
    else:
        shap_values = shap_results

    # ---- Convert SHAP values to numpy and transpose to (N, H, W, C) ----
    if isinstance(shap_values, list):
        shap_numpy = []
        for s in shap_values:
            if torch.is_tensor(s):
                s = s.detach().cpu().numpy()
            
            # Transpose from (N, C, H, W) to (N, H, W, C) or (N, C, H, W, 1) to (N, H, W, C)
            if len(s.shape) == 5: # (N, C, H, W, R)
                s = s[..., 0]
            
            s = np.transpose(s, (0, 2, 3, 1))
            
            # Normalize for better visualization contrast
            if np.max(np.abs(s)) > 0:
                s = s / (np.max(np.abs(s)) + 1e-8)
            shap_numpy.append(s)
    else:
        if torch.is_tensor(shap_values):
            shap_values = shap_values.detach().cpu().numpy()
        
        if len(shap_values.shape) == 5:
            shap_values = shap_values[..., 0]
            
        shap_values = np.transpose(shap_values, (0, 2, 3, 1))
        
        if np.max(np.abs(shap_values)) > 0:
            shap_values = shap_values / (np.max(np.abs(shap_values)) + 1e-8)
        shap_numpy = [shap_values]

    # ---- Denormalize images for visualization ----
    test_images_vis = denormalize(test_images.clone(), mean, std)
    test_numpy = np.transpose(test_images_vis.detach().cpu().numpy(), (0, 2, 3, 1))
    
    # Clip for safety
    test_numpy = np.clip(test_numpy, 0, 1)

    # ---- Handle grayscale properly ----
    if test_numpy.shape[-1] == 1:
        test_numpy = np.repeat(test_numpy, 3, axis=-1)

    # ---- Plot ----
    plt.clf()
    # Note: shap.image_plot expects a list of arrays for shap_values if plotting per class
    shap.image_plot(shap_numpy, test_numpy, show=False)

    # ---- Save ----
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close('all')

    return save_path