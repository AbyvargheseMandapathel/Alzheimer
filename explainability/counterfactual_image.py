import torch
import numpy as np
import cv2

class ImageCounterfactual:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def generate_counterfactual_map(self, input_tensor, target_class_idx):
        """
        Generates a saliency map showing which regions most strongly oppose 
        the target class (or support the current class if target is different).
        
        This is a 'Counterfactual Map' because it shows where the image would 
        need to change to become the target class.
        """
        self.model.eval()
        input_tensor = input_tensor.clone().detach().to(self.device)
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        
        # We want to see how to INCREASE the probability of the TARGET class
        # (e.g. Non-Demented)
        target_score = output[0, target_class_idx]
        
        self.model.zero_grad()
        target_score.backward()

        # The gradient shows how to increase the target score.
        # Regions with high positive gradient are 'responsible' for the gap.
        gradients = input_tensor.grad.data.cpu().numpy()[0]
        
        # Take the absolute sum across channels (or just the positive part)
        # Positive gradients => these pixels can increase the target class score if increased.
        # Negative gradients => these pixels can increase the target class score if decreased.
        # For a counterfactual map, we usually look at the magnitude or the direction.
        
        # We'll use the absolute mean across RGB channels to show relevant regions.
        cf_map = np.mean(np.abs(gradients), axis=0)
        
        # Normalize
        cf_map = (cf_map - cf_map.min()) / (cf_map.max() - cf_map.min() + 1e-8)
        
        # Resize back to original
        cf_map = cv2.resize(cf_map, (input_tensor.shape[3], input_tensor.shape[2]))
        
        return cf_map

def overlay_counterfactual(original_img_np, cf_map, alpha=0.5):
    """
    Overlays the counterfactual heatmap on the original image.
    Uses a different colormap (e.g., GREEN/VIRIDIS) to distinguish from Grad-CAM.
    """
    # Use COLORMAP_VIRIDIS or COLORMAP_SUMMER for a 'constructive' feel
    heatmap = cv2.applyColorMap(np.uint8(255 * cf_map), cv2.COLORMAP_VIRIDIS)
    heatmap = np.float32(heatmap) / 255.0
    heatmap = heatmap[:, :, ::-1] # BGR to RGB
    
    result = heatmap * alpha + original_img_np * (1 - alpha)
    result = np.clip(result, 0, 1)
    
    return np.uint8(255 * result)
