import torch
import numpy as np
import cv2

class ImageCounterfactual:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def generate_counterfactual_map(self, input_tensor, target_class_idx, iterations=50, lr=0.01):
        """
        Generates an optimized counterfactual map.
        It iteratively searches for the minimal pixel changes required to move 
        the image towards the target class prediction.
        """
        self.model.eval()
        # Create a trainable perturbation tensor starting at zero
        perturbation = torch.zeros_like(input_tensor, requires_grad=True, device=self.device)
        
        optimizer = torch.optim.Adam([perturbation], lr=lr)
        
        for _ in range(iterations):
            # Apply perturbation to original image
            # We use tanh or clipping to ensure it stays in a valid (normalized) range
            adv_input = input_tensor + perturbation
            
            # Forward pass
            outputs = self.model(adv_input)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Loss 1: Maximize target class probability
            target_loss = -torch.log(probs[0, target_class_idx] + 1e-10)
            
            # Loss 2: Minimize perturbation size (L1 for sparsity)
            l1_loss = torch.norm(perturbation, p=1) * 0.1
            
            total_loss = target_loss + l1_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Optional: constraint to keep perturbation small
            with torch.no_grad():
                perturbation.clamp_(-0.5, 0.5)

        # The 'Counterfactual Map' is the absolute difference 
        # (normalized to show where changes occurred)
        diff = torch.abs(perturbation).detach().cpu().numpy()[0]
        cf_map = np.mean(diff, axis=0) # Mean across channels
        
        # Robust Normalization
        # Use 95th percentile for max to avoid being dominated by single outlier pixels
        vmax = np.percentile(cf_map, 99)
        if vmax > 0:
            cf_map = np.clip(cf_map / (vmax + 1e-8), 0, 1)
        
        # Resize back to original
        cf_map = cv2.resize(cf_map, (input_tensor.shape[3], input_tensor.shape[2]))
        
        return cf_map

def overlay_counterfactual(original_img_np, cf_map, alpha=0.6):
    """
    Overlays the counterfactual heatmap on the original image.
    Uses VIRIDIS (Green/Yellow) to represent 'Constructive Change'.
    """
    # Smooth the map slightly for better visualization
    cf_map = cv2.GaussianBlur(cf_map, (5, 5), 0)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cf_map), cv2.COLORMAP_VIRIDIS)
    heatmap = np.float32(heatmap) / 255.0
    heatmap = heatmap[:, :, ::-1] # BGR to RGB
    
    # Use additive blend for better visibility on dark MRI backgrounds
    # This highlights the changes like a neon overlay
    result = heatmap * alpha + original_img_np * (1.0)
    result = np.clip(result, 0, 1)
    
    return np.uint8(255 * result)
