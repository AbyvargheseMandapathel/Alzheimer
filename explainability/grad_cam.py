import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling on gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        
        # Robust normalization
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
        else:
            # Fallback if no positive influence is found (prevent "null" image)
            cam = np.zeros_like(cam)
            
        # Resize to original image size (W, H)
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        
        return cam, class_idx


def overlay_gradcam(original_img_np, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays a Grad-CAM heatmap on an original image.
    original_img_np should be (H, W, 3) with values in range [0, 1].
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap = np.float32(heatmap) / 255.0
    # RGB to BGR logic normally handled here, but plt expects RGB
    heatmap = heatmap[:, :, ::-1] 
    
    cam_result = heatmap * alpha + original_img_np * (1 - alpha)
    cam_result = cam_result / np.max(cam_result)
    
    return np.uint8(255 * cam_result)
