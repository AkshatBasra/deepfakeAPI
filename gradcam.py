import numpy as np
import torch
import cv2
import config

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Traverse the model to find the target layer
        target_layer = self._get_layer(model, target_layer_name.split('.'))
        
        if target_layer is None:
            raise ValueError(f"Target layer {target_layer_name} not found in model.")
            
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def _get_layer(self, module, name_parts):
        if not name_parts:
            return module
        if hasattr(module, name_parts[0]):
            return self._get_layer(getattr(module, name_parts[0]), name_parts[1:])
        return None

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        # Forward pass requires gradients for input
        input_tensor.requires_grad = True
        
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # For binary classification with a single logit, backprop on the sum (safe for (1,) tensors)
        output.sum().backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            return None
            
        # Get activations and gradients for the first item in batch
        gradients = self.gradients.cpu().data.numpy()[0] # (C, H, W)
        activations = self.activations.cpu().data.numpy()[0] # (C, H, W)
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2)) # (C,)
        
        # Weight the channels
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32) # (H, W)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        # ReLU (only consider positive influences) and normalize
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
            
        return heatmap

def make_gradcam_heatmap(img_tensor: torch.Tensor, model: torch.nn.Module, target_layer_name: str):
    """
    Generates a Grad-CAM heatmap for a given input tensor and model.
    """
    cam = GradCAM(model, target_layer_name)
    heatmap = cam.generate(img_tensor)
    return heatmap

def generate_heatmap_overlay(original_img: np.ndarray, heatmap: np.ndarray, alpha=0.4):
    """
    Overlays the heatmap on the original image.
    """
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Resize heatmap to match original image size
    jet = cv2.resize(jet, (original_img.shape[1], original_img.shape[0]))

    # Superimpose the heatmap on original image
    superimposed_img = jet * alpha + original_img
    
    # Clip and convert back to uint8
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img
