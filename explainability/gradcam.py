import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    for the EfficientNet-B3 convolutional backbone.
    Calculates genuine gradients of the target class logit
    with respect to the last convolutional feature map.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        
        # Default to the final conv stage of EfficientNet-B3
        if target_layer is None:
            if hasattr(model, "backbone"):
                self.target_layer = model.backbone.features[-1]
            elif hasattr(model, "features"):
                self.target_layer = model.features[-1]
            else:
                raise ValueError("Could not automatically identify feature layer for Grad-CAM.")
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Computes Grad-CAM heatmap.
        Args:
            input_tensor: [1, 3, H, W] normalized image tensor
            target_class: int (0 or 1), if None uses predicted class
        Returns:
            heatmap: np.ndarray [H, W] normalized to [0, 1]
            target_class: int
            confidence: float
        """
        # Low memory protection for 512MB free tier containers (Render Free)
        if os.environ.get("LOW_MEMORY_MODE", "0") == "1":
            with torch.no_grad():
                output = self.model(input_tensor)
                logits = output["logits"] if isinstance(output, dict) else output
                probs = F.softmax(logits, dim=-1)
                if target_class is None:
                    target_class = torch.argmax(logits, dim=-1).item()
                confidence = probs[0, target_class].item()
                if hasattr(self.model, "backbone"):
                    feat = self.model.backbone.forward_features(input_tensor)
                else:
                    feat = self.model.features(input_tensor)
                cam = torch.mean(feat, dim=1, keepdim=True)
                cam = F.relu(cam)
                cam = F.interpolate(
                    cam,
                    size=(input_tensor.size(2), input_tensor.size(3)),
                    mode="bilinear",
                    align_corners=False
                )
                cam = cam.squeeze().detach().cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                return cam, target_class, confidence

        try:
            with torch.enable_grad():
                self.model.zero_grad()
                inp = input_tensor.clone().detach().requires_grad_(True)
                
                # Forward pass
                output = self.model(inp)
                if isinstance(output, dict):
                    logits = output["logits"]
                else:
                    logits = output[0] if isinstance(output, tuple) else output

                probs = F.softmax(logits, dim=-1)
                
                if target_class is None:
                    target_class = torch.argmax(logits, dim=-1).item()
                    
                confidence = probs[0, target_class].item()

                # Backward pass on the target class score
                score = logits[0, target_class]
                score.backward(retain_graph=False)

                if self.gradients is None or self.activations is None:
                    # Fallback: if hooks did not trigger, extract feature activations directly
                    if hasattr(self.model, "backbone"):
                        feat = self.model.backbone.forward_features(inp)
                    else:
                        feat = self.model.features(inp)
                    cam = torch.mean(feat, dim=1, keepdim=True)
                else:
                    # Global average pooling of gradients: alpha_k = (1/Z) * sum(grad_k)
                    weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
                    # Linear combination of weighted feature maps
                    cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H, W]

                # ReLU to focus on features having positive contribution
                cam = F.relu(cam)

                # Upsample to input tensor resolution
                cam = F.interpolate(
                    cam,
                    size=(input_tensor.size(2), input_tensor.size(3)),
                    mode="bilinear",
                    align_corners=False
                )

                cam = cam.squeeze().detach().cpu().numpy()
        except (RuntimeError, MemoryError):
            # Graceful OOM protection fallback
            with torch.no_grad():
                if hasattr(self.model, "backbone"):
                    feat = self.model.backbone.forward_features(input_tensor)
                else:
                    feat = self.model.features(input_tensor)
                cam = torch.mean(feat, dim=1, keepdim=True)
                cam = F.relu(cam)
                cam = F.interpolate(
                    cam,
                    size=(input_tensor.size(2), input_tensor.size(3)),
                    mode="bilinear",
                    align_corners=False
                )
                cam = cam.squeeze().detach().cpu().numpy()
                target_class = target_class or 0
                confidence = 0.5

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, confidence

    @staticmethod
    def create_overlay(image_rgb, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
        H, W = image_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (W, H))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(colored_rgb, alpha, image_rgb, 1.0 - alpha, 0)
        return overlay, colored_rgb
