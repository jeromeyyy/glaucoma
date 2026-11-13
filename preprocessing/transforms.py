import cv2
import numpy as np
import torch
from torchvision import transforms

def apply_clahe_enhancement(image_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Enhances fundus image contrast via CLAHE on L-channel (LAB)
    and selective green-channel sharpening.
    """
    if not isinstance(image_rgb, np.ndarray):
        image_rgb = np.array(image_rgb)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced

def get_inference_transform(image_size=(300, 300)):
    """
    Transforms for inference:
    Enhance -> Resize -> Normalize to ImageNet stats
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform

def denormalize_tensor(tensor):
    """
    Converts normalized torch tensor (C, H, W) back to numpy RGB image (H, W, C) [0, 255]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img
