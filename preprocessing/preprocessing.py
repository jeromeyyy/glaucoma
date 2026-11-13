import os
import io
import cv2
import numpy as np
from PIL import Image
import torch

from .transforms import apply_clahe_enhancement, get_inference_transform

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

def validate_image_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file does not exist: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported image format '{ext}'. Allowed formats: {ALLOWED_EXTENSIONS}")
    return True

def load_image(image_input):
    """
    Loads image from file path, bytes, or file-like object and returns RGB numpy array.
    """
    if isinstance(image_input, str):
        validate_image_path(image_input)
        img = Image.open(image_input).convert('RGB')
        return np.array(img)
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert('RGB')
        return np.array(img)
    elif hasattr(image_input, 'read'):
        data = image_input.read()
        if hasattr(image_input, 'seek'):
            image_input.seek(0)
        img = Image.open(io.BytesIO(data)).convert('RGB')
        return np.array(img)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        return image_input
    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert('RGB'))
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

def preprocess_for_inference(image_input, image_size=(300, 300), apply_enhancement=True):
    """
    Loads, enhances, resizes and formats image for model inference.
    Returns:
        tensor: torch.Tensor [1, 3, H, W] normalized
        enhanced_rgb: np.ndarray [H, W, 3] in [0, 255] for visualizers
    """
    raw_rgb = load_image(image_input)
    if apply_enhancement:
        enhanced_rgb = apply_clahe_enhancement(raw_rgb)
    else:
        enhanced_rgb = raw_rgb

    resized_rgb = cv2.resize(enhanced_rgb, (image_size[1], image_size[0]))

    transform = get_inference_transform(image_size)
    tensor = transform(enhanced_rgb).unsqueeze(0)
    return tensor, resized_rgb
