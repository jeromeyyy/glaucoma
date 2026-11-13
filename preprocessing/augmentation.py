import albumentations as A 

from albumentations.pytorch import ToTensorV2
import numpy as np

def get_training_augmentation(image_size=(300, 300)):
    """
    Albumentations data augmentation pipeline tailored for fundus photography:
    - Flips (Horizontal, Vertical)
    - Shift, Scale, Rotate
    - Color Jitter (Brightness, Contrast)
    - Safe distortion for optical fundus variation
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_validation_augmentation(image_size=(300, 300)):
    """
    Validation pipeline: deterministic resizing and normalization only.
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
