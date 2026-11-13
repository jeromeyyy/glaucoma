import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config
from preprocessing.transforms import apply_clahe_enhancement
from preprocessing.augmentation import get_training_augmentation, get_validation_augmentation

class FundusDataset(Dataset):
    """
    Standard PyTorch Dataset for Retinal Fundus Images.
    Supports folders arranged as:
        root_dir/
            Normal/ (or 0/)
            Glaucoma/ (or 1/)
    or split directories:
        root_dir/train/Normal/, root_dir/val/Glaucoma/, etc.
    """
    def __init__(self, samples, transform=None, apply_clahe=True):
        """
        Args:
            samples: list of tuples (image_path, label_int)
            transform: albumentations or torchvision transform
            apply_clahe: whether to enhance fundus contrast
        """
        self.samples = samples
        self.transform = transform
        self.apply_clahe = apply_clahe

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        img_np = np.array(image)
        
        if self.apply_clahe:
            img_np = apply_clahe_enhancement(img_np)

        if self.transform is not None:
            # Albumentations transform expects named argument image=...
            augmented = self.transform(image=img_np)
            tensor = augmented["image"]
        else:
            tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0

        return tensor, torch.tensor(label, dtype=torch.long), img_path

import numpy as np

def scan_dataset_directory(dir_path):
    """
    Scans a directory for Normal and Glaucoma images.
    Returns:
        samples: list of (path, label)
        counts: dict of category counts
    """
    if not os.path.exists(dir_path):
        return [], {"Normal": 0, "Glaucoma": 0, "status": "Directory not found"}

    normal_patterns = ["*normal*", "*Normal*", "*0*", "*non-glaucoma*"]
    glaucoma_patterns = ["*glaucoma*", "*Glaucoma*", "*1*"]

    samples = []
    normal_count = 0
    glaucoma_count = 0

    # Check for direct class subfolders
    for sub in os.listdir(dir_path):
        sub_lower = sub.lower()
        sub_full = os.path.join(dir_path, sub)
        if not os.path.isdir(sub_full): continue

        if any(p in sub_lower for p in ["negative", "normal", "healthy", "non_glaucoma", "non-glaucoma"]):
            label = 0
        elif any(p in sub_lower for p in ["positive", "glaucoma", "glaucomatous"]):
            label = 1
        else:
            continue

        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"):
            for f in glob.glob(os.path.join(sub_full, ext)):
                samples.append((f, label))
                if label == 1: glaucoma_count += 1
                else: normal_count += 1

    # If images were directly in split folders or loose files
    if len(samples) == 0:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in [".png", ".jpg", ".jpeg"]:
                    full_path = os.path.join(root, file)
                    path_lower = full_path.lower()
                    if "_g_acrima" in path_lower or "_g_" in path_lower:
                        samples.append((full_path, 1))
                        glaucoma_count += 1
                    elif "acrima" in path_lower and "_g_" not in path_lower:
                        samples.append((full_path, 0))
                        normal_count += 1
                    elif any(p in path_lower for p in ["negative", "normal", "healthy", "non_glaucoma", "non-glaucoma"]):
                        samples.append((full_path, 0))
                        normal_count += 1
                    elif any(p in path_lower for p in ["positive", "glaucoma", "glaucomatous"]):
                        samples.append((full_path, 1))
                        glaucoma_count += 1

    status = "Available" if (normal_count + glaucoma_count) > 0 else "Empty / Incompatible structure"
    return samples, {
        "Normal": normal_count,
        "Glaucoma": glaucoma_count,
        "Total": normal_count + glaucoma_count,
        "status": status
    }

def check_all_datasets():
    """
    Checks all configured dataset paths without crashing.
    Returns status report dictionary.
    """
    datasets_to_check = {
        "ODIR": Config.ODIR_DIR,
        "G1020": Config.G1020_DIR,
        "PAPILA": Config.PAPILA_DIR,
        "AIROGS": Config.AIROGS_DIR,
        "RIM-ONE DL": Config.RIMONE_DIR,
        "Synthetic Fundus": Config.SYNTHETIC_DIR
    }

    report = {}
    for name, path in datasets_to_check.items():
        _, meta = scan_dataset_directory(path)
        meta["path"] = path
        report[name] = meta

    return report

def get_dataloaders(
    dataset_name="synthetic",
    batch_size=8,
    image_size=(300, 300),
    num_workers=0
):
    """
    Creates train, val, and test DataLoaders for the selected dataset.
    Falls back gracefully to synthetic fundus if external dataset is missing.
    """
    dataset_dirs = {
        "synthetic": Config.SYNTHETIC_DIR,
        "odir": Config.ODIR_DIR,
        "g1020": Config.G1020_DIR,
        "papila": Config.PAPILA_DIR,
        "airogs": Config.AIROGS_DIR,
        "rimone": Config.RIMONE_DIR
    }
    
    target_dir = dataset_dirs.get(dataset_name.lower(), Config.SYNTHETIC_DIR)
    
    # Check if train/val/test subdirectories exist
    train_dir = os.path.join(target_dir, "train")
    val_dir = os.path.join(target_dir, "val")
    test_dir = os.path.join(target_dir, "test")
    
    train_samples, _ = scan_dataset_directory(train_dir)
    val_samples, _ = scan_dataset_directory(val_dir)
    test_samples, _ = scan_dataset_directory(test_dir)
    
    # If not split into train/val/test folders, split entire dataset 70/15/15
    if len(train_samples) == 0:
        all_samples, meta = scan_dataset_directory(target_dir)
        if len(all_samples) == 0:
            # Fall back to synthetic
            print(f"Dataset {dataset_name} empty or not found at {target_dir}. Falling back to synthetic fundus dataset.")
            from .sample_generator import create_synthetic_dataset
            create_synthetic_dataset(Config.SYNTHETIC_DIR)
            return get_dataloaders("synthetic", batch_size, image_size, num_workers)
            
        np.random.seed(Config.SEED)
        np.random.shuffle(all_samples)
        n = len(all_samples)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)
        train_samples = all_samples[:train_end]
        val_samples = all_samples[train_end:val_end]
        test_samples = all_samples[val_end:]

    train_tf = get_training_augmentation(image_size)
    val_tf = get_validation_augmentation(image_size)

    train_ds = FundusDataset(train_samples, transform=train_tf)
    val_ds = FundusDataset(val_samples, transform=val_tf)
    test_ds = FundusDataset(test_samples, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
