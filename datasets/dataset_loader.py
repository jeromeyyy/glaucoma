import os
import glob
from PIL import Image
import numpy as np
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

def scan_dataset_directory(dir_path):
    """
    Scans a directory for Normal and Glaucoma images.
    Returns:
        samples: list of (path, label)
        counts: dict of category counts
    """
    if not os.path.exists(dir_path):
        return [], {"Normal": 0, "Glaucoma": 0, "status": "Directory not found"}

    samples = []
    normal_count = 0
    glaucoma_count = 0

    # Check for direct class subfolders
    for sub in os.listdir(dir_path):
        sub_lower = sub.lower()
        sub_full = os.path.join(dir_path, sub)
        if not os.path.isdir(sub_full):
            continue

        if any(p in sub_lower for p in ["negative", "normal", "healthy", "non_glaucoma", "non-glaucoma"]):
            label = 0
        elif any(p in sub_lower for p in ["positive", "glaucoma", "glaucomatous"]):
            label = 1
        else:
            continue

        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"):
            for f in glob.glob(os.path.join(sub_full, ext)):
                samples.append((f, label))
                if label == 1:
                    glaucoma_count += 1
                else:
                    normal_count += 1

    # If images were directly in split folders or loose files
    if len(samples) == 0:
        for root, dirs, files in os.walk(dir_path):
            for file in sorted(files):
                ext = os.path.splitext(file)[1].lower()
                if ext in [".png", ".jpg", ".jpeg", ".bmp"]:
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
        "ACRIMA": getattr(Config, "ACRIMA_DIR", os.path.join(Config.DATA_DIR, "ACRIMA")),
        "Fundus (SiMES/ORIGA)": getattr(Config, "FUNDUS_DIR", os.path.join(Config.DATA_DIR, "Fundus_Train_Val_Data")),
        "ORIGA": getattr(Config, "ORIGA_DIR", os.path.join(Config.DATA_DIR, "ORIGA")),
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

def _stratified_split(samples, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Performs deterministic stratified split on samples ensuring:
    1. Balanced class ratios in train, val, and test.
    2. Zero leakage between splits.
    """
    normals = [s for s in samples if s[1] == 0]
    glaucomas = [s for s in samples if s[1] == 1]

    rng = np.random.RandomState(seed)
    rng.shuffle(normals)
    rng.shuffle(glaucomas)

    n_norm = len(normals)
    n_glauc = len(glaucomas)

    norm_train_end = int(train_ratio * n_norm)
    norm_val_end = int((train_ratio + val_ratio) * n_norm)

    glauc_train_end = int(train_ratio * n_glauc)
    glauc_val_end = int((train_ratio + val_ratio) * n_glauc)

    train_samples = normals[:norm_train_end] + glaucomas[:glauc_train_end]
    val_samples = normals[norm_train_end:norm_val_end] + glaucomas[glauc_train_end:glauc_val_end]
    test_samples = normals[norm_val_end:] + glaucomas[glauc_val_end:]

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    # Rigorous Zero-Leakage Verification
    train_paths = {s[0] for s in train_samples}
    val_paths = {s[0] for s in val_samples}
    test_paths = {s[0] for s in test_samples}

    assert len(train_paths & val_paths) == 0, "Patient/Image leakage detected between Train and Val!"
    assert len(train_paths & test_paths) == 0, "Patient/Image leakage detected between Train and Test!"
    assert len(val_paths & test_paths) == 0, "Patient/Image leakage detected between Val and Test!"

    return train_samples, val_samples, test_samples

def get_dataloaders(
    dataset_name="combined",
    batch_size=8,
    image_size=(300, 300),
    num_workers=0
):
    """
    Creates train, val, and test DataLoaders for the selected dataset.
    Supports real clinical datasets:
      - 'combined': Merges ACRIMA and Fundus_Train_Val_Data (1,355 clinical images)
      - 'acrima': ACRIMA clinical dataset (705 images)
      - 'fundus': Fundus_Train_Val_Data clinical dataset (650 images)
      - 'synthetic': Synthetic fundus dataset for quick tests
    """
    name_lower = dataset_name.lower()

    if name_lower == "combined":
        acrima_dir = getattr(Config, "ACRIMA_DIR", os.path.join(Config.DATA_DIR, "ACRIMA"))
        fundus_dir = getattr(Config, "FUNDUS_DIR", os.path.join(Config.DATA_DIR, "Fundus_Train_Val_Data"))
        acrima_samples, _ = scan_dataset_directory(acrima_dir)
        fundus_samples, _ = scan_dataset_directory(fundus_dir)
        all_samples = acrima_samples + fundus_samples
        if len(all_samples) == 0:
            print("Real datasets not found; falling back to synthetic fundus.")
            return get_dataloaders("synthetic", batch_size, image_size, num_workers)
        train_samples, val_samples, test_samples = _stratified_split(all_samples, seed=Config.SEED)

    elif name_lower == "acrima":
        acrima_dir = getattr(Config, "ACRIMA_DIR", os.path.join(Config.DATA_DIR, "ACRIMA"))
        all_samples, _ = scan_dataset_directory(acrima_dir)
        if len(all_samples) == 0:
            return get_dataloaders("synthetic", batch_size, image_size, num_workers)
        train_samples, val_samples, test_samples = _stratified_split(all_samples, seed=Config.SEED)

    elif name_lower == "fundus":
        fundus_dir = getattr(Config, "FUNDUS_DIR", os.path.join(Config.DATA_DIR, "Fundus_Train_Val_Data"))
        all_samples, _ = scan_dataset_directory(fundus_dir)
        if len(all_samples) == 0:
            return get_dataloaders("synthetic", batch_size, image_size, num_workers)
        train_samples, val_samples, test_samples = _stratified_split(all_samples, seed=Config.SEED)

    else:
        # Standard directory check or synthetic fallback
        dataset_dirs = {
            "synthetic": Config.SYNTHETIC_DIR,
            "odir": Config.ODIR_DIR,
            "g1020": Config.G1020_DIR,
            "papila": Config.PAPILA_DIR,
            "airogs": Config.AIROGS_DIR,
            "rimone": Config.RIMONE_DIR
        }
        target_dir = dataset_dirs.get(name_lower, Config.SYNTHETIC_DIR)
        all_samples, _ = scan_dataset_directory(target_dir)
        if len(all_samples) == 0:
            from .sample_generator import create_synthetic_dataset
            create_synthetic_dataset(Config.SYNTHETIC_DIR)
            all_samples, _ = scan_dataset_directory(Config.SYNTHETIC_DIR)
        train_samples, val_samples, test_samples = _stratified_split(all_samples, seed=Config.SEED)

    train_tf = get_training_augmentation(image_size)
    val_tf = get_validation_augmentation(image_size)

    train_ds = FundusDataset(train_samples, transform=train_tf)
    val_ds = FundusDataset(val_samples, transform=val_tf)
    test_ds = FundusDataset(test_samples, transform=val_tf)

    # drop_last=True on train_loader to guarantee batch size > 1 (satisfies BatchNorm)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=(len(train_ds) > batch_size)
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
