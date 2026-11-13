import os
import cv2
import numpy as np

def generate_synthetic_fundus_image(is_glaucoma=False, size=(300, 300)):
    """
    Generates a realistic synthetic retinal fundus image for testing
    when real external datasets (ODIR, G1020, PAPILA, etc.) are not locally present.
    Simulates:
    - Retinal background (red-orange gradient)
    - Optic disc (yellowish-orange oval)
    - Optic cup (paler yellow inner circle, enlarged CDR if glaucoma)
    - Retinal vasculature (branching temporal and nasal arcades)
    - Macula / fovea (darker temporal region)
    """
    H, W = size
    # 1. Retinal background (circular field of view)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    center = (W // 2, H // 2)
    radius = int(min(H, W) * 0.46)
    
    # Gradient radial mask
    y, x = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = dist_from_center <= radius
    
    # Orange-red retinal color with subtle shading
    bg_r = np.clip(180 - dist_from_center * 0.25, 40, 210).astype(np.uint8)
    bg_g = np.clip(70 - dist_from_center * 0.15, 10, 90).astype(np.uint8)
    bg_b = np.clip(25 - dist_from_center * 0.08, 5, 40).astype(np.uint8)
    
    img[:, :, 0] = np.where(mask, bg_r, 0)
    img[:, :, 1] = np.where(mask, bg_g, 0)
    img[:, :, 2] = np.where(mask, bg_b, 0)
    
    # 2. Optic Disc (typically nasal or right/left side)
    disc_center = (int(W * 0.35), int(H * 0.5))
    disc_radius = int(min(H, W) * 0.13)
    cv2.circle(img, disc_center, disc_radius, (235, 175, 75), -1)  # Warm orange-yellow disc
    
    # 3. Optic Cup (central excavation)
    # Glaucoma: enlarged CDR (0.65 - 0.85) vs Normal CDR (0.3 - 0.45)
    cdr = np.random.uniform(0.68, 0.84) if is_glaucoma else np.random.uniform(0.32, 0.44)
    cup_radius = int(disc_radius * cdr)
    cup_color = (250, 220, 130) if is_glaucoma else (240, 195, 100)
    cv2.circle(img, disc_center, cup_radius, cup_color, -1)
    
    # 4. Retinal Blood Vessels (superior and inferior branching arcades)
    vessel_color = (120, 20, 10)  # Dark red
    # Superior arcade
    pts_sup = np.array([
        [disc_center[0], disc_center[1]],
        [int(disc_center[0] + W * 0.1), int(disc_center[1] - H * 0.2)],
        [int(disc_center[0] + W * 0.25), int(disc_center[1] - H * 0.35)],
        [int(disc_center[0] + W * 0.4), int(disc_center[1] - H * 0.38)]
    ], np.int32)
    cv2.polylines(img, [pts_sup], False, vessel_color, thickness=3 if not is_glaucoma else 2)
    
    # Inferior arcade
    pts_inf = np.array([
        [disc_center[0], disc_center[1]],
        [int(disc_center[0] + W * 0.1), int(disc_center[1] + H * 0.2)],
        [int(disc_center[0] + W * 0.25), int(disc_center[1] + H * 0.35)],
        [int(disc_center[0] + W * 0.4), int(disc_center[1] + H * 0.38)]
    ], np.int32)
    cv2.polylines(img, [pts_inf], False, vessel_color, thickness=3 if not is_glaucoma else 2)
    
    # Minor vessel branches
    cv2.line(img, (disc_center[0], disc_center[1]), (int(disc_center[0] - W * 0.15), int(disc_center[1] - H * 0.1)), vessel_color, 2)
    cv2.line(img, (disc_center[0], disc_center[1]), (int(disc_center[0] - W * 0.15), int(disc_center[1] + H * 0.1)), vessel_color, 2)
    
    # 5. Macula & Fovea (dark avascular zone temporal to the optic disc)
    macula_center = (int(W * 0.65), int(H * 0.5))
    cv2.circle(img, macula_center, int(min(H, W) * 0.08), (140, 50, 15), -1)
    cv2.circle(img, macula_center, int(min(H, W) * 0.02), (90, 30, 10), -1)
    
    # Subtle Gaussian blur for optical realism
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img

def create_synthetic_dataset(base_dir="data/synthetic_fundus", num_train_per_class=20, num_val_per_class=5):
    """
    Generates a structured synthetic dataset for training and verification.
    """
    splits = {
        "train": num_train_per_class,
        "val": num_val_per_class,
        "test": num_val_per_class
    }
    
    created_counts = {}
    for split, count in splits.items():
        for label, is_glaucoma in [("Normal", False), ("Glaucoma", True)]:
            folder = os.path.join(base_dir, split, label)
            os.makedirs(folder, exist_ok=True)
            for idx in range(count):
                img = generate_synthetic_fundus_image(is_glaucoma=is_glaucoma)
                filename = f"{label.lower()}_{idx:03d}.png"
                filepath = os.path.join(folder, filename)
                cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            created_counts[f"{split}/{label}"] = count
            
    print(f"Synthetic dataset created at {base_dir}: {created_counts}")
    return created_counts

if __name__ == "__main__":
    create_synthetic_dataset()
