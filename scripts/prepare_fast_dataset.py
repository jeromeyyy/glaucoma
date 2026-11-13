import os
import glob
from PIL import Image
from tqdm import tqdm

src_base = os.path.join("data", "Fundus_Train_Val_Data", "Fundus_Scanes_Sorted")
dst_base = os.path.join("data", "Fundus_Resized_300", "Fundus_Scanes_Sorted")

splits = [
    ("Train", "Glaucoma_Positive"),
    ("Train", "Glaucoma_Negative"),
    ("Validation", "Glaucoma_Positive"),
    ("Validation", "Glaucoma_Negative")
]

total_processed = 0
for split, cls in splits:
    src_dir = os.path.join(src_base, split, cls)
    dst_dir = os.path.join(dst_base, split, cls)
    os.makedirs(dst_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(src_dir, "*.*"))
    print(f"Processing {split}/{cls}: {len(files)} images...")
    for f in files:
        fname = os.path.basename(f)
        out_path = os.path.join(dst_dir, fname)
        if not os.path.exists(out_path):
            try:
                with Image.open(f) as img:
                    img_rgb = img.convert("RGB")
                    resized = img_rgb.resize((300, 300), Image.BILINEAR)
                    resized.save(out_path, format="JPEG", quality=90)
                total_processed += 1
            except Exception as e:
                print(f"Error processing {f}: {e}")

print(f"Dataset preparation complete! Processed {total_processed} images into {dst_base}")

