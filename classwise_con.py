import os
import shutil
from collections import Counter

# Source paths
image_dir = "./VisDrone2019-DET-val/images"
label_dir = "./VisDrone2019-DET-val/labels"

# Output root folders
image_output_root = "./VisDrone_by_class_val/images"
label_output_root = "./VisDrone_by_class_val/labels"

os.makedirs(image_output_root, exist_ok=True)
os.makedirs(label_output_root, exist_ok=True)

# Get all images
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    label_filename = img_name.replace(".jpg", ".txt")
    label_path = os.path.join(label_dir, label_filename)

    class_ids = []

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if parts:
                    class_ids.append(int(parts[0]))

    # Use majority class if any, otherwise skip
    if class_ids:
        majority_class = Counter(class_ids).most_common(1)[0][0]
    else:
        print(f"Skipping {img_name}: no labels")
        continue

    # Create class-specific folders under images/ and labels/
    image_out_class_dir = os.path.join(image_output_root, f"class_{majority_class}")
    label_out_class_dir = os.path.join(label_output_root, f"class_{majority_class}")
    os.makedirs(image_out_class_dir, exist_ok=True)
    os.makedirs(label_out_class_dir, exist_ok=True)

    # Copy image and label to appropriate class folders
    shutil.copy(img_path, os.path.join(image_out_class_dir, img_name))

    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(label_out_class_dir, label_filename))
