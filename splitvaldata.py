import os
import shutil
import random

def split_images_and_labels(train_root, val_root, val_ratio=0.2, seed=42):
    random.seed(seed)

    train_images_root = os.path.join(train_root, 'images')
    train_labels_root = os.path.join(train_root, 'labels')
    
    val_images_root = os.path.join(val_root, 'images')
    val_labels_root = os.path.join(val_root, 'labels')
    
    os.makedirs(val_images_root, exist_ok=True)
    os.makedirs(val_labels_root, exist_ok=True)

    for class_name in os.listdir(train_images_root):
        image_class_dir = os.path.join(train_images_root, class_name)
        label_class_dir = os.path.join(train_labels_root, class_name)

        if not os.path.isdir(image_class_dir):
            continue

        # Make corresponding validation folders
        os.makedirs(os.path.join(val_images_root, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_labels_root, class_name), exist_ok=True)

        image_files = os.listdir(image_class_dir)
        random.shuffle(image_files)

        val_count = int(len(image_files) * val_ratio)
        val_files = image_files[:val_count]

        for file in val_files:
            # Move image
            img_src = os.path.join(image_class_dir, file)
            img_dst = os.path.join(val_images_root, class_name, file)
            shutil.move(img_src, img_dst)

            # Move corresponding label
            label_file = os.path.splitext(file)[0] + '.txt'
            label_src = os.path.join(label_class_dir, label_file)
            label_dst = os.path.join(val_labels_root, class_name, label_file)
            
            if os.path.exists(label_src):
                shutil.move(label_src, label_dst)
            else:
                print(f"Warning: Label not found for image {file}")

        print(f"{class_name}: Moved {val_count} image-label pairs to validation set.")

# Usage
split_images_and_labels(
    train_root='VisDrone_by_class_train',
    val_root='val2',
    val_ratio=0.2
)
