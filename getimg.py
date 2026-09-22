import os
import shutil

# Paths
labels_root = "./labels"   # your labels folder (class1, class2)
images_root = "./VisDrone2019-DET-train/images"   # all images are here
output_root = "./Images_train"   # final structure will be created here

# Loop through each class folder in labels
for class_folder in os.listdir(labels_root):
    class_label_path = os.path.join(labels_root, class_folder)
    class_output_path = os.path.join(output_root, class_folder)

    if os.path.isdir(class_label_path):
        # Create class folder in output
        os.makedirs(class_output_path, exist_ok=True)

        # Copy label files
        for txt_file in os.listdir(class_label_path):
            if txt_file.endswith(".txt"):
                base_name = os.path.splitext(txt_file)[0]
                # Create images subfolder inside class folder
                images_subfolder = os.path.join(class_output_path)
                os.makedirs(images_subfolder, exist_ok=True)

                # Find matching image in images_root
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    img_file = base_name + ext
                    img_path = os.path.join(images_root, img_file)
                    if os.path.exists(img_path):
                        shutil.copy(img_path, os.path.join(images_subfolder, img_file))
                        print(f"Copied {img_file} → {images_subfolder}")
                        break
