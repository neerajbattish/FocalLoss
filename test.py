import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from ResNet import ResNet50  # Ensure this exists and is correct

# ✅ Define custom dataset OUTSIDE main guard
class VisDroneClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_dir = os.path.join(root_dir, "images")
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        for idx, class_name in enumerate(sorted(os.listdir(self.image_dir))):
            class_path = os.path.join(self.image_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = idx
            for img_file in os.listdir(class_path):
                if img_file.endswith(".jpg"):
                    self.samples.append((os.path.join(class_path, img_file), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ✅ Main logic under main guard
if __name__ == '__main__':
    # 🔧 Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📡 Using device: {device}")

    classes = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    num_classes = len(classes)

    # 🎨 Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # or (256, 256), depending on model input
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 📂 Datasets and DataLoaders
    print("📁 Loading dataset...")
    train_dataset = VisDroneClassificationDataset("./VisDrone_by_class_train", transform=transform)
    val_dataset = VisDroneClassificationDataset("./VisDrone_by_class_val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    print(f"✅ Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # 🧠 Model setup
    print("🔁 Initializing model...")
    net = ResNet50(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    print("✅ Model ready. Starting training...\n")

    # 🚀 Training loop
    EPOCHS = 100
    for epoch in range(EPOCHS):
        print(f"\n🚀 Epoch {epoch+1}/{EPOCHS} started...")
        start_time = time.time()

        net.train()
        losses = []
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0 and i > 0:
                print(f"   🧮 Mini-batch {i}: Avg Loss = {running_loss / 100:.4f}")
                running_loss = 0.0

        avg_loss = sum(losses) / len(losses)
        scheduler.step(avg_loss)

        # 🧪 Validation
        print("🔍 Evaluating on validation set...")
        net.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        class_accuracies = cm.diagonal() / cm.sum(axis=1)

        print(f'\n📊 Epoch {epoch+1} Validation Accuracy:')
        for idx, acc in enumerate(class_accuracies):
            print(f'   {classes[idx]:<17}: {acc:.2%}')

        end_time = time.time()
        print(f"⏱️ Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds")
        print("-" * 50)

    print("✅ Training Completed")
