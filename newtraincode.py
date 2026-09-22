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
from collections import Counter
from ResNet import ResNet50  # Ensure ResNet50 is correctly implemented
from losses import AFL_FL # Enable optimized backend for convolutions
torch.backends.cudnn.benchmark = True

# ✅ Custom Dataset Class
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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📡 Using device: {device}")

    classes = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
               'tricycle', 'awning-tricycle', 'bus', 'motor']
    num_classes = len(classes)

    # 🧪 Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 📂 Load Datasets
    print("📁 Loading dataset...")
    train_dataset = VisDroneClassificationDataset("./train", transform=transform)
    val_dataset = VisDroneClassificationDataset("./val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print(f"✅ Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    def count_class_distribution(dataloader, num_classes):
        class_counter = Counter()
        for _, labels in dataloader:
            labels = labels.view(-1).tolist()
            class_counter.update(labels)
        # Ensure all classes are represented (even if count is 0)
        return [class_counter.get(i, 0) for i in range(num_classes)]

    

    # Get class count lists
    train_class_counts = count_class_distribution(train_loader, num_classes)
    val_class_counts = count_class_distribution(val_loader, num_classes)

    total_class_counts = [train + val for train, val in zip(train_class_counts, val_class_counts)]
    # 🧠 Model
    print("🔁 Initializing model...")
    net = ResNet50(num_classes).to(device)
    
    criterion=AFL_FL(total_class_counts)


    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training

    EPOCHS = 200
    best_acc = 0.0
    best_model_path = "AFL_FL.pth"
    all_logs = []

    for epoch in range(EPOCHS):
        print(f"\n🚀 Epoch {epoch+1}/{EPOCHS} started...")
        start_time = time.time()
        net.train()
        losses = []
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                features, outputs = net(inputs)
                loss = criterion(outputs, labels)



            losses.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if i % 100 == 0 and i > 0:
                print(f"   🧮 Mini-batch {i}: Avg Loss = {running_loss / 100:.4f}")
                running_loss = 0.0

        avg_loss = sum(losses) / len(losses)
        scheduler.step(avg_loss)

        print("🔍 Evaluating on validation set...")
        net.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    features, outputs = net(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        #class_accuracies = cm.diagonal() / cm.sum(axis=1)
        with np.errstate(divide='ignore',invalid='ignore'):
            class_accuracies = cm.diagonal() / cm.sum(axis=1)
            class_accuracies = np.nan_to_num(class_accuracies)
        
        mean_val_acc = np.mean(class_accuracies)
        if mean_val_acc > best_acc:
            best_acc = mean_val_acc
            torch.save(net.state_dict(), best_model_path)
            print(f"🌟 Best model updated at epoch {epoch+1} with acc={best_acc:.4f}")

        epoch_log = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracies': class_accuracies.tolist(),
            'confusion_matrix': cm.tolist()
        }
        all_logs.append(epoch_log)

        print(f'\n📊 Epoch {epoch+1} Validation Accuracy:')
        for idx, acc in enumerate(class_accuracies):
            print(f'   {classes[idx]:<17}: {acc:.2%}')

        end_time = time.time()
        print(f"⏱️ Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds")
        print("-" * 50)

    print("✅ Training Completed")
    print(f"💾 Best model saved at {best_model_path} with accuracy={best_acc:.4f}")



    # 📄 Save logs
    import json
    import csv



    with open("AFL_FL_Final.csv", "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["Epoch", "Loss"] + classes)
        for log in all_logs:
            row = [log['epoch'], log['loss']] + [f"{acc:.4f}" for acc in log['accuracies']]
            writer.writerow(row)

    print("💾 Logs saved to AFL_FL_Final.csv")
