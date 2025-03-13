import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import zipfile
import requests
import kagglehub
import random
import shutil

import os
import random
import shutil

# Number of total samples per class (not total across both)
num_samples = 1000  # Adjust this as needed
val_split = 0.2  # 20% of the data for validation

# Define dataset directories
dataset_path = "/Users/sophiasalta/.cache/kagglehub/datasets/bhavikjikadara/dog-and-cat-classification-dataset/versions/1"
print("Dataset downloaded to:", dataset_path)

train_dir = os.path.join(dataset_path, "PetImages")  # The correct root directory
cat_train_dir = os.path.join(train_dir, "Cat")
dog_train_dir = os.path.join(train_dir, "Dog")

# Define subset directories
subset_dir = os.path.join(dataset_path, "Subset")
subset_train_dir = os.path.join(subset_dir, "train")
subset_val_dir = os.path.join(subset_dir, "val")

subset_cat_train_dir = os.path.join(subset_train_dir, "Cat")
subset_dog_train_dir = os.path.join(subset_train_dir, "Dog")
subset_cat_val_dir = os.path.join(subset_val_dir, "Cat")
subset_dog_val_dir = os.path.join(subset_val_dir, "Dog")

# Function to clear and recreate directories
def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Delete existing directory
    os.makedirs(directory)  # Recreate empty directory

# Clear and recreate subset directories
for directory in [subset_cat_train_dir, subset_dog_train_dir, subset_cat_val_dir, subset_dog_val_dir]:
    clear_directory(directory)

# Function to split data into train and validation
def split_data(src_dir, train_dest, val_dest, num_samples, val_ratio):
    all_images = os.listdir(src_dir)
    sampled_images = random.sample(all_images, min(num_samples, len(all_images)))

    print(f"Source: {src_dir}, Sampled Images: {len(sampled_images)}")  # Debugging

    val_size = int(len(sampled_images) * val_ratio)
    train_images = sampled_images[val_size:]
    val_images = sampled_images[:val_size]

    # Copy files to respective directories
    for img in train_images:
        shutil.copy(os.path.join(src_dir, img), os.path.join(train_dest, img))
    for img in val_images:
        shutil.copy(os.path.join(src_dir, img), os.path.join(val_dest, img))

# Create train and validation subsets
if os.path.exists(cat_train_dir) and os.path.exists(dog_train_dir):
    split_data(cat_train_dir, subset_cat_train_dir, subset_cat_val_dir, num_samples, val_split)
    split_data(dog_train_dir, subset_dog_train_dir, subset_dog_val_dir, num_samples, val_split)

    print("Subset created successfully!")
    print("Train cat images:", len(os.listdir(subset_cat_train_dir)))
    print("Validation cat images:", len(os.listdir(subset_cat_val_dir)))
    print("Train dog images:", len(os.listdir(subset_dog_train_dir)))
    print("Validation dog images:", len(os.listdir(subset_dog_val_dir)))
else:
    print("Error: One or both directories do not exist. Check the dataset structure.")

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 18 * 18, 256)
        self.fc2 = nn.Linear(256, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.silu(self.conv1(x)))
        x = self.pool(F.silu(self.conv2(x)))
        x = self.pool(F.silu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.silu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize model
model = CNNModel().to(device)

# MixUp Augmentation
def mixup_data(x, y, alpha=0.2):
    """Apply MixUp or CutMix randomly"""
    if np.random.rand() < 0.5:  # 50% chance to apply CutMix
        return cutmix_data(x, y, alpha)
    else:
        return mixup_only(x, y, alpha)

def mixup_only(x, y, alpha=0.4):
    """Standard MixUp"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# CutMix Augmentation
def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation"""
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = x.shape
    rand_index = torch.randperm(batch_size).to(x.device)

    # Generate bounding box
    cx, cy = np.random.randint(W), np.random.randint(H)
    rw, rh = int(W * np.sqrt(1 - lam)), int(H * np.sqrt(1 - lam))
    x1, x2 = np.clip([cx - rw // 2, cx + rw // 2], 0, W)
    y1, y2 = np.clip([cy - rh // 2, cy + rh // 2], 0, H)

    # Apply CutMix
    x[:, :, y1:y2, x1:x2] = x[rand_index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (H * W))  # Adjust lambda
    y_a, y_b = y, y[rand_index]
    return x, y_a, y_b, lam

# Loss function for MixUp & CutMix
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Dataset & DataLoader
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(subset_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Loss function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

epochs = 20  # Change this to the number of epochs you want

# Training loop with CutMix & MixUp
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        images, labels_a, labels_b, lam = mixup_data(images, labels)  # Apply MixUp & CutMix

        optimizer.zero_grad()
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        running_loss += loss.item()

        # Compute Accuracy
        _, preds = torch.max(outputs, 1)
        correct += (lam * (preds == labels_a).sum().item()) + ((1 - lam) * (preds == labels_b).sum().item())
        total += labels.size(0)
        batch_acc = 100 * correct / total  # Batch-wise accuracy
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {batch_acc:.2f}%")

    # Compute full epoch accuracy
    epoch_acc = 100 * correct / total
    avg_train_loss = running_loss / len(train_loader)
    
    print(f"Epoch {epoch+1} Completed - Avg Loss: {avg_train_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%")

val_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

validation_dataset = datasets.ImageFolder(subset_val_dir, transform=val_transform)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# Validation Phase
model.eval()
val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in validation_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)  # Standard loss, no MixUp/CutMix
        val_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_val_loss = val_loss / len(validation_loader)
val_accuracy = 100 * correct / total  # Regular accuracy

print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")