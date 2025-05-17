import os
import pandas as pd
from PIL import Image
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

ROOT_PATH = "stage-3/week_1/data/handwriting"
DIRECTORY_FILENAME = "image_labels.csv"
TRAINING_FOLDER_STRUCTURE = "augmented_images/augmented_images1"
TEST_FOLDER_DIR = "handwritten-english-characters-and-digits/combined_folder/test"
NUM_EPOCHS = 20
KERNEL_SIZE = 3

class HandwritingCSV(Dataset):
    def __init__(self, csv_path: str, root_dir: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root = root_dir
        self.transform = transform

        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fn = row['filename']
        label_str = row['label']
        label = self.class_to_idx[label_str]

        # CSV filename e.g.,"0/0.031.2.augmented.png"
        img_path = os.path.join(
            self.root,
            TRAINING_FOLDER_STRUCTURE,
            fn
        )
        img = Image.open(img_path) # PIL Image
        if self.transform:
            img = self.transform(img) # Grayscale → ToTensor → Normalize
        return img, label

class CharCNN(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256), # 28 -> 14 -> 7 -> 3
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_training_transform() -> transforms.Compose:
    # define transform pipeline
    return transforms.Compose([
        transforms.RandomResizedCrop((28, 28)),
        # transforms.Resize((28, 28)), 
        transforms.Grayscale(num_output_channels=1), # for single-channel
        transforms.ToTensor(), # → [0,1]
        transforms.Normalize((0.5,), (0.5,)) # → [-1,1], will improve the sensitivity of tensors
    ])

def get_testing_transform() -> transforms.Compose:
    # define transform pipeline
    return transforms.Compose([
        # transforms.RandomResizedCrop((28, 28)),
        transforms.Resize((28, 28)), 
        transforms.Grayscale(num_output_channels=1), # for single-channel
        transforms.ToTensor(), # → [0,1]
        transforms.Normalize((0.5,), (0.5,)) # → [-1,1], will improve the sensitivity of tensors
    ])

def load_train_data(
        transform: transforms.Compose,
        root_path: str,
        directory_filename: str
    ) -> Tuple:
    train_csv = os.path.join(
        root_path,
        directory_filename
    )

    train_dataset = HandwritingCSV(
        csv_path = train_csv,
        root_dir = root_path,
        transform = transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=64,
        shuffle=True, num_workers=8
    )

    return train_dataset, train_loader

def load_test_data(
        transform: transforms.Compose,
        root_path: str,
        test_folder_dir: str
    ) -> Tuple:
    test_root = os.path.join(
        root_path,
        test_folder_dir
    )

    # will load files and make the each folder under 
    # the root of files as label (e.g. '0','1',...)
    test_dataset = datasets.ImageFolder(
        root=test_root,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset, batch_size=64,
        shuffle=False, num_workers=8
    )

    return test_dataset, test_loader

def train_and_test(
        model: CharCNN, 
        criterion, 
        optimizer,
        epochs: int
    ):

    num_epochs = epochs
    for epoch in range(1, num_epochs+1):
        # train
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        test_acc   = 100.0 * correct / total
        print(f"Epoch {epoch:02d}/{num_epochs:02d} — "
            f"Train Loss: {train_loss:.4f} — "
            f"Test Acc: {test_acc:.2f}%")
    

if __name__ == "__main__":
    
    training_transform = get_training_transform()
    testing_transform = get_testing_transform()
    train_dataset, train_loader = load_train_data(training_transform, ROOT_PATH, DIRECTORY_FILENAME)
    test_dataset, test_loader = load_test_data(testing_transform, ROOT_PATH, TEST_FOLDER_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(train_dataset.classes) # number of classes
    model = CharCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_and_test(model, criterion, optimizer, NUM_EPOCHS)