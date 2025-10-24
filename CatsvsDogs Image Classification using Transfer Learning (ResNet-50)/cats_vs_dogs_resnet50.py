import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available. Please enable GPU in the environment settings.")
    return torch.device("cuda")


def get_data_loaders(data_dir, batch_size=32):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.classes


def initialize_model(num_classes=2):
    model = models.resnet50(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace last layer to match number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs=5):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        val_acc = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f} - Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        if scheduler:
            scheduler.step()

    print(f"Best Validation Accuracy: {best_acc:.4f}")


def evaluate_model(model, val_loader, device):
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    val_acc = running_corrects.double() / len(val_loader.dataset)
    return val_acc


def predict_image(image_path, model, class_names, device):
    img = Image.open(image_path)

    plt.imshow(img)
    plt.axis('off')
    plt.title("Input Image")
    plt.show()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img_t = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
    predicted_label = class_names[pred.item()]
    print("Predicted Class:", predicted_label)
    return predicted_label


def main():
    set_seed()

    data_dir = "/kaggle/input/cats-dogs"  # Update with your dataset path
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.001

    device = get_device()
    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size)

    model = initialize_model(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs)

    # Example prediction on a test image
    test_image_path = os.path.join(data_dir, "val/cat/Ragdoll_158_jpg.rf.f9c36ee093139d00405f4f0838dd00d8.jpg")
    predict_image(test_image_path, model, class_names, device)


if __name__ == "__main__":
    main()
