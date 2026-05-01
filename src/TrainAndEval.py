import os
import torch
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2

def get_transforms(is_grayscale, resize_to=224):
    """
    Returns transforms for dataset depending on grayscale or RGB input.
    """
    transform_list = []
    if is_grayscale:
        transform_list.append(transforms.Grayscale(3))  # convert 1-channel → 3-channel
    transform_list.extend([
        transforms.Resize((resize_to, resize_to)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    return transforms.Compose(transform_list)


def train_and_evaluate(model, train_dataset, test_dataset,
                       num_classes,model_name="model", dataset_name="data", batch_size=64, lr=0.001,
                       num_epochs=5, device=None, freeze_features=False,save_dir="./saved_models"):
    """
    General training & evaluation function for VGG, ResNet, DenseNet on MNIST, FashionMNIST, CIFAR10/100
    """

    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir,exist_ok=True)
    model_path=os.path.join(save_dir,f"{model_name}_{dataset_name}.pth")

    if os.path.exists(model_path):
        print("model founded ")

        #  Adjust classifier BEFORE loading weights
        if hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

        # Now load checkpoint safely
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        return model, None



    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #  Adjust classifier for VGG & AlexNet
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)

    #  Adjust classifier for ResNet, DenseNet
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    # Optionally freeze feature extractor
    if freeze_features and hasattr(model, "features"):
        for param in model.features.parameters():
            param.requires_grad = False
    elif freeze_features and hasattr(model, "conv1"):  # ResNet
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():  # keep final fc trainable
            param.requires_grad = True

    # Move model to device
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(),model_path)

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

    return model, acc