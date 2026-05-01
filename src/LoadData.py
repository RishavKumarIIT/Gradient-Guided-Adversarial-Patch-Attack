from datasets import load_dataset
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torch
import scipy.io as sio

def LoadDataSet(ds, split_val=1.0, train=True, download=True, data_dir="data"):
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    ds = ds.lower()
    classes=0

    # ---------------- Real Datasets ----------------
    if ds == "cifar10":
        dataset = datasets.CIFAR10(root=data_dir, train=train, transform=transform, download=download)
        classes=10
    elif ds == "cifar100":
        dataset = datasets.CIFAR100(root=data_dir, train=train, transform=transform, download=download)
        classes=100
    elif ds == "mnist":
        dataset = datasets.MNIST(root=data_dir, train=train, transform=transform, download=download)
        classes=10
    elif ds == "fashionmnist":
        dataset = datasets.FashionMNIST(root=data_dir, train=train, transform=transform, download=download)
        classes=10
    elif ds == "svhn":
        dataset = datasets.SVHN(root=data_dir, split="train" if train else "test", transform=transform, download=download)
        classes=10
    elif ds == "imagenet":
        dataset = ImageFolder(root=f"{data_dir}/imagenet/train" if train else f"{data_dir}/imagenet/val",transform=transform)
    elif ds == "custom":
        dataset = ImageFolder(root=f"{data_dir}/custom/train" if train else f"{data_dir}/custom/train",transform=transform)

    # ---------------- Synthetic Datasets ----------------
    elif ds == "fakedata":
        dataset = datasets.FakeData(size=1000, image_size=(3, 32, 32), num_classes=10, transform=transform)
    elif ds == "synthetic":
        images = torch.rand(1000, 3, 32, 32)
        labels = torch.randint(0, 10, (1000,))
        dataset = torch.utils.data.TensorDataset(images, labels)

    else:
        raise ValueError(f"Dataset '{ds}' not supported.")

    # ---------------- Apply Split ----------------
    if 0 < split_val < 1.0:
        split_size = int(len(dataset) * split_val)
        dataset, _ = torch.utils.data.random_split(dataset, [split_size, len(dataset) - split_size])

    return dataset,classes