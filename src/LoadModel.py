import torch
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2

def load_model(model_name="resnet50", device="cpu"):
    model_name = model_name.lower()

    try:
        if model_name == "vgg13":
            from torchvision.models import VGG13_Weights
            model = models.vgg13(weights=VGG13_Weights.DEFAULT)
        elif model_name == "vgg16":
            from torchvision.models import VGG16_Weights
            model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        elif model_name == "vgg19":
            from torchvision.models import VGG19_Weights
            model = models.vgg19(weights=VGG19_Weights.DEFAULT)
        elif model_name == "resnet18":
            from torchvision.models import ResNet18_Weights
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_name == "resnet34":
            from torchvision.models import ResNet34_Weights
            model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        elif model_name == "resnet50":
            from torchvision.models import ResNet50_Weights
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == "densenet121":
            from torchvision.models import DenseNet121_Weights
            model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        elif model_name == "densenet169":
            from torchvision.models import DenseNet169_Weights
            model = models.densenet169(weights=DenseNet169_Weights.DEFAULT)
        elif model_name == "densenet201":
            from torchvision.models import DenseNet201_Weights
            model = models.densenet201(weights=DenseNet201_Weights.DEFAULT)
        else:
            raise ValueError(f"Model '{model_name}' not supported.")

    except ImportError:
        try:
            model = getattr(models, model_name)(pretrained=True)
        except AttributeError:
            raise ValueError(f"Model '{model_name}' not found in torchvision.models.")

    model = model.eval().to(device)
    return model
