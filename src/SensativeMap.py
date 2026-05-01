import torch
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2

import time


def replace_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU(inplace=False))
        else:
            replace_relu(child)
    return model



#image_path = "/home/mtech/rishav/data/content.jpg"#r"C:\Users\CSE IIT BHILAI\Adversial-ML-Project\shape_cam\data\target_images\tiger_shark.jpg"
#image = Image.open(image_path).convert("RGB").resize((224, 224))
#image_np = np.array(image)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def FetchTopFeatures(image,model):
    start_time = time.time()
    image_np = np.array(image)

    transform = T.Compose([
       T.ToTensor(),
       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    img_tensor.requires_grad_()  # ensure gradient is tracked

    
    img_tensor = img_tensor.to(device)

    # -------------------- Load VGG16 and Fix ReLU --------------------
    #model = models.vgg16(pretrained=True)
    model = replace_relu(model)
    model = model.eval().to(device)

    # -------------------- Forward & Backward Pass --------------------
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    score = output[0, pred_class]
    score.backward()

    # -------------------- Sensitivity Map: ∂L/∂x --------------------
    grad = img_tensor.grad.data.squeeze().cpu()  # shape: [3, H, W]

    # Compute per-pixel gradient magnitude (across channels)
    sensitivity = grad.abs().mean(dim=0).numpy()  # shape: [H, W]
    sensitivity = (sensitivity - sensitivity.min()) / (sensitivity.max() + 1e-8)
    sensitivity_resized = cv2.resize(sensitivity, (224, 224))

    # -------------------- Top Sensitive Points --------------------
    flat = sensitivity_resized.flatten()
    top_indices = np.argpartition(-flat, 100)
    top_coords = [np.unravel_index(idx, sensitivity_resized.shape) for idx in top_indices]
    top_coords_sorted = sorted(top_coords, key=lambda c: sensitivity_resized[c], reverse=True)

    # -------------------- Overlay Heatmap and Dots --------------------
    heatmap = cv2.applyColorMap(np.uint8(255 * sensitivity_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_np, 0.5, heatmap, 0.5, 0)

    # Red dots at top sensitive points
    for (y, x) in top_coords_sorted[:10]:
        cv2.circle(overlay, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

    # -------------------- Show Result --------------------
    #plt.figure(figsize=(8, 8))
    #plt.imshow(overlay[..., ::-1])  # BGR to RGB
    #plt.title(f"VGG16 Sensitivity Map - Pred Class ID: {pred_class}")
    #plt.axis("off")
    #plt.show()
    end_time = time.time()  
    time_taken = end_time - start_time
    return model,top_coords_sorted,time_taken


