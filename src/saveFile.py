import torch
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2

def SaveImage(image, step,model):
    image_pil = Image.fromarray(image.astype(np.uint8)).convert("RGB").resize((224, 224))
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        top_score, top_idx = torch.max(prob, dim=0)

    # Denormalize for visualization
    denorm = T.Compose([
        T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
    img_vis = denorm(img_tensor[0]).permute(1, 2, 0).cpu().numpy()
    img_vis = np.clip(img_vis * 255, 0, 255).astype(np.uint8).copy()  # ✅ Fix: ensure OpenCV-compatible



    save_path = "outputs/step_{step:04d}.jpg"
    Image.fromarray(img_vis).save(save_path, format='JPEG')

    #plt.figure(figsize=(4, 4))
    #plt.imshow(img_vis)
    #plt.title(f"Step {step:04d} | Class: {top_idx.item()}")
    #plt.axis("off")
    #plt.show()

    return top_idx.item(), top_score.item()
