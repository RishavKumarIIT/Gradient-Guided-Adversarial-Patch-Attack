import time
import torch
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def AddPatches(image,model,top_coords_sorted):
    start_time = time.time() 

    img_np = np.array(image).copy()

   
    transform = T.Compose([
       T.ToTensor(),
       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        original_output = model(img_tensor)
        true_label = original_output.argmax(dim=1).item()

    #print(" True Label (before attack):", true_label)

    
    max_steps = 100
    threshold = 0.3
    initial_alpha = 0.01
    initial_patch_size = 1
    max_patch_size = 28
    growth_rate = 1.05
    alpha_growth = 0.02

    # --------- Initial State ---------
    best_img = img_np.copy()
    best_score = 1.0
    patch_h = patch_w = initial_patch_size
    alpha = initial_alpha

    # --------- Attack Loop ---------
    for step in range(max_steps):
        temp_img = img_np.copy()

        #for focus_y, focus_x in tqdm(top_coords_sorted[:1000], desc="Applying patches"):
        for focus_y, focus_x in top_coords_sorted:
            patch_image = image.convert("RGBA")
            patch_resized = patch_image.resize((patch_w, patch_h))
            patch_np = np.array(patch_resized)

            patch_rgb = patch_np[..., :3]
            patch_alpha = (patch_np[..., 3:] / 255.0).astype(np.float32)

            img_h, img_w = temp_img.shape[:2]

            # Compute patch placement coordinates
            top = max(0, min(focus_y - patch_h // 2, img_h - patch_h))
            left = max(0, min(focus_x - patch_w // 2, img_w - patch_w))
            roi = temp_img[top:top + patch_h, left:left + patch_w].astype(np.float32)

            # Compute average color of the ROI
            #noise = np.random.uniform(-10, 10, size=(patch_h, patch_w, 3)).astype(np.uint8)
            #patch_rgb = np.clip(avg_color + noise, 0, 255)
            #patch_alpha = np.ones((patch_h, patch_w, 1), dtype=np.float32)

            base_color = temp_img[focus_y, focus_x].astype(np.float32)  # shape: (3,)
            patch_rgb = np.tile(base_color, (patch_h, patch_w, 1)).astype(np.uint8)
            patch_alpha = np.ones((patch_h, patch_w, 1), dtype=np.float32)

            # Final blending: weighted by transparency
            effective_alpha = alpha * patch_alpha
            blended = (effective_alpha * patch_rgb + (1 - effective_alpha) * roi).astype(np.uint8)
            temp_img[top:top + patch_h, left:left + patch_w] = blended

        # Evaluate classifier
        temp_tensor = transform(Image.fromarray(temp_img)).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(temp_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_label = output.argmax(dim=1).item()
            pred_score = probs[0, pred_label].item()

        if pred_score < best_score:
            best_score = pred_score
            best_img = temp_img.copy()

        #print(f"Step {step:3d}: Score = {pred_score:.4f}, alpha = {alpha:.2f}, patch = {patch_h}x{patch_w}")
        #print("True label:", true_label, "| Predicted label:", pred_label)

        if  pred_label != true_label:
            best_img = temp_img.copy()
            #print(" Classifier fooled!")
            break

        if step%10==0 or alpha==1:
            patch_h = patch_w = min(max_patch_size, int(patch_h + growth_rate))
        alpha = min(1.0, alpha + alpha_growth)

    #print(f"Step {step:3d}: Score = {pred_score:.4f}, alpha = {alpha:.2f}, patch = {patch_h}x{patch_w}")
    #print("True label:", true_label, "| Predicted label:", pred_label)
    end_time = time.time()  
    time_taken = end_time - start_time
    return best_img,true_label,pred_label,alpha,patch_h,time_taken



    # Save final image
    #SaveImage(best_img, 16)
