import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from PIL import Image
import os
from torch.utils.data import DataLoader
import torch.optim as optim

from LoadData import LoadDataSet
from LoadModel import load_model
from TrainAndEval import train_and_evaluate
from SensativeMap import FetchTopFeatures
from InitializeAndAddPatches import AddPatches
from Evaluation import(
    calculate_pixel_change_percentage,
    show_all_pixel_changes_grid,
    show_pixel_changes
)

def overall_transparency(original, perturbed):
    diff = np.abs(perturbed - original)
    alpha_map = np.mean(diff, axis=-1)  
    return np.mean(alpha_map)




#  Dataset and model lists
datasets_list = [
    "svhn"
     ,"cifar10", "cifar100", "fashionmnist"#,"custom"
    ,"imagenet"
    
]

models_list = [
  "resnet34"
  , "vgg13",
    "densenet121" ,"vgg16"
  , "vgg19","resnet18", "resnet50", "densenet169", "densenet201"
]

# Create output folder
os.makedirs("saved_images", exist_ok=True)
image_pairs = []

#  Main experiment loop
for model_name in models_list:
    model = load_model(model_name)  # Your function for loading models

    for ds in datasets_list:
        train_dataset,classes = LoadDataSet(ds, split_val=1) 
        test_dataset,classes = LoadDataSet(ds, split_val=1)  
        if ds!="imagenet":
            pass
          #model,acc=train_and_evaluate(model,train_dataset,test_dataset,classes,model_name,ds)
        else:
            test_dataset,classes = LoadDataSet(ds, split_val=1)
        dataset=test_dataset
        
        alpha_val = []
        change_pix = []
        alpha_30_2=0
        alpha_30_5=0
        alpha_30_7=0

        alpha_50_2=0
        alpha_50_5=0
        alpha_50_7=0

        alpha_70_2=0
        alpha_70_5=0
        alpha_70_7=0

        total_time_30=0
        total_time_50=0
        total_time_70=0
        total_time_100=0

        thirty=0
        fifty=0
        seventy=0

        time_taken_feature=0

        print(f"\n Running Model: {model_name} | Dataset: {ds} | Samples: {len(dataset)}")

        for idx, (img, label) in tqdm(enumerate(dataset), total=len(dataset), desc="Processing Images"):
            # Convert Tensor to PIL if needed
            if isinstance(img, torch.Tensor):
                pil_img = to_pil_image(img)
            else:
                pil_img = img

            # Resize and convert to RGB
            pil_img = pil_img.convert("RGB").resize((224, 224))

            # Save image to disk for your patch functions
            image_path = os.path.join("saved_images", f"{ds}{model_name}_sample{idx}.png")
            pil_img.save(image_path)

            # Run your project code
            image_np = np.array(pil_img).copy()
            model, top_coords_sorted,time_taken_fe = FetchTopFeatures(pil_img, model)
            best_img, true_label, pred_label, alpha, patch_h,time_taken = AddPatches(
                pil_img , model, top_coords_sorted
            )
            #alpha=overall_transparency(image_np, best_img)/100

            # Calculate change percentage
            change_percent = calculate_pixel_change_percentage(image_np, best_img, 10)
            alpha_val.append(alpha)
            change_pix.append(change_percent)

            # If you want to keep visual samples
            #if idx == 0:
            image_pairs.append([image_np, best_img,change_percent ,alpha])#true_label , pred_label])
            total_time_100+=time_taken
            time_taken_feature+=time_taken_fe
            
            if change_percent<=2 and alpha<=0.3:
                alpha_30_2+=1
                total_time_30+=time_taken
                thirty+=1
            if change_percent<=5 and alpha<=0.3:
                alpha_30_5+=1
                total_time_50+=time_taken
                fifty+=1
            if change_percent<=7 and alpha<=0.3:
                alpha_30_7+=1
                total_time_70+=time_taken
                seventy+=1

            if change_percent<=2 and alpha<=0.5:
                alpha_50_2+=1
            if change_percent<=5 and alpha<=0.5:
                alpha_50_5+=1
            if change_percent<=7 and alpha<=0.5:
                alpha_50_7+=1

            if change_percent<=2 and alpha<=0.7:
                alpha_70_2+=1
            if change_percent<=4 and alpha<=0.7:
                alpha_70_5+=1
            if change_percent<=7 and alpha<=0.7:
                alpha_70_7+=1

        #  Metrics
        average_alpha = sum(alpha_val) / len(alpha_val) if alpha_val else 0
        average_pixelChange = sum(change_pix) / len(change_pix) if change_pix else 0
        avg_time_taken=total_time_100/len(dataset)
        avg_time_taken_fe=time_taken_feature/len(dataset)
        #print(len(dataset),time_taken,total_time_100)

        print(f" Model: {model_name} | Dataset: {ds}")
        print(f"    Average alpha: {average_alpha:.4f}")
        print(f"    Average pixel percent change: {average_pixelChange:.4f}")
        print(f"    Average time taken: {avg_time_taken:.4f}")
        print(f"    Average time taken for features selection: { avg_time_taken_fe:.4f}")
        print(total_time_30/(thirty+1),total_time_50/(fifty+1),total_time_70/(seventy+1))
        
        print(f"    Average alpha_30_2: {alpha_30_2/len(dataset):.4f}")
        print(f"    Average alpha_30_5: {alpha_30_5/len(dataset):.4f}")
        print(f"    Average alpha_30_7: {alpha_30_7/len(dataset):.4f}")
        

        print(f"    Average alpha_50_2: {alpha_50_2/len(dataset):.4f}")
        print(f"    Average alpha_50_5: {alpha_50_5/len(dataset):.4f}")
        print(f"    Average alpha_50_7: {alpha_50_7/len(dataset):.4f}")

        print(f"    Average alpha_70_2: {alpha_70_2/len(dataset):.4f}")
        print(f"    Average alpha_70_5: {alpha_70_5/len(dataset):.4f}")
        print(f"    Average alpha_70_7: {alpha_70_7/len(dataset):.4f}")
        #break

    break

show_all_pixel_changes_grid(image_pairs)
