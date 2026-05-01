import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_pixel_changes(original_img, modified_img):
 
    assert original_img.shape == modified_img.shape, "Image shapes must match"
    
    # Convert to RGB if needed (OpenCV uses BGR by default)
    if original_img.shape[2] == 3:
        original_rgb = original_img.copy()
        modified_rgb = modified_img.copy()
    else:
        raise ValueError("Images must have 3 color channels (RGB)")

    # Compute pixel-wise difference
    diff_mask = np.any(original_rgb != modified_rgb, axis=2)  # Shape: (H, W), True where different
    diff_visual = modified_rgb.copy()

    # Highlight changed pixels in red on the modified image
    diff_visual[diff_mask] = [255, 0, 0]  # Red color for changed pixels
    return diff_visual




def show_all_pixel_changes_grid(image_pairs):
    """
    Display original, modified, and difference images in a single grid.
    Each row corresponds to one image set.
    
    :param image_pairs: List of tuples [(original1, modified1), (original2, modified2), ...]
    """
    num_images = len(image_pairs)
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)  # handle single row case

    for idx, (original_img, modified_img,true_label,pred_label) in enumerate(image_pairs):
        diff = show_pixel_changes(original_img, modified_img)

        axes[idx, 0].imshow(original_img.astype(np.uint8))
        axes[idx, 0].set_title(f'True Level {true_label}')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(modified_img.astype(np.uint8))
        axes[idx, 1].set_title(f'predicted level {pred_label}')
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(diff)
        axes[idx, 2].set_title(f'Difference {idx+1}')
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.show()



def calculate_pixel_change_percentage(original_img, modified_img, threshold=5):
    """
    Calculates the percentage of pixels that changed significantly 
    (by more than `threshold`) between original and modified images.
    
    Args:
        original_img (np.ndarray): Original image (H x W x 3), dtype=uint8
        modified_img (np.ndarray): Modified image (H x W x 3), dtype=uint8
        threshold (int): Pixel intensity change threshold (0–255)
        
    Returns:
        float: Percentage of significantly changed pixels
    """
    assert original_img.shape == modified_img.shape, "Images must be the same shape"
    
    # Convert to grayscale
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    modified_gray = cv2.cvtColor(modified_img, cv2.COLOR_RGB2GRAY)

    # Compute absolute pixel-wise difference
    abs_diff = np.abs(original_gray.astype(np.int16) - modified_gray.astype(np.int16))

    # Count pixels where change exceeds threshold
    changed_pixels = np.sum(abs_diff > threshold)
    total_pixels = original_gray.size

    percentage_change = (changed_pixels / total_pixels) * 100
    return percentage_change
