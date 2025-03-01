import os
import numpy as np
import re
import cv2
from glob import glob
from tqdm import tqdm

# Configuration
image_output_folder = '/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/Structure-Output'
output_path = '/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/Output'
sharpening_params = {'kernel_size': (5, 5), 'sigma': 0, 'alpha': 1.5, 'beta': -0.5, 'gamma': 0}
saturation_increment = 30
color_balance_inc = np.array([15, 10, -5], dtype="int16")  # Adjust RGB values for warmth/coolness

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

def sort_key(path):
    filename = os.path.basename(path)
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else float('inf')

def process_image(image_path, output_image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Unable to read image {image_path}")
        return

    # Apply Gaussian Blur and Unsharp Masking
    blurred = cv2.GaussianBlur(img, sharpening_params['kernel_size'], sharpening_params['sigma'])
    sharp = cv2.addWeighted(img, sharpening_params['alpha'], blurred, sharpening_params['beta'], sharpening_params['gamma'])

    # Adjust Saturation
    hsv = cv2.cvtColor(sharp, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, saturation_increment)
    s = np.clip(s, 0, 255)
    enhanced_hsv = cv2.merge((h, s, v))
    final = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    # Adjust Color Balance
    balanced = np.clip(final + color_balance_inc, 0, 255).astype(np.uint8)

    # Save the final image
    cv2.imwrite(output_image_path, balanced)

# Get sorted list of image paths
image_output_paths = sorted(glob(os.path.join(image_output_folder, '*')), key=sort_key)

# Process images with progress tracking
for i, image_path in enumerate(tqdm(image_output_paths, desc="Processing Images")):
    filename = os.path.basename(image_path)
    output_image_path = os.path.join(output_path, filename)
    process_image(image_path, output_image_path)

