from argparse import Namespace
from glob import glob
import torch
from PIL import Image
import numpy as np
import cv2
import albumentations as A
import inference_utils
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from diffusers import StableDiffusionDepth2ImgPipeline, StableDiffusionXLImg2ImgPipeline
import os

args = Namespace(
    janus_model_path='deepseek-ai/Janus-Pro-1B',
    model_path='/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/outputs',
    pretrained_model_path='stabilityai/stable-diffusion-2-depth',
    image_path='/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/Input',
    output_path='/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/Output',
    augmented_folder='/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/Inference/5-variant_folder/augmented',
    output_alphabet=['a', 'b', 'c', 'd', 'e'],
    seed=[42, 123, 456, 789, 112],
    device='cuda' if torch.cuda.is_available() else 'cpu',
    prompt_for_understanding_image='Describe the image and its structural details, quality, foreground and background separation, and intricate textures. Emphasize the key depth-based elements that define the scene and generate a detailed textual description. Answer in 50 words',
    negative_prompt="bad, deformed, ugly, bad anotomy, bad resolution, bad quality, bad asthetic, blurry",
    num_variations=5,
    augment1=A.Compose([
        A.Compose([
            A.HorizontalFlip(p=0.9),
            A.VerticalFlip(p=0.9)], p=0.7)
    ]),
    augment2=A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
        A.CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), p=0.6),
        # A.Equalize(p=0.3),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=8, val_shift_limit=8, p=0.7),
    ]),
    input_path_list=[]
)
number = 1
import cv2
import numpy as np
import random

def transform_colors(image):
    # Convert to HSV for better color manipulation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV for rainbow colors
    color_ranges = {
        'red': ([0, 50, 50], [10, 255, 255], [170, 50, 50], [180, 255, 255]),
        'orange': ([11, 50, 50], [20, 255, 255]),
        'yellow': ([21, 50, 50], [30, 255, 255]),
        'green': ([35, 40, 40], [85, 255, 255]),
        'blue': ([100, 50, 50], [130, 255, 255]),
        'indigo': ([131, 50, 50], [140, 255, 255]),
        'violet': ([141, 50, 50], [160, 255, 255])
    }

    # Define 6 replacement colors (excluding white and black)
    replacement_colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255]   # Cyan
    ]

    # Shuffle the replacement colors to randomize the assignment
    random.shuffle(replacement_colors)

    # Create a mask for black pixels
    black_mask = cv2.inRange(image, np.array([0, 0, 0]), np.array([50, 50, 50]))

    # Convert black to gray
    image[black_mask > 0] = [128, 128, 128]

    # Create a mask for white pixels
    white_mask = cv2.inRange(image, np.array([200, 200, 200]), np.array([255, 255, 255]))

    # Track used replacement colors
    used_colors = set()

    # Replace white pixels with one of the 6 replacement colors (ensuring no duplicates)
    if np.any(white_mask):  # Check if there are any white pixels
        available_colors = [c for c in replacement_colors if tuple(c) not in used_colors]
        if available_colors:
            replacement_color = random.choice(available_colors)
            image[white_mask > 0] = replacement_color
            used_colors.add(tuple(replacement_color))  # Mark this color as used

    # Replace rainbow colors with the 6 predefined colors (ensuring no duplicates)
    for i, (color, ranges) in enumerate(color_ranges.items()):
        if color == 'red':
            lower1, upper1, lower2, upper2 = ranges
            mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower, upper = ranges
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Assign a replacement color that hasn't been used yet
        available_colors = [c for c in replacement_colors if tuple(c) not in used_colors]
        if available_colors:
            replacement_color = random.choice(available_colors)
            image[mask > 0] = replacement_color
            used_colors.add(tuple(replacement_color))  # Mark this color as used

    return image

import re

args.input_path_list = sorted(glob(args.image_path + '/*'))
def sort_key(path):
    filename = os.path.basename(path)
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else float('inf')
def adjust_brightness(img):
    """Automatically adjusts brightness:
    - Increases brightness if the image is too dark.
    - Decreases brightness if the image is too bright.
    """
    if img is None:
        print("Error: Image not loaded properly in adjust_brightness()")
        return None
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean()  # Compute average brightness

    if brightness < 100:  # Dark images
        brightness_factor = np.random.uniform(0.3, 0.6)  # Increase brightness
    elif brightness > 180:  # Bright images
        brightness_factor = np.random.uniform(-0.6, -0.3)  # Decrease brightness
    else:
        brightness_factor = 0  # Keep original

    # Apply brightness adjustment
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + (brightness_factor * 255), 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

args.input_path_list = sorted(args.input_path_list, key=sort_key)
args.input_path_list = args.input_path_list[number-1:]
print(args.input_path_list[0])

for i, image_path in enumerate(args.input_path_list):
    answers = []
    init_image = Image.open(image_path).convert("RGB")
    init_image_np = np.array(init_image)
    for j in range(args.num_variations):
        augmented_image1 = args.augment1(image=init_image_np)
        augmented_image2 = args.augment2(image=augmented_image1['image'])
        init_image2 = Image.fromarray(augmented_image2['image'])
        # init_image2 = transform_colors(np.array(init_image2))
        # init_image2 = Image.fromarray(init_image2)
        init_image2.save(f"{args.augmented_folder}/augmented_image{i+1}{args.output_alphabet[j]}.png")
