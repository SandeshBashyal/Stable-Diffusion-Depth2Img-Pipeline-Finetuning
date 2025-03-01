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
import re
import random

args = Namespace(
    janus_model_path='deepseek-ai/Janus-Pro-1B',
    model_path='/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/outputs',
    pretrained_model_path='stabilityai/stable-diffusion-2-depth',
    image_path='/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/Input',
    output_path='/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/Structure-Output',
    augmented_folder='/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/Inference/5-variant_folder/augmented',
    output_alphabet=['a', 'b', 'c', 'd', 'e'],
    seed=42,
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
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.5, p=0.8),
        A.CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), p=0.6),
        # A.Equalize(p=0.3),
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.7),
    ]),
    input_path_list=[]
)
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
number = 1
config = AutoConfig.from_pretrained(args.janus_model_path)
language_config = config.language_config
language_config._attn_implementation = 'eager'
vl_gpt = AutoModelForCausalLM.from_pretrained(args.janus_model_path, language_config=language_config, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda() if args.device == 'cuda' else vl_gpt.to(torch.float16)
vl_chat_processor = VLChatProcessor.from_pretrained(args.janus_model_path)
tokenizer = vl_chat_processor.tokenizer

pipe1 = StableDiffusionDepth2ImgPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(args.device)
pipe1.enable_model_cpu_offload()

pipe2 = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to(args.device)
pipe2.enable_model_cpu_offload()

args.input_path_list = sorted(glob(args.augmented_folder + '/*'))
def sort_key(path):
    filename = os.path.basename(path)
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else float('inf')


args.input_path_list = sorted(args.input_path_list, key=sort_key)
args.input_path_list = args.input_path_list[number-1:]
print(args.input_path_list[0])
print(len(args.input_path_list))

for i, image_path in enumerate(args.input_path_list):
    filename = os.path.basename(image_path)

# Remove the prefix "augmented_image" and the suffix ".png"
    identifier = filename.replace("augmented_image", "").replace(".png", "")

    print(identifier)
    init_image = Image.open(image_path).convert("RGB")
    init_image_np = np.array(init_image)
    augmented_image2 = args.augment2(image=init_image_np)
    init_image2 = Image.fromarray(augmented_image2['image'])
    output_path = f"{args.output_path}/output_{identifier}.png"
    init_image2.save(output_path)
    
    answer = inference_utils.multimodal_understanding(
        image=image_path,
        question=args.prompt_for_understanding_image,
        seed=args.seed,
        top_p=0.8,
        temperature=1.0,
        vl_chat_processor=vl_chat_processor,
        vl_gpt=vl_gpt,
        tokenizer=tokenizer,
        cuda_device=args.device,
    )

    prompt = answer
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    strength, guidance_scale = inference_utils.__generate_strength(args.seed), inference_utils.__generate_guidance_scale(args.seed)
    image = pipe1(prompt=prompt, image=init_image, negative_prompt=args.negative_prompt, strength=strength, guidance_scale=guidance_scale, generator=generator).images[0]
    image.save(output_path)

    # Refine image using Stable Diffusion XL Refiner
    image = pipe2(prompt=prompt, prompt_2= 'Refine the image into 4k quality, fine edges, high definition, high resolution, finer details', image=image, requires_aesthetics_score=False, negative_prompt=args.negative_prompt).images[0]
    image.save(output_path)
    
    init_image = Image.open(output_path).convert("RGB")
    init_image_np = np.array(init_image)
    augmented_image2 = args.augment2(image=init_image_np)
    init_image2 = Image.fromarray(augmented_image2['image'])
    init_image2.save(output_path)
    
    # img = cv2.imread(f"{args.output_path}/output_{i+66}{args.output_alphabet[j]}.png")
    # if img is not None:
    #     adjusted_img = adjust_brightness(img)
    #     if adjusted_img is not None:
    #         cv2.imwrite(f"{args.output_path}/output_{i+66}{args.output_alphabet[j]}.png", adjusted_img)
    # img = cv2.imread(f"{args.output_path}/output_{identifier}.png")
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # # Increase saturation
    # h, s, v = cv2.split(hsv)
    # s = cv2.add(s, 30)  # Adjust this value for more/less intensity
    # s = np.clip(s, 0, 255)

    # # Merge back and convert to BGR
    # enhanced_hsv = cv2.merge((h, s, v))
    # final = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    # cv2.imwrite(f"{args.output_path}/output_{i+number}{args.output_alphabet[j]}.png", final)
    # img = cv2.imread(f"{args.output_path}/output_{i+number}{args.output_alphabet[j]}.png")
    # inc = np.array([15, 10, -5], dtype="int16")  # Adjust RGB values for warmth/coolness
    # balanced = np.clip(img + inc, 0, 255).astype(np.uint8)

    # cv2.imwrite(f"{args.output_path}/output_{i+number}{args.output_alphabet[j]}.png", balanced)     
print("Inference completed successfully!")                