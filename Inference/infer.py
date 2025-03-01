from argparse import Namespace
from glob import glob
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torchvision.transforms as transforms
import torchvision
import albumentations as A

args = Namespace(
    janus_model_path = 'deepseek-ai/Janus-Pro-1B',
    model_path = '/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/outputs',
    pretrained_model_path = 'stabilityai/stable-diffusion-2-depth',
    image_path = '/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/Input',
    input_path_list = [],
    output_path = '/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/Output',
    single_image_path = '/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/Input/input_7.png',
    augment1 = A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=0.9),  # Flip horizontally
            A.VerticalFlip(p=0.9)], p=0.7)
    ]),    # Flip vertically
    augment2 = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.8, contrast_limit=0.8, p=0.8),  # Adjust brightness and contrast
        A.CLAHE(clip_limit=1.0, tile_grid_size=(8,8), p=0.6),  # Apply Contrast Limited Adaptive Histogram Equalization
        A.Equalize(p=0.3),  # Apply histogram equalization
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5), # Randomly change brightness, contrast, and saturation
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),]),      # Adjust hue and saturation
    output_alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
    seed = [42, 123, 456, 789, 112, 10, 20, 30, 40, 50],
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    prompt_for_understanding_image = 'Analyze the depth information of the image and describe its structural details, 4k quality, foreground and background separation, and intricate textures. Emphasize the key depth-based elements that define the scene and generate a detailed textual description suitable for guiding image variation synthesis. Answer in 50 words',
    negative_prompt = "bad, deformed, ugly, bad anotomy, bad resolution, bad quality, bad asthetic, blurry",
    num_variations = 10,
    augmented_folder = '/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/Inference/5-variant_folder/augmented',
    generated_folder = '/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/Inference/5-variant_folder/generated',
    augmented_folder_list = [],
    generated_folder_list = [],
)

import inference_utils
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image
import numpy as np
import os
import time
from IPython.display import display
from urllib.request import urlopen

config = AutoConfig.from_pretrained(args.janus_model_path)
language_config = config.language_config
language_config._attn_implementation = 'eager'
vl_gpt = AutoModelForCausalLM.from_pretrained(args.janus_model_path,
                                             language_config=language_config,
                                             trust_remote_code=True)
if args.device == 'cuda':
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
else:
    vl_gpt = vl_gpt.to(torch.float16)

vl_chat_processor = VLChatProcessor.from_pretrained(args.janus_model_path)
tokenizer = vl_chat_processor.tokenizer

import torch
import requests
from PIL import Image
import random
from diffusers import StableDiffusionDepth2ImgPipeline, StableDiffusionUpscalePipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe1 = StableDiffusionDepth2ImgPipeline.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16,
).to(args.device)
pipe1.enable_model_cpu_offload()

# Load the pre-trained upscaler model from Stability AI
pipe2 = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16
).to("cuda")
pipe2.enable_model_cpu_offload()

pipe3 = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe3 = pipe3.to("cuda")
pipe3.enable_model_cpu_offload()

import re

args.input_path_list = sorted(glob(args.image_path + '/*'))
def sort_key(path):
    filename = os.path.basename(path)
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else float('inf')

args.input_path_list = sorted(args.input_path_list, key=sort_key)
args.input_path_list = args.input_path_list[65:]
print(args.input_path_list[0])
for i in range(len(args.input_path_list)):
    # print(args.input_path_list[i])
    answers = []

    for j in range(args.num_variations):
    # Load the image
        init_image = Image.open(args.input_path_list[i]).convert("RGB")
        init_image_np = np.array(init_image)
        
        # Get the original image dimensions
        input_width, input_height = init_image.size
        # Compute new dimensions
        new_width, new_height = input_width // 2, input_height // 2
        
        random.seed(42 + i)  # Modify seed to ensure different variations for the same image
        np.random.seed(42 + i)
        torch.manual_seed(42 + i)

        # Alternate between the two augmentation methods
        if j % 2 == 0:
            # First method (Augmentation 1 + Augmentation 2 + RandomCrop)
            # Apply the first augmentation (flip transformations)
            augmented_image = args.augment1(image=init_image_np)
            init_image = Image.fromarray(augmented_image['image'])

            # Apply the second set of augmentations (brightness, contrast, color)
            augmented_image = args.augment2(image=np.array(init_image))
            init_image = Image.fromarray(augmented_image['image'])
            
            # Apply random crop transformation
            augmented_image = A.RandomCrop(width=new_width, height=new_height, p=1.0)
            cropped_image = augmented_image(image=np.array(init_image))['image']

            # Convert back to PIL Image
            init_image2 = Image.fromarray(cropped_image)

        else:
            # Second method (Augmentation 1 + Augmentation 2 + Resize + Crop)
            # Apply the first augmentation (flip transformations)
            augmented_image1 = args.augment1(image=init_image_np)
            init_image1 = Image.fromarray(augmented_image1['image'])

            # Apply the second set of augmentations (brightness, contrast, color)
            augmented_image2 = args.augment2(image=np.array(init_image1))
            init_image2 = Image.fromarray(augmented_image2['image'])
            
            # Apply resize transformation
            resize_transform = A.Resize(width=new_width, height=new_height, p=1.0)
            init_image2 = resize_transform(image=np.array(init_image2))['image']  # Ensure this returns a dictionary and access 'image'
            
            # Apply random crop transformation
            crop_transform = A.RandomCrop(width=new_width, height=new_height, p=1.0)
            cropped_image = crop_transform(image=np.array(init_image2))['image']  # Same here, access 'image' in the dict

            # Convert back to PIL Image
            init_image2 = Image.fromarray(cropped_image)

        # Optionally, save or display the final image
        init_image2.save(f"{args.augmented_folder}/augmented_image{i}{args.output_alphabet[j]}.png")
        
    for j in range(args.num_variations):
        answer = inference_utils.multimodal_understanding(
            image = f"{args.augmented_folder}/augmented_image{i}{args.output_alphabet[j]}.png",
            question=args.prompt_for_understanding_image,
            seed = args.seed[j],
            top_p = 0.8,
            temperature = 1.0,
            vl_chat_processor = vl_chat_processor,
            vl_gpt = vl_gpt,
            tokenizer = tokenizer,
            cuda_device = args.device,
        )
        answers.append(answer)
        # print(answers)

    for j in range(args.num_variations):
        init_image2 = Image.open(f"{args.augmented_folder}/augmented_image{i}{args.output_alphabet[j]}.png").convert("RGB")
        prompt = answers[j]
        alphabet = args.output_alphabet[j]
        seed = args.seed[j]
        generator = torch.Generator(device=args.device).manual_seed(seed)
        strength, guidance_scale = inference_utils.__generate_strength(seed), inference_utils.__generate_guidance_scale(seed)
        # print(strength, guidance_scale)
        image = pipe1(prompt=prompt, image=init_image2, negative_prompt=args.negative_prompt, strength=strength, guidance_scale=guidance_scale, generator=generator).images[0]
        image.save(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png")
        image = Image.open(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png").convert("RGB")

        # Resize the image to half of the target resolution (if target is 2x upscaling)
        width, height = image.size
        image_resized = image.resize((width // 2, height // 2))
        # print(f'Resized image dimensions: {image_resized.size}')
        # Define your prompt
        # prompt = answers[j]
        # Upscale image by 2x using the Stable Diffusion model
        upscaled_image = pipe2(prompt=prompt, image=image_resized, guidance_scale=7.5, num_inference_steps = 30).images[0]
        upscaled_image = upscaled_image.resize((input_width, input_height))
        upscaled_image.save(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png")
        img = cv2.imread(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png")
        # Create a blurred version of the image
        blurred = cv2.GaussianBlur(img, (5,5), 0)

        # Apply unsharp masking
        sharp = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

        cv2.imwrite(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png", sharp)
        img = cv2.imread(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Increase saturation
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, 30)  # Adjust this value for more/less intensity
        s = np.clip(s, 0, 255)

        # Merge back and convert to BGR
        enhanced_hsv = cv2.merge((h, s, v))
        final = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png", final)
        
        img = cv2.imread(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png")
        inc = np.array([15, 10, -5], dtype="int16")  # Adjust RGB values for warmth/coolness
        balanced = np.clip(img + inc, 0, 255).astype(np.uint8)

        cv2.imwrite(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png", balanced)
        image = Image.open(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png").convert("RGB")
        prompt = answers[j]
        image = pipe3(prompt= prompt, image=image, requires_aesthetics_score = True).images[0]
        image.save(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png")
        img = cv2.imread(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png")
        # Create a blurred version of the image
        blurred = cv2.GaussianBlur(img, (5,5), 0)

        # Apply unsharp masking
        sharp = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

        cv2.imwrite(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png", sharp)
        img = cv2.imread(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Increase saturation
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, 30)  # Adjust this value for more/less intensity
        s = np.clip(s, 0, 255)

        # Merge back and convert to BGR
        enhanced_hsv = cv2.merge((h, s, v))
        final = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png", final)
        
        img = cv2.imread(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png")
        inc = np.array([15, 10, -5], dtype="int16")  # Adjust RGB values for warmth/coolness
        balanced = np.clip(img + inc, 0, 255).astype(np.uint8)

        cv2.imwrite(f"{args.output_path}/output_{i+1}{args.output_alphabet[j]}.png", balanced)

print("Inference completed successfully!")