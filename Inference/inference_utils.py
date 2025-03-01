import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image

import numpy as np
import os
import time

import PIL
import torch
from torchvision import transforms
import diffusers
import transformers
import os
from jupyter_compare_view import compare
import random
# Compare images
from IPython.display import display

def multimodal_understanding(image, question, seed, top_p, temperature, vl_chat_processor, vl_gpt, tokenizer, cuda_device):
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
    
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(cuda_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)
    
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

def __remove_variables_for_memory(args):
    # Keep the args variable and delete all other global variables
    keep_variables = [name for name in globals().keys() if name == 'args']

    # Iterate over a copy of globals() to avoid modifying it while iterating
    for var in list(globals().keys()):
        # Keep only the args object
        if var not in keep_variables and not var.startswith('__'):
            del globals()[var]

    print("Cleared all global variables except: args")
    
def __visualize_depth(image_path):
    image = PIL.Image.open(image_path).convert("RGB")

    # Define transformations (without dtype)
    image_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])

    # Apply transformations
    image = image_transform(image)

    # Convert to float16 and add batch dimension
    image = image.to(dtype=torch.float16).unsqueeze(0).to("cuda")

    # Depth estimation pipeline (make sure `pipe` is defined)
    depth_map = pipe.depth_estimator(image).predicted_depth

    # Convert image back to PIL
    image_pil = transforms.ToPILImage()(image[0].cpu())

    # Normalize depth map
    depth_min = torch.amin(depth_map, dim=[0, 1, 2], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[0, 1, 2], keepdim=True)
    depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
    depth_map = depth_map[0, :, :]

    # Convert depth map to PIL
    depth_map_pil = transforms.ToPILImage()(depth_map.cpu())

    compare_result = compare(depth_map_pil, image_pil, cmap="gray", start_mode="horizontal", start_slider_pos=0.73)
    display(compare_result)

def __generate_strength(seed):
    import random
    random.seed(seed)
    return random.uniform(0.35, 0.55)

def __generate_guidance_scale(seed):
    import random
    random.seed(seed)
    return random.uniform(5, 9)

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