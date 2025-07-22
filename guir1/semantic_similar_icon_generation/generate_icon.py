from huggingface_hub import login

# Log in to Hugging Face Hub using the token stored in Colab secrets
INPUT_TOKEN=""
login(token=INPUT_TOKEN)

# --- SETUP ---
# !pip install diffusers transformers accelerate
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image
from IPython.display import display, HTML
import os, ast, torch

# --- PARAMETERS ---
metadata_file = './inspect_samples_filtered/metadata.txt'
original_samples_dir = './inspect_samples_filtered'
icon_dir = './gen_icons'
out_replaced_dir = './replaced_samples'
out_review_dir = './preview_review'

os.makedirs(icon_dir, exist_ok=True)
os.makedirs(out_replaced_dir, exist_ok=True)
os.makedirs(out_review_dir, exist_ok=True)

USE_SD3 = True
if not USE_SD3:
    
    # --- LOAD THE LATEST STABLE DIFFUSION (Use SDXL or SD3.5 if available) ---
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

else:
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

# --- FUNCTION: Replace region in img with icon ---
def replace_icon_in_img(img_path, bbox, icon_path):
    img = Image.open(img_path).convert('RGBA')
    icon = Image.open(icon_path).convert('RGBA')
    x1, y1, x2, y2 = bbox
    icon_resized = icon.resize((x2-x1, y2-y1), Image.LANCZOS)
    img.paste(icon_resized, (x1, y1), icon_resized)
    return img
# --- FUNCTION: Generate icon from prompt ---
def generate_icon(prompt, out_path, size=128):
    if not USE_SD3:
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
            prompt,
            device=pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )

        img = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=10,
            guidance_scale=7.5
        ).images[0]
    else:
        img = pipe(
            prompt,
            negative_prompt="",
            num_inference_steps=15,
            guidance_scale=7.0,
        ).images[0]
    img = img.resize((size, size), Image.LANCZOS)
    img.save(out_path)
    return img

# --- MAIN LOOP ---
with open(metadata_file, 'r') as f:
    for idx, line in enumerate(f):
        meta = ast.literal_eval(line.strip())
        instr = meta['instruction'].strip().lower()
        bbox = meta['gt_bbox']
        orig_img_path = os.path.join(original_samples_dir, f"sample_{meta['idx']}.jpg")
        icon_path = os.path.join(icon_dir, f"{instr.replace(' ', '_').replace('/', '_')}.png")
        if "/" in instr:
            instr = instr.replace('/', ' or ')
        # Generate icon if not exists
        if not os.path.exists(icon_path):
            
            prompt = f"{instr}, flat minimal vector UI icon, material design style, SVG, centered, 2d, high contrast, no background, no text, transparent background, simple and clean, for a modern app"
            icon_size = (max(1, bbox[2]-bbox[0]), max(1, bbox[3]-bbox[1])) if isinstance(bbox, list) and len(bbox) == 4 else (128, 128)
            print(f"[{idx}] Generating icon for: '{instr}' | Size: {icon_size}")
            generate_icon(prompt, icon_path, size=icon_size[0])
        else:
            print(f"[{idx}] Icon already exists for: '{instr}'")

        # Open and process images
        try:
            orig_img = Image.open(orig_img_path).convert('RGBA')
            # Parse bbox if needed
            if isinstance(bbox, list) and len(bbox) == 4:
                pass  # already good
            elif isinstance(bbox, str):
                import re
                bbox = list(map(int, re.findall(r'\d+', bbox)))
                if len(bbox) != 4:
                    print(f"[{idx}] Parse error for bbox: {bbox}")
                    continue
            else:
                print(f"[{idx}] Skipping due to invalid bbox: {bbox}")
                continue

            replaced_img = replace_icon_in_img(orig_img_path, bbox, icon_path)
        except Exception as e:
            print(f"[{idx}] Error processing image: {e}")
            continue

        # Save replaced image
        replaced_img_path = os.path.join(out_replaced_dir, f"replaced_{idx}.png")
        replaced_img.save(replaced_img_path)

        # Save preview (original + replaced side by side)
        preview = Image.new('RGBA', (orig_img.width + replaced_img.width, max(orig_img.height, replaced_img.height)), (255,255,255,255))
        preview.paste(orig_img, (0,0))
        preview.paste(replaced_img, (orig_img.width,0))
        preview_path = os.path.join(out_review_dir, f"preview_{idx}_{instr}.png")
        preview.save(preview_path)

        print(f"[{idx}] Processed '{instr}'. Saved replaced to {replaced_img_path}, preview to {preview_path}")

print("All done! Review your preview_review/ images on your local machine.")
