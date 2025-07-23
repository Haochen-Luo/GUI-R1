from huggingface_hub import login

# Log in to Hugging Face Hub using the token stored in Colab secrets
login(token='')
import os, ast, re, torch, time
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import pipeline

# Paths and folders
metadata_file = './inspect_samples_filtered/metadata.txt'
original_samples_dir = './inspect_samples_filtered'
output_root = './llava_batch_out'
icon_dir = os.path.join(output_root, 'gen_icons')
out_replaced_dir = os.path.join(output_root, 'replaced_samples')
out_review_dir = os.path.join(output_root, 'preview_review')
os.makedirs(icon_dir, exist_ok=True)
os.makedirs(out_replaced_dir, exist_ok=True)
os.makedirs(out_review_dir, exist_ok=True)
llava_judgments_file = os.path.join(out_review_dir, "llava_judgments.txt")

# Icon generation
USE_SD3 = True
if not USE_SD3:
    from diffusers import StableDiffusionXLPipeline
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    sd_pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to("cuda" if torch.cuda.is_available() else "cpu")
else:
    from diffusers import StableDiffusion3Pipeline
    sd_pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
    ).to("cuda")

def generate_icon(prompt, out_path, size=(128, 128)):
    # Now size is tuple (w, h) for true bbox!
    if not USE_SD3:
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = sd_pipe.encode_prompt(
            prompt,
            device=sd_pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )
        img = sd_pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=10,
            guidance_scale=7.5
        ).images[0]
    else:
        img = sd_pipe(
            prompt,
            negative_prompt="",
            num_inference_steps=15,
            guidance_scale=7.0,
        ).images[0]
    img = img.resize(size, Image.LANCZOS)
    img.save(out_path)
    return img

def replace_icon_in_img(img_path, bbox, icon_path):
    img = Image.open(img_path).convert('RGBA')
    icon = Image.open(icon_path).convert('RGBA')
    x1, y1, x2, y2 = bbox
    icon_resized = icon.resize((x2-x1, y2-y1), Image.LANCZOS)
    img.paste(icon_resized, (x1, y1), icon_resized)
    return img

# LLaVA batch judge (unchanged)
llava_pipe = pipeline(
    "image-text-to-text",
    model="llava-hf/llava-1.5-7b-hf",
    device="cuda",
    max_new_tokens=48
)
def batch_llava_judge(icon_paths, instructions):
    results = []
    for p, inst in zip(icon_paths, instructions):
        msg = [{
            "role": "user",
            "content": [
                {"type": "image", "path": p},
                {"type": "text", "text": f"Does icon somehow suggest '{inst}'? It might be a very high level or abstract way. Answer yes or no and explain."}
            ]
        }]
        out = llava_pipe(text=msg, max_new_tokens=48)
        try:
            answer = out[0]['generated_text'][-1]['content'].strip()
        except Exception as e:
            print("WARN: Unexpected llava output format", out)
            answer = ""
        print("LLaVA answer:", answer)
        is_good = answer.lower().startswith("yes")
        results.append((is_good, answer))
    return results

# --- Utility for reading llava_judgments.txt as a dict {idx: is_good} ---
def load_llava_judgments(judgments_file):
    idx2good = {}
    if not os.path.exists(judgments_file):
        return idx2good
    with open(judgments_file, "r") as f:
        for line in f:
            if '|' not in line: continue
            idx = int(line.split(']')[0][1:])
            answer = line.split('|', 1)[-1].strip().lower()
            idx2good[idx] = answer.startswith('yes')
    return idx2good


skip_first_judge = False
if os.path.exists(llava_judgments_file) and os.path.getsize(llava_judgments_file) > 0:
    skip_first_judge = True
    print("⚡ Detected existing llava_judgments.txt, will skip first LLaVA judge pass and just (re)generate icons for 'not good' ones.\n")
# --- MAIN LOOP ---
BATCH_SIZE = 8

while True:
    # Load all icon tasks
    icon_tasks = []
    with open(metadata_file, 'r') as f:
        for idx, line in enumerate(f):
            meta = ast.literal_eval(line.strip())
            instr = meta['instruction'].strip().lower()
            bbox = meta['gt_bbox']
            if isinstance(bbox, str):
                bbox = list(map(int, re.findall(r'\d+', bbox)))
            orig_img_path = meta["img_file"]
            icon_name = instr.replace(' ', '_').replace('/', '_')
            icon_path = os.path.join(icon_dir, f"{icon_name}.png")
            icon_tasks.append({
                'idx': idx,
                'instr': instr,
                'bbox': bbox,
                'orig_img_path': orig_img_path,
                'icon_path': icon_path,
                'icon_name': icon_name
            })

    # --- Load judgments, skip good ones ---
    idx2good = load_llava_judgments(llava_judgments_file)
    remaining_tasks = [t for t in icon_tasks if not idx2good.get(t['idx'], False)]

    print(f"\n⏳ {len(remaining_tasks)} icons left to improve ...")

    if not remaining_tasks:
        print("✅ All icons are good according to LLaVA!")
        break
    

        
    # --- 1. Generate icons only for "not good" ones ---
    for task in tqdm(remaining_tasks, desc="Icon generation"):
        bbox = task['bbox']
        icon_size = (max(1, bbox[2]-bbox[0]), max(1, bbox[3]-bbox[1]))
        # if not os.path.exists(task['icon_path']):
        prompt = f"{task['instr']}, flat minimal vector UI icon, material design style, SVG, centered, 2d, high contrast, no background, no text, transparent background, simple and clean, for a modern app"
        try:
            
            generate_icon(prompt, task['icon_path'], size=icon_size)
        except Exception as e:
            print(f"[{task['idx']}] Error generating icon: {e}")

        
    # --- 2. Judge with LLaVA only "not good" icons ---
    batch_records = []
    for batch_start in tqdm(range(0, len(remaining_tasks), BATCH_SIZE), desc="LLaVA batch review"):
        batch = remaining_tasks[batch_start:batch_start+BATCH_SIZE]
        batch_icon_paths = [t['icon_path'] for t in batch]
        batch_instructions = [t['instr'] for t in batch]
        mask_existing = [os.path.exists(p) for p in batch_icon_paths]
        batch_icon_paths = [p for p, ok in zip(batch_icon_paths, mask_existing) if ok]
        batch_instructions = [ins for ins, ok in zip(batch_instructions, mask_existing) if ok]
        real_batch = [t for t, ok in zip(batch, mask_existing) if ok]
        if not batch_icon_paths:
            continue
        results = batch_llava_judge(batch_icon_paths, batch_instructions)
        for task, (is_good, answer) in zip(real_batch, results):
            task['is_good'] = is_good
            task['vlm_explanation'] = answer
            batch_records.append(task)
            # --- Record the result for loop resumption ---
            with open(llava_judgments_file, "a") as jf:
                jf.write(f"[{task['idx']}] {task['instr']} | {answer}\n")
        time.sleep(1)  # be nice to the GPU (and web UI if running)

    # --- 3. Icon replacement and preview for "newly approved" icons ---
    # Only those just marked as good in this round
    for task in tqdm([t for t in batch_records if t.get('is_good', False)], desc="Replacement & preview"):
        try:
            bbox = task['bbox']
            orig_img_path = task['orig_img_path']
            icon_path = task['icon_path']
            orig_img = Image.open(orig_img_path).convert('RGBA')
            replaced_img = replace_icon_in_img(orig_img_path, bbox, icon_path)
            # Save replaced image
            replaced_img_path = os.path.join(out_replaced_dir, f"replaced_{task['idx']}.png")
            replaced_img.save(replaced_img_path)
            # Draw red rectangle on replaced image for preview
            preview_img = replaced_img.copy()
            draw = ImageDraw.Draw(preview_img)
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
            # Preview: original | (replaced + red box)
            preview = Image.new('RGBA', (orig_img.width + preview_img.width, max(orig_img.height, preview_img.height)), (255,255,255,255))
            preview.paste(orig_img, (0,0))
            preview.paste(preview_img, (orig_img.width,0))
            preview_path = os.path.join(out_review_dir, f"preview_{task['idx']}_{task['icon_name']}.png")
            preview.save(preview_path)
            print(f"[{task['idx']}] Done: replaced to {replaced_img_path}, preview to {preview_path}")
        except Exception as e:
            print(f"[{task['idx']}] Error in replacement: {e}")

    print("♻️ Looping again to catch any still-missing/failed icons...\n")

print("🎉 ALL ICONS PASSED LLaVA JUDGE! Review your icons and previews in:", output_root)
