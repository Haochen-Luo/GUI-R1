import os
import json
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import ray
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import re
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from io import BytesIO
from datasets import load_dataset
from datasets import Dataset as hf_dataset
import random

ray.init()

MODEL_PATH = ""
SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=1024,
    stop_token_ids=[],
)
DATA_PATH = ""
MICRO_BATCH = 6

def extract_coord(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    try:
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            coord_match = re.search(bbox_pattern, content_answer)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        else:
            coord_pattern = r'\{.*\((\d+),\s*(\d+))\s*.*\}'
            coord_match = re.search(coord_pattern, content)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        return [0, 0, 0, 0], False
    except:
        return [0, 0, 0, 0], False

# ---- Perturbation functions ----
def perturb_icon(image, bbox, perturb_type):
    x1, y1, x2, y2 = map(int, bbox)
    icon_crop = image.crop((x1, y1, x2, y2))

    if perturb_type == "none":
        return image

    elif perturb_type == "blackdot":
        draw = ImageDraw.Draw(icon_crop)
        for _ in range(max(3, (x2-x1)//15)):
            cx = random.randint(0, x2-x1-1)
            cy = random.randint(0, y2-y1-1)
            r = max(1, (min(x2-x1, y2-y1)) // 8)
            draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(0,0,0,255))
        image.paste(icon_crop, (x1, y1))
        return image

    elif perturb_type == "flip":
        icon_crop = icon_crop.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)
        image.paste(icon_crop, (x1, y1))
        return image

    elif perturb_type == "blur":
        icon_crop = icon_crop.filter(ImageFilter.GaussianBlur(radius=3))
        image.paste(icon_crop, (x1, y1))
        return image

    elif perturb_type == "jitter":
        enhancer = ImageEnhance.Color(icon_crop)
        icon_crop = enhancer.enhance(random.uniform(0.4, 2.2))
        enhancer = ImageEnhance.Brightness(icon_crop)
        icon_crop = enhancer.enhance(random.uniform(0.6, 1.5))
        image.paste(icon_crop, (x1, y1))
        return image

    elif perturb_type == "occlude":
        overlay = Image.new('RGBA', icon_crop.size, (0,0,0,100))
        icon_crop = Image.alpha_composite(icon_crop.convert("RGBA"), overlay)
        image.paste(icon_crop, (x1, y1))
        return image

    elif perturb_type == "pixelate":
        w, h = icon_crop.size
        icon_crop = icon_crop.resize((max(1,w//6), max(1,h//6)), resample=Image.NEAREST)
        icon_crop = icon_crop.resize((w, h), resample=Image.NEAREST)
        image.paste(icon_crop, (x1, y1))
        return image

    else:
        return image

class MultiModalDataset(Dataset):
    def __init__(self, data, processor, perturb_type):
        self.data = data
        self.processor = processor
        self.perturb_type = perturb_type
        self.processor.max_pixels=2097152

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample["image"]
        image = Image.open(BytesIO(image["bytes"]))
        text = sample["instruction"]

        # -- If bbox available and icon perturbation requested --
        if "gt_bbox" in sample and self.perturb_type != "none":
            bbox = sample["gt_bbox"]
            if isinstance(bbox, str):
                import re
                bbox = list(map(int, re.findall(r'\d+', bbox)))
            if len(bbox) == 4:
                image = perturb_icon(image, bbox, self.perturb_type)
        # END PERTURBATION

        text=(
            f"You are RUN1-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{text}', with the action history being 'None'.\n"
            "Please provide the action to perform (enumerate from ['click']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
            "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
            "<think> ... </think> <answer>[{'action': enum[ 'click'], 'point': [x, y], 'input_text': 'no input text [default]'}]</answer>\n"
            "Example:\n"
            "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
        )
        text = '<image>\n' + text
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        resized_height = inputs['image_grid_thw'][0][1] * self.processor.image_processor.patch_size
        resized_width = inputs['image_grid_thw'][0][2] * self.processor.image_processor.patch_size
        origin_height = image_inputs[0].size[1]
        origin_width = image_inputs[0].size[0]
        scale_x = origin_width / resized_width
        scale_y = origin_height / resized_height

        del inputs

        sample["scale"]=[scale_x.item(),scale_y.item()]
        sample["image_size"]=[origin_width,origin_height]

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
            "original_sample": sample,
        }

def custom_collate_fn(batch):
    collated_batch = {
        "prompts": [],
        "multi_modal_data": [],
        "mm_processor_kwargs": [],
        "original_samples": [],
    }
    for item in batch:
        collated_batch["prompts"].append(item["prompt"])
        collated_batch["multi_modal_data"].append(item["multi_modal_data"])
        collated_batch["mm_processor_kwargs"].append(item["mm_processor_kwargs"])
        collated_batch["original_samples"].append(item["original_sample"])
    return collated_batch

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, model_path, sampling_params):
        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 1, "video": 1},
        )
        self.sampling_params = sampling_params

    def process_data(self, dataloader):
        results = []

        for batch in tqdm(dataloader):
            prompts = batch["prompts"]
            multi_modal_data = batch["multi_modal_data"]
            mm_processor_kwargs = batch["mm_processor_kwargs"]
            original_samples = batch["original_samples"]

            llm_inputs = [
                {
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": mm_kwargs,
                }
                for prompt, mm_data, mm_kwargs in zip(prompts, multi_modal_data, mm_processor_kwargs)
            ]
            outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
            for original_sample, output in zip(original_samples, outputs):
                generated_text = output.outputs[0].text
                gt_bbox = original_sample["gt_bbox"]
                original_sample["pred"] = generated_text
                pred_coord, _ = extract_coord(generated_text)
                original_sample["pred_coord"] = [pred_coord[0]*original_sample["scale"][0],pred_coord[1]*original_sample["scale"][1]]
                original_sample["scale"]=[]
                original_sample["image"]=''
                original_sample["perturb_type"] = args.perturb_type
                results.append(original_sample)
        return results

def main(args):
    MODEL_PATH = args.model_path
    DATA_PATH = args.data_path
    if DATA_PATH.endswith('parquet'):
        data=load_dataset("parquet", data_files=DATA_PATH, split="train")
    else:
        data = [json.loads(s) for s in open(DATA_PATH, "r")] if DATA_PATH.endswith(".jsonl") else json.load(open(DATA_PATH,"r"))
    OUTPUT_DIR = args.output_path
    num_actors = args.num_actor
    OUTPUT_DIR = os.path.join(OUTPUT_DIR,MODEL_PATH.split('/')[-1])
    basename = DATA_PATH.split("/")[-1]
    if basename.endswith(".jsonl"):
        basename = basename.replace(".jsonl", f"_pred_{args.perturb_type}.json")
    elif basename.endswith(".parquet"):
        basename = basename.replace(".parquet", f"_pred_{args.perturb_type}.json")
    else:
        basename = f"{basename}_pred_{args.perturb_type}.json"
    NEW_FILE = os.path.join(OUTPUT_DIR, basename)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_chunks = [hf_dataset.from_dict(data[i::num_actors]) for i in range(num_actors)]

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    workers = [Worker.remote(MODEL_PATH, SAMPLING_PARAMS) for _ in range(num_actors)]

    futures = []
    for i, chunk in enumerate(data_chunks):
        dataset = MultiModalDataset(chunk, processor, args.perturb_type)
        dataloader = DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
        futures.append(workers[i].process_data.remote(dataloader))

    all_results = ray.get(futures)
    with open(NEW_FILE, "w") as ans_file:
        for worker_results in all_results:
            for sample in worker_results:
                ans_file.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='<model_path>')
    parser.add_argument('--data_path', type=str, default="<data_path>")
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--num_actor', type=int, default=8)
    parser.add_argument('--perturb_type', type=str, default='none',
                        choices=['none', 'blackdot', 'flip', 'blur', 'jitter', 'occlude', 'pixelate'],
                        help='Type of icon perturbation to apply')
    args = parser.parse_args()
    main(args)
