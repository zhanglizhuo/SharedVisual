#!/usr/bin/env python3
"""Debug: show raw model responses for a few SCB5 samples."""
import argparse, os, random
from pathlib import Path

import torch
from PIL import Image

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

CLASS_NAME_MAP = {
    "BowHead": "bowing head",
    "TurnHead": "turning head",
    "hand-raising": "raising hand",
    "On-stage interaction": "on stage interaction",
    "blackboard-writing": "writing on blackboard",
    "blackBoard": "standing near a blackboard or whiteboard",
    "stand": "standing at the front of a classroom",
    "guide": "guiding students in classroom",
    "answer": "answering questions in class",
}


def match_class_name(response, true_class, all_classes):
    resp = response.lower().strip().rstrip(".,;:!?\n\t ")
    true_lower = true_class.lower()
    if resp == true_lower:
        return True
    valid = list(dict.fromkeys([c.lower() for c in all_classes]
                               + [CLASS_NAME_MAP.get(c, c).lower() for c in all_classes]))
    matched = [v for v in valid if v in resp]
    if len(matched) == 1:
        return matched[0] == true_lower or CLASS_NAME_MAP.get(true_class, true_class).lower() == matched[0]
    elif len(matched) > 1:
        return (max(matched, key=len) == true_lower
                or CLASS_NAME_MAP.get(true_class, true_class).lower() == max(matched, key=len))
    return False


def build_prompt(classes):
    classes_str = ", ".join(CLASS_NAME_MAP.get(c, c) for c in classes)
    return ("Classify the classroom activity in this image. "
            f"Choose exactly one category from the list: {classes_str}. "
            "Output only the category name, nothing else.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", default="llava-hf/llava-1.5-13b-hf", nargs="?")
    ap.add_argument("--data-dir", default="/tmp/scb5")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    from transformers import LlavaForConditionalGeneration, AutoProcessor
    print(f"Loading {args.model}...", flush=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()

    # TeacherBehavior: 8 classes, hardest
    classes = ["guide", "answer", "On-stage interaction", "blackboard-writing",
               "teacher", "stand", "screen", "blackBoard"]
    subdir = "SCB5_TeacherBehavior/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2"
    img_dir = f"{args.data_dir}/{subdir}/images/val"
    lbl_dir = f"{args.data_dir}/{subdir}/labels/val"
    fnames = sorted(f for f in os.listdir(img_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
    random.seed(42)
    sample = random.sample(fnames, min(args.n, len(fnames)))

    prompt = build_prompt(classes)
    print(f"PROMPT: {prompt}\n", flush=True)

    for fn in sample:
        img = Image.open(os.path.join(img_dir, fn)).convert("RGB")
        lp = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")
        with open(lp) as f:
            cid = int(f.readline().split()[0])
        chat = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(chat, add_generation_prompt=True)
        inputs = processor(text=text, images=img, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                 temperature=None, top_p=None,
                                 pad_token_id=processor.tokenizer.eos_token_id)
        resp = processor.decode(out[0][inputs.input_ids.shape[1]:],
                                skip_special_tokens=True).strip()
        ok = match_class_name(resp, classes[cid], classes)
        print(f"img={fn} true={classes[cid]} ({cid}) match={ok}")
        print(f"  resp={resp!r}\n", flush=True)


if __name__ == "__main__":
    main()
