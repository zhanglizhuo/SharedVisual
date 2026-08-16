#!/usr/bin/env python3
"""Exp 3 extension: prompt variants for the two remaining low-accuracy classes.

Review-round addition. The original prompt experiment (exp3_prompt_variants.py)
tested blackboard-writing, answer, and blackBoard; stand and screen are the two
other low-accuracy classes on the same benchmark and were not tested. This
script evaluates them with the identical protocol (OpenAI ViT-L/14, "a photo of
{variant}" vs the other 7 class names, argmax over 8 logits, val images of the
class).

Output: results/exp3_stand_screen_prompts.json
"""
import json
import os
from pathlib import Path

import torch
import clip
import numpy as np
from PIL import Image

CLASSES = ["guide", "answer", "On-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackBoard"]

TARGETS = {
    "stand": [
        "stand",
        "a person standing",
        "a teacher standing in front of the classroom",
        "standing in class",
        "a student standing at attention",
    ],
    "screen": [
        "screen",
        "a screen in a classroom",
        "a display screen showing slides",
        "a projection screen in class",
        "a screen on the classroom wall",
    ],
}

REPO_ROOT = Path(os.environ.get(
    "SHAREDVISUAL_ROOT",
    str(Path(__file__).resolve().parents[1]),
))
RESULTS_DIR = Path(os.environ.get(
    "SHAREDVISUAL_RESULTS",
    str(REPO_ROOT / "results"),
))
with open(RESULTS_DIR / "scb5_true_class_mapping.json") as f:
    all_mapping = json.load(f)

dataset_base = Path(os.environ.get(
    "SHAREDVISUAL_SCB5_TEACHER",
    str(Path(os.environ.get("SHAREDVISUAL_DATA", str(Path.home() / "data")))
        / "SCB5_TeacherBehavior"
        / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2"
        / "images" / "val"),
))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model, preprocess = clip.load("ViT-L/14", device=device)

results = {}
for class_name, variants in TARGETS.items():
    print(f"\n=== {class_name} ===")
    image_paths = [dataset_base / Path(p).name for p, v in all_mapping.items()
                   if v["true_class"] == class_name]
    images = [Image.open(p).convert("RGB") for p in image_paths if p.exists()]
    print(f"  Images: {len(images)}")

    cls_results = {}
    for variant in variants:
        texts = [f"a photo of {variant}"] + [f"a photo of {c}" for c in CLASSES if c != class_name]
        text_tokens = clip.tokenize(texts).to(device)

        correct = 0
        probs_all = []
        for i in range(0, len(images), 32):
            batch = images[i:i+32]
            image_tensors = torch.stack([preprocess(img) for img in batch]).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_tensors)
                text_features = model.encode_text(text_tokens)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.T
                preds = logits.argmax(dim=-1)
                correct += (preds == 0).sum().item()
                probs_all.append(logits.softmax(dim=-1)[:, 0].cpu())

        acc = correct / len(images) * 100 if images else 0
        all_probs = torch.cat(probs_all).numpy() if probs_all else np.array([])
        mean_prob = float(all_probs.mean()) if len(all_probs) > 0 else 0

        cls_results[variant] = {"accuracy": round(acc, 1), "mean_prob": round(mean_prob, 3)}
        print(f"  prompt='{variant[:50]:50s}'  acc={acc:.1f}%  mean_prob={mean_prob:.3f}")

    results[class_name] = cls_results

out_path = RESULTS_DIR / "exp3_stand_screen_prompts.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
