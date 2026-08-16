#!/usr/bin/env python3
"""
Exp 3 v2: Semantic vs. Visual Confusion — Proper evaluation.
Replicates the AnchorScore evaluation method: prompt variant vs original 7 other classes.
"""
import json
import torch
import clip
import numpy as np
from pathlib import Path
from PIL import Image

CLASSES = ["guide", "answer", "On-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackBoard"]

DIVERGENT = {
    "blackboard-writing": {
        "variants": [
            "blackboard-writing",
            "a blackboard with writing on it",
            "writing on a blackboard",
            "a person writing on a blackboard",
            "a teacher writing on the board",
        ],
    },
    "answer": {
        "variants": [
            "answer",
            "a student answering a question",
            "a student raising hand to answer",
            "a student speaking in class",
            "answering a question",
        ],
    },
    "blackBoard": {
        "variants": [
            "blackBoard",
            "a person standing near a blackboard",
            "presenting at a blackboard",
            "standing by the board",
            "a teacher at the blackboard",
        ],
    },
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
for class_name, info in DIVERGENT.items():
    print(f"\n=== {class_name} ===")
    image_paths = [dataset_base / Path(p).name for p, v in all_mapping.items() if v["true_class"] == class_name]
    images = [Image.open(p).convert("RGB") for p in image_paths if p.exists()]
    print(f"  Images: {len(images)}")

    cls_results = {}
    for variant in info["variants"]:
        # Evaluate against all 8 classes (variant replaces the original class name)
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
                probs = logits.softmax(dim=-1)
                probs_all.append(probs[:, 0].cpu())

        acc = correct / len(images) * 100 if images else 0
        all_probs = torch.cat(probs_all).numpy() if probs_all else np.array([])
        mean_prob = float(all_probs.mean()) if len(all_probs) > 0 else 0

        cls_results[variant] = {"accuracy": round(acc, 1), "mean_prob": round(mean_prob, 3)}
        print(f"  prompt='{variant[:50]:50s}'  acc={acc:.1f}%  mean_prob={mean_prob:.3f}")

    results[class_name] = cls_results

out_path = RESULTS_DIR / "exp3_prompt_variants_v2.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
