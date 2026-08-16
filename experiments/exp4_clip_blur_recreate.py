#!/usr/bin/env python3
"""Recreate the producer script behind results/exp4_clip_blur.json.

Context: exp4_clip_blur.json (CLIP per-class accuracy under Gaussian blur,
TeacherBehavior 8 classes) entered the repo in the first commit with NO
committed producer script (AUDIT_REPORT.md Finding 1). Its sigma=0 values
(blackboard-writing=45.3, answer=0.0, blackBoard=1.7, teacher=83.7) exactly
match the OpenAI ViT-L/14 `a photo of {class}` protocol used by
experiments/exp3_prompt_variants.py (exp3_prompt_variants_v2.json), so that
is the protocol reconstructed here.

This script REGENERATES exp4_clip_blur.json from raw val images using the
SAME blur pipeline as experiments/exp_perturbation_blur.py
(PIL ImageFilter.GaussianBlur(radius=sigma), JPEG quality=90) so the
perturbation-synchrony comparison (analysis/blur_synchrony.py)
uses one identical blur definition on both sides.

Two sample-selection regimes are evaluated so the legacy values can be
reproduced or explicitly disconfirmed:
  - "all" : every labeled val image (class-imbalanced, n=3240)
  - "50"  : first 50 images per class in sorted filename order (n=390,
            identical to collect_samples() in exp_perturbation_blur.py)

Usage (V100, dl_env): python3 experiments/exp4_clip_blur_recreate.py
"""
import json
import os
import torch
import clip
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
import io
import base64

DATA_ROOT = Path(os.environ.get(
    "SHAREDVISUAL_DATA",
    "/home/broadsense/works/lizhuo/AutoResearchClaw/datasets_scb"))
SCB5_BASE = (DATA_ROOT / "SCB5_TeacherBehavior"
             / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2")
IMG_DIR = SCB5_BASE / "images" / "val"
LBL_DIR = SCB5_BASE / "labels" / "val"
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get(
    "SHAREDVISUAL_RESULTS", str(REPO_ROOT / "results")))

CLASSES = ["guide", "answer", "On-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackBoard"]
SIGMAS = [0, 1, 2, 4, 8]
N_CAP = 50


def read_label(path):
    with open(path) as f:
        line = f.readline().strip()
    return int(line.split()[0]) if line else None


def collect_all():
    samples = []
    for fname in sorted(os.listdir(IMG_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lbl = LBL_DIR / (os.path.splitext(fname)[0] + ".txt")
        if not lbl.exists():
            continue
        cid = read_label(lbl)
        if cid is None or not (0 <= cid < len(CLASSES)):
            continue
        samples.append((str(IMG_DIR / fname), CLASSES[cid]))
    return samples


def collect_capped():
    counts = {c: 0 for c in CLASSES}
    samples = []
    for fname in sorted(os.listdir(IMG_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lbl = LBL_DIR / (os.path.splitext(fname)[0] + ".txt")
        if not lbl.exists():
            continue
        cid = read_label(lbl)
        if cid is None or not (0 <= cid < len(CLASSES)):
            continue
        cls = CLASSES[cid]
        if counts[cls] >= N_CAP:
            continue
        samples.append((str(IMG_DIR / fname), cls))
        counts[cls] += 1
    return samples


def blur_bytes(img_path, sigma):
    if sigma == 0:
        return open(img_path, "rb").read()
    img = Image.open(img_path).convert("RGB")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    buf = io.BytesIO()
    blurred.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def eval_sigma(model, preprocess, device, samples, sigma):
    correct = {c: 0 for c in CLASSES}
    total = {c: 0 for c in CLASSES}
    texts = clip.tokenize([f"a photo of {c}" for c in CLASSES]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    for img_path, cls in samples:
        data = blur_bytes(img_path, sigma)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = img_feat @ text_features.T
            pred = logits.argmax(dim=-1).item()
        total[cls] += 1
        if pred == CLASSES.index(cls):
            correct[cls] += 1
    return {c: (100.0 * correct[c] / total[c] if total[c] else 0.0)
            for c in CLASSES}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    all_samples = collect_all()
    capped = collect_capped()
    print(f"all={len(all_samples)} capped={len(capped)}")

    for regime, samples in [("all", all_samples), ("50", capped)]:
        print(f"\n=== regime={regime} ===")
        result = {}
        for s in SIGMAS:
            acc = eval_sigma(model, preprocess, device, samples, s)
            result[f"blur_sigma_{s}"] = {c: round(v, 1) for c, v in acc.items()}
            print(f"  sigma={s}: {result[f'blur_sigma_{s}']}")

        out = RESULTS_DIR / f"exp4_clip_blur_recreated_{regime}.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()