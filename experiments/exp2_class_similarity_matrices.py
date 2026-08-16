"""Exp 2 (supplement): SCB5 8x8 class-similarity matrices for the Mantel test.

Computes per-class centroid cosine-similarity matrices on the SCB5
TeacherBehavior images for the four vision encoders used in the paper:
  - CLIP ViT-L/14 (OpenAI)
  - CLIP ViT-L/14 (LAION)
  - CLIP ViT-B/32 (OpenAI)
  - Qwen2-VL-7B vision encoder (QwenViT)

For each encoder we build per-class centroids (L2-normalized mean of image
features) and take cosine similarity between centroids, producing the 8x8
matrices consumed by analysis/mantel_test.py (exact Mantel
test over all 8! = 40,320 class relabelings).

Reproducibility note: the committed results/
class_similarity_matrices.json was generated during the experiment phase
with N_PER_CLASS=50 (see experiments/analyze_class_geometry.py, which
reports the same 50-image/class setup in class_geometry_analysis.json). This
script is the canonical pipeline that regenerates the matrices; running it
may yield slightly different entries than the committed file if the cached
feature extraction differs (image subset, Qwen resize), but it produces an
independently valid matrix set for the same encoders and classes.

Run on V100 (dl_env conda env):
  python experiments/exp2_class_similarity_matrices.py
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
import clip
import open_clip
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

CLASSES = ["guide", "answer", "On-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackBoard"]
REPO_ROOT = Path(os.environ.get(
    "SHAREDVISUAL_ROOT",
    str(Path(__file__).resolve().parents[1]),
))
RESULTS_DIR = Path(os.environ.get(
    "SHAREDVISUAL_RESULTS",
    str(REPO_ROOT / "results"),
))
SCB5_BASE = Path(os.environ.get(
    "SHAREDVISUAL_SCB5_TEACHER",
    str(Path(os.environ.get("SHAREDVISUAL_DATA", str(Path.home() / "data")))
        / "SCB5_TeacherBehavior"
        / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2"
        / "images" / "val"),
))
N_PER_CLASS = 50


def class_similarity_matrix(feat_dict):
    classes_sorted = sorted(feat_dict.keys())
    centroids = []
    for c in classes_sorted:
        mu = np.array(feat_dict[c], dtype=np.float64).mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        centroids.append(mu)
    C = np.array(centroids)
    return classes_sorted, (C @ C.T).tolist()


def main():
    print("=" * 60)
    print("Exp 2 supplement: SCB5 8x8 class-similarity matrices (4 encoders)")
    print("=" * 60)

    device = "cuda"

    with open(RESULTS_DIR / "scb5_true_class_mapping.json") as f:
        all_mapping = json.load(f)

    image_paths = []
    img_to_label = {}
    counts = {c: 0 for c in CLASSES}
    for p, v in all_mapping.items():
        cls = v["true_class"]
        if cls not in CLASSES or counts[cls] >= N_PER_CLASS:
            continue
        img_path = str(SCB5_BASE / Path(p).name)
        if Path(img_path).exists():
            image_paths.append(img_path)
            img_to_label[img_path] = cls
            counts[cls] += 1

    print("Images:", len(image_paths), "per class:", dict(counts))

    feats = {}

    print("\nExtracting CLIP features...")
    clip_models = [
        ("CLIP ViT-L/14 (OpenAI)", "clip", "ViT-L/14", "openai"),
        ("CLIP ViT-L/14 (LAION)", "open_clip", "ViT-L-14", "laion2b_s32b_b82k"),
        ("CLIP ViT-B/32 (OpenAI)", "clip", "ViT-B/32", "openai"),
    ]
    for name, backend, mname, pretrained in clip_models:
        print("  Loading %s..." % name)
        if backend == "clip":
            model, preprocess = clip.load(mname, device=device)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                mname, pretrained=pretrained, device=device)
        feats[name] = {c: [] for c in CLASSES}
        for i in range(0, len(image_paths), 32):
            batch_paths = image_paths[i:i + 32]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            image_tensors = torch.stack([preprocess(img) for img in images]).to(device)
            with torch.no_grad():
                f = model.encode_image(image_tensors)
                f = f / f.norm(dim=-1, keepdim=True)
            for p, v in zip(batch_paths, f.cpu().numpy()):
                feats[name][img_to_label[p]].append(v)
        feats[name] = {c: np.array(v) for c, v in feats[name].items()}

    print("\nLoading Qwen2-VL-7B...")
    qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    qwen.eval()

    print("Extracting Qwen2-VL vision features...")
    feats["Qwen2-VL Vision"] = {c: [] for c in CLASSES}
    for p in tqdm(image_paths):
        img = Image.open(p).convert("RGB")
        img = img.resize((224, 224))
        conv = [{"role": "user", "content": [
            {"type": "image", "max_pixels": 224 * 224},
            {"type": "text", "text": "x"},
        ]}]
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            vision_out = qwen.visual(inputs["pixel_values"], inputs["image_grid_thw"])
            pooled = vision_out.mean(dim=0, keepdim=True)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        feats["Qwen2-VL Vision"][img_to_label[str(p)]].append(pooled.squeeze(0).cpu().numpy())
    feats["Qwen2-VL Vision"] = {c: np.array(v) for c, v in feats["Qwen2-VL Vision"].items()}

    del qwen
    torch.cuda.empty_cache()

    print("\nComputing class-similarity matrices...")
    matrices = {}
    for name, feat_dict in feats.items():
        classes_sorted, sim = class_similarity_matrix(feat_dict)
        assert classes_sorted == sorted(CLASSES)
        matrices[name] = sim
        print(f"  {name}: 8x8 matrix computed")

    out = {
        "n_classes": 8,
        "n_per_class": N_PER_CLASS,
        "class_names": sorted(CLASSES),
        "matrices": matrices,
    }
    out_path = RESULTS_DIR / "class_similarity_matrices.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("Next: python analysis/mantel_test.py")


if __name__ == "__main__":
    main()
