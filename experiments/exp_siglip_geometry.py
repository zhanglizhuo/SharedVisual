"""SigLIP geometry analysis: fill the SigLIP gap in CKA + Mantel.

Why this exists: the paper rules out the shared-encoder hypothesis primarily via
SigLIP (rho=0.716 > CLIP group 0.482), but the geometric verification (CKA,
Mantel) in Step 3 omitted SigLIP. This script adds SigLIP to both analyses to
close that gap.

Outputs (to results/siglip_geometry.json):
  - SigLIP 8x8 class-centroid similarity matrix (50 imgs/class, same pipeline as
    exp2_class_similarity_matrices.py) -> merge into class_similarity_matrices.json
  - 5-encoder CKA matrix (CLIP x3 + QwenViT + SigLIP) on 160 imgs (20/class),
    same pipeline as exp2_mllm_cka.py

Run on V100 (dl_env):
  python experiments/exp_siglip_geometry.py
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"  # use cached weights; avoid mirror 404s

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
RESULTS_DIR = Path(os.environ.get("SHAREDVISUAL_RESULTS", str(Path(__file__).resolve().parents[1] / "results")))
SCB5_BASE = (Path(os.environ.get("SHAREDVISUAL_DATA", str(Path.home() / "data")))
             / "SCB5_TeacherBehavior"
             / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2"
             / "images" / "val")
N_MANTEL = 50
N_CKA = 20
SIGLIP_NAME = "ViT-SO400M-14-SigLIP-384"
SIGLIP_PRETRAINED = "webli"


def linear_cka(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    XXT = X @ X.T
    YYT = Y @ Y.T
    hsic = (XXT * YYT).sum()
    hsic_xx = (XXT * XXT).sum()
    hsic_yy = (YYT * YYT).sum()
    return (hsic / (hsic_xx.sqrt() * hsic_yy.sqrt())).item()


def class_similarity_matrix(feat_dict):
    classes_sorted = sorted(feat_dict.keys())
    centroids = []
    for c in classes_sorted:
        mu = np.array(feat_dict[c], dtype=np.float64).mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        centroids.append(mu)
    C = np.array(centroids)
    return classes_sorted, (C @ C.T).tolist()


def select_images(n_per_class):
    """Same selection logic as exp2_class_similarity_matrices.py."""
    with open(RESULTS_DIR / "scb5_true_class_mapping.json") as f:
        all_mapping = json.load(f)
    image_paths, img_to_label = [], {}
    counts = {c: 0 for c in CLASSES}
    for p, v in all_mapping.items():
        cls = v["true_class"]
        if cls not in CLASSES or counts[cls] >= n_per_class:
            continue
        img_path = str(SCB5_BASE / Path(p).name)
        if Path(img_path).exists():
            image_paths.append(img_path)
            img_to_label[img_path] = cls
            counts[cls] += 1
    return image_paths, img_to_label, counts


def extract_clip(name, backend, mname, pretrained, image_paths, device, batch=32):
    if backend == "clip":
        model, preprocess = clip.load(mname, device=device)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            mname, pretrained=pretrained, device=device)
    feats = []
    for i in range(0, len(image_paths), batch):
        batch_paths = image_paths[i:i + batch]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        t = torch.stack([preprocess(img) for img in images]).to(device)
        with torch.no_grad():
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    return torch.cat(feats)


def extract_qwen(image_paths, device):
    qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16,
        device_map="auto", attn_implementation="eager")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    qwen.eval()
    feats = []
    for p in tqdm(image_paths, desc="QwenViT"):
        img = Image.open(p).convert("RGB").resize((224, 224))
        conv = [{"role": "user", "content": [
            {"type": "image", "max_pixels": 224 * 224},
            {"type": "text", "text": "x"}]}]
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            out = qwen.visual(inputs["pixel_values"], inputs["image_grid_thw"])
            pooled = out.mean(dim=0, keepdim=True)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        feats.append(pooled.cpu())
    del qwen
    torch.cuda.empty_cache()
    return torch.cat(feats)


def extract_siglip(image_paths, device, batch=16):
    print("  Loading SigLIP ViT-SO400M-384 (open_clip, webli weights)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        SIGLIP_NAME, pretrained=SIGLIP_PRETRAINED, device=device)
    model.eval()
    feats = []
    for i in tqdm(range(0, len(image_paths), batch), desc="SigLIP"):
        batch_paths = image_paths[i:i + batch]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        t = torch.stack([preprocess(img) for img in images]).to(device)
        with torch.no_grad():
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(feats)


def main():
    print("=" * 60)
    print("SigLIP geometry: CKA + Mantel matrix (closes the SigLIP gap)")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Mantel pipeline (50/class) ---
    img50, lab50, cnt50 = select_images(N_MANTEL)
    print(f"Mantel: {len(img50)} images, per class: {cnt50}")

    # --- CKA pipeline: first 20/class, deterministic subset of the 50 ---
    img20, lab20, cnt20 = select_images(N_CKA)
    print(f"CKA:    {len(img20)} images, per class: {cnt20}")

    # Sanity: the 20/class set must be a prefix of the 50/class set
    assert set(img20).issubset(set(img50)), "CKA 20-set must be subset of Mantel 50-set"

    # === Extract features for Mantel (50/class, SigLIP only; CLIP/Qwen matrices exist) ===
    feats_mantel = {}
    print("\nExtracting SigLIP features (50/class for Mantel)...")
    siglip_50 = extract_siglip(img50, device)
    siglip_by_class = {c: [] for c in CLASSES}
    for p, v in zip(img50, siglip_50.numpy()):
        siglip_by_class[lab50[p]].append(v)
    feats_mantel["SigLIP (ViT-SO400M)"] = {c: np.array(v) for c, v in siglip_by_class.items()}

    siglip_classes, siglip_matrix = class_similarity_matrix(feats_mantel["SigLIP (ViT-SO400M)"])
    assert siglip_classes == sorted(CLASSES)

    # === Extract features for CKA (20/class, ALL 5 encoders to rebuild full matrix) ===
    feats_cka = {}
    print("\nExtracting features (20/class for CKA)...")
    feats_cka["CLIP ViT-L/14 (OpenAI)"] = extract_clip("CLIP ViT-L/14 (OpenAI)", "clip", "ViT-L/14", "openai", img20, device)
    feats_cka["CLIP ViT-L/14 (LAION)"] = extract_clip("CLIP ViT-L/14 (LAION)", "open_clip", "ViT-L-14", "laion2b_s32b_b82k", img20, device)
    feats_cka["CLIP ViT-B/32 (OpenAI)"] = extract_clip("CLIP ViT-B/32 (OpenAI)", "clip", "ViT-B/32", "openai", img20, device)
    feats_cka["Qwen2-VL Vision"] = extract_qwen(img20, device)
    feats_cka["SigLIP (ViT-SO400M)"] = extract_siglip(img20, device)

    # 5-encoder CKA matrix
    names = list(feats_cka.keys())
    cka_matrix = {}
    print("\n=== 5-encoder CKA matrix ===")
    for i, ni in enumerate(names):
        cka_matrix[ni] = {}
        row = []
        for j, nj in enumerate(names):
            if i == j:
                s = 1.0
            else:
                s = round(linear_cka(feats_cka[ni], feats_cka[nj]), 4)
            cka_matrix[ni][nj] = s
            row.append(f"{s:.4f}")
        print(f"  {ni[:25]:<25} " + "  ".join(f"{x:>7}" for x in row))

    out = {
        "description": "SigLIP geometry supplement; closes the SigLIP gap in CKA + Mantel",
        "siglip_model": f"open_clip {SIGLIP_NAME} ({SIGLIP_PRETRAINED})",
        "n_per_class_mantel": N_MANTEL,
        "n_per_class_cka": N_CKA,
        "class_names": sorted(CLASSES),
        "siglip_mantel_matrix": siglip_matrix,
        "cka_matrix": cka_matrix,
    }
    out_path = RESULTS_DIR / "siglip_geometry.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("\nNext (local): merge SigLIP matrix into class_similarity_matrices.json,")
    print("rerun analysis/mantel_test.py, update paper Step 3 + figures.")


if __name__ == "__main__":
    main()
