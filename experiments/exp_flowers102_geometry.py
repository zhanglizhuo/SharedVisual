"""#5 Cross-domain geometry pilot: Flowers102 (102 fine-grained flower species).

Tests whether the cross-encoder class-neighborhood geometry convergence we
documented on SCB5 (classroom) and Stanford40 (human actions) generalizes to a
completely different visual domain: fine-grained flower classification. If
Mantel rho across encoders remains high (>0.85) on flowers, the
shared-geometry claim is strengthened as a general property of vision
representation learning rather than an artifact of two specific benchmarks.

Geometry-only (no MLLM eval): extract features from 5 encoders, build
102x102 class-centroid cosine-similarity matrices, run Mantel tests for all
C(5,2)=10 pairs. Uses 20 images/class for stable centroids (Flowers102 has
variable images/class; we cap at 20 for tractable runtime and to match the
SCB5 CKA setup).

Idempotent: per-encoder features cached to npz (skip if exists). Dataset
auto-downloaded via torchvision to DATA_DIR. Run on V100:
  python experiments/exp_flowers102_geometry.py
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr
import clip
import open_clip

REPO_ROOT = Path(os.environ.get(
    "SHAREDVISUAL_ROOT",
    str(Path(os.environ.get("SHAREDVISUAL_ROOT", str(Path(__file__).resolve().parents[1]))))))
RESULTS_DIR = Path(os.environ.get(
    "SHAREDVISUAL_RESULTS", str(REPO_ROOT / "results")))
DATA_DIR = Path(os.environ.get(
    "SHAREDVISUAL_DATA", "/tmp/flowers102_data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_PER_CLASS = 20
SIGLIP_NAME = "ViT-SO400M-14-SigLIP-384"
SIGLIP_PRETRAINED = "webli"
ENCODERS = [
    ("CLIP ViT-L/14 (OpenAI)", "clip", "ViT-L/14", "openai",
     RESULTS_DIR / "flowers_clip_l14_oa.npz"),
    ("CLIP ViT-L/14 (LAION)", "open_clip", "ViT-L-14", "laion2b_s32b_b82k",
     RESULTS_DIR / "flowers_clip_l14_laion.npz"),
    ("CLIP ViT-B/32 (OpenAI)", "clip", "ViT-B/32", "openai",
     RESULTS_DIR / "flowers_clip_b32_oa.npz"),
    ("SigLIP (ViT-SO400M)", "open_clip", SIGLIP_NAME, SIGLIP_PRETRAINED,
     RESULTS_DIR / "flowers_siglip.npz"),
    ("Qwen2-VL Vision", "qwen", None, None,
     RESULTS_DIR / "flowers_qwen.npz"),
]
OUT_PATH = RESULTS_DIR / "flowers102_geometry.json"


def load_flowers102():
    """Return (image_paths, labels) where labels are 0-101 class indices."""
    from torchvision.datasets import Flowers102
    # torchvision downloads to root; split='train'+'val'+'test' to maximize coverage
    paths, labels = [], []
    seen_count = {}
    for split in ["train", "val", "test"]:
        ds = Flowers102(root=str(DATA_DIR), split=split, download=True)
        for i in range(len(ds)):
            img, lbl = ds[i]
            # cap per class to N_PER_CLASS for tractable runtime
            if seen_count.get(lbl, 0) >= N_PER_CLASS:
                continue
            # save to a tmp path so feature extractors can load by path uniformly
            p = DATA_DIR / f"flowers_imgs" / f"{lbl:03d}_{seen_count.get(lbl,0):03d}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                img.convert("RGB").save(p, format="JPEG", quality=95)
            paths.append(str(p))
            labels.append(int(lbl))
            seen_count[lbl] = seen_count.get(lbl, 0) + 1
    try:
        class_names = Flowers102(root=str(DATA_DIR), split="train").classes
    except AttributeError:
        class_names = [f"flower_{i:03d}" for i in range(len(set(labels)))]
    print(f"Flowers102: {len(paths)} images, {len(set(labels))} classes, "
          f"{N_PER_CLASS}/class cap")
    return paths, labels, class_names


def extract_clip_features(encoder_spec, image_paths, device, batch=64):
    name, backend, mname, pretrained, cache_path = encoder_spec
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        print(f"  {name}: cache hit ({len(d.files)} entries)")
        return {k: d[k] for k in d.files}
    print(f"  {name}: extracting ...")
    if backend == "clip":
        model, preprocess = clip.load(mname, device=device)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            mname, pretrained=pretrained, device=device)
    feats = []
    for i in tqdm(range(0, len(image_paths), batch), desc=name[:25]):
        batch_paths = image_paths[i:i + batch]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        t = torch.stack([preprocess(img) for img in images]).to(device)
        with torch.no_grad():
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    np.savez(cache_path, feats=feats)
    print(f"  {name}: {feats.shape} -> {cache_path}")
    del model
    torch.cuda.empty_cache()
    return {"feats": feats}


def extract_qwen_features(encoder_spec, image_paths, device):
    name, _, _, _, cache_path = encoder_spec
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        print(f"  {name}: cache hit")
        return {"feats": d["feats"]}
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print(f"  {name}: extracting ...")
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
        feats.append(pooled.squeeze(0).cpu().numpy())
    feats = np.array(feats)
    np.savez(cache_path, feats=feats)
    print(f"  {name}: {feats.shape} -> {cache_path}")
    del qwen
    torch.cuda.empty_cache()
    return {"feats": feats}


def class_centroid_matrix(feats, labels, class_names):
    """Build N_classes x N_classes cosine similarity matrix of class centroids."""
    unique = sorted(set(labels))
    centroids = []
    for c in unique:
        idx = [i for i, l in enumerate(labels) if l == c]
        mu = feats[idx].mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        centroids.append(mu)
    C = np.array(centroids)
    return (C @ C.T).tolist(), unique


def mantel_spearman(mat_a, mat_b):
    """Spearman of upper triangles (no permutation here — done locally on full mantel_test.py)."""
    n = len(mat_a)
    iu = np.triu_indices(n, k=1)
    r, _ = spearmanr(np.array(mat_a)[iu], np.array(mat_b)[iu])
    return float(r)


def main():
    print("=" * 60)
    print("Flowers102 cross-domain geometry pilot (5 encoders, 102 classes)")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_paths, labels, class_names = load_flowers102()

    feats_by_enc = {}
    for enc in ENCODERS:
        name = enc[0]
        if enc[1] == "qwen":
            feats_by_enc[name] = extract_qwen_features(enc, image_paths, device)["feats"]
        else:
            feats_by_enc[name] = extract_clip_features(enc, image_paths, device)["feats"]

    # Build class-centroid similarity matrices
    print("\nBuilding class-centroid matrices ...")
    matrices = {}
    for name, feats in feats_by_enc.items():
        mat, unique = class_centroid_matrix(feats, labels, class_names)
        matrices[name] = mat
        print(f"  {name}: {len(unique)} classes")

    # Pairwise Mantel Spearman (permutation p-values computed locally by mantel_test.py)
    print("\n=== Cross-encoder Mantel rho (Flowers102) ===")
    names = list(matrices.keys())
    pair_rhos = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho = mantel_spearman(matrices[names[i]], matrices[names[j]])
            print(f"  {names[i][:25]:25s} vs {names[j][:25]:25s} rho={rho:.4f}")
            pair_rhos.append({"encoder_a": names[i], "encoder_b": names[j],
                              "mantel_rho": rho})

    out = {
        "dataset": "Flowers102 (torchvision, 102 fine-grained flower species)",
        "n_per_class_cap": N_PER_CLASS,
        "encoders": [e[0] for e in ENCODERS],
        "class_names": class_names,
        "matrices": matrices,
        "mantel_pair_rhos": pair_rhos,
        "note": ("Mantel rho values here are point estimates (upper-triangle "
                 "Spearman). For exact/random permutation p-values, the matrices "
                 "in this file can be consumed by a modified mantel_test.py."),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")
    print("\nNext (local): integrate into paper + run permutation p-values.")


if __name__ == "__main__":
    main()
