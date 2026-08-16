"""Exp 6: Cross-dataset validation of shared class-confusability geometry.

Computes REAL 40x40 class similarity matrices on Stanford40 (9532 images, 40
action classes) using two independently-trained vision encoders:
  - CLIP ViT-L/14 (OpenAI)
  - Qwen2-VL-7B vision encoder (QwenViT)

For each encoder we build a per-class centroid (L2-normalized mean of image
features), then take cosine similarity between centroids to get a 40x40
inter-class geometry matrix. We report the Spearman correlation between the
upper triangles of the two matrices as the cross-encoder geometry agreement.

This validates that the shared class-neighborhood structure is not an artifact
of the SCB5 classroom dataset: two encoders trained on different data with
different objectives converge to similar class geometry on an unrelated dataset.

Reproducibility:
  - Dataset is downloaded once and cached under DATA_DIR.
  - Per-encoder features are cached to RESULTS_DIR/*.npz so the geometry /
    Spearman step can be rerun without re-extraction.
  - The Qwen2-VL processor performs a 2x2 patch merge, so the vision tower
    emits t*h*w/4 tokens per image; we pool exactly that many tokens per image
    (asserted against the tower output length to guard against silent
    truncation, which would otherwise yield empty mean -> NaN centroids).

Run on V100 (dl_env conda env):
  python experiments/exp6_stanford40_geometry.py
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths (V100 layout by default; override via env for other hosts)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(os.environ.get(
    "SHAREDVISUAL_ROOT",
    str(Path(__file__).resolve().parents[1]),
))
RESULTS_DIR = Path(os.environ.get(
    "SHAREDVISUAL_RESULTS",
    str(REPO_ROOT / "results"),
))
DATA_DIR = Path(os.environ.get(
    "SHAREDVISUAL_DATA",
    "/tmp/stanford40_data",
))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASET_URL = "http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip"
CLIP_CACHE = RESULTS_DIR / "exp6_clip_features.npz"
QWEN_CACHE = RESULTS_DIR / "exp6_qwen_features.npz"
OUT_PATH = RESULTS_DIR / "exp6_stanford40_geometry.json"

CLIP_BATCH = 128
QWEN_BATCH = 8  # Qwen2-VL vision tower is memory-heavy on V100 (32GB)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def ensure_dataset() -> list:
    img_dir = DATA_DIR / "JPEGImages"
    if not img_dir.is_dir():
        zip_path = DATA_DIR / "Stanford40_JPEGImages.zip"
        if not zip_path.exists():
            import urllib.request
            print(f"Downloading {DATASET_URL} ...")
            urllib.request.urlretrieve(DATASET_URL, zip_path)
        print("Extracting ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
    paths = sorted(img_dir.glob("*.jpg"))
    print(f"Found {len(paths)} images in {img_dir}")
    return paths


def parse_class(p: Path) -> str:
    return p.stem.rsplit("_", 1)[0].replace("_", " ")


# ---------------------------------------------------------------------------
# Feature caches
# ---------------------------------------------------------------------------
def load_cache(path: Path):
    if path.exists():
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}
    return None


def save_cache(path: Path, feat_dict: dict):
    np.savez(path, **{c: np.array(v) for c, v in feat_dict.items()})
    print(f"  cached {len(feat_dict)} classes -> {path}")


# ---------------------------------------------------------------------------
# Qwen2-VL preprocessing (chat-template based, same as exp2_mllm_cka)
# ---------------------------------------------------------------------------
def make_qwen_preprocess(processor, device):
    def preprocess(images):
        msgs = [{"role": "user",
                 "content": [{"type": "image", "image": img}]} for img in images]
        texts = [processor.apply_chat_template([m], tokenize=False) for m in msgs]
        inputs = processor(text=texts, images=images, padding=True,
                           return_tensors="pt")
        pv = inputs["pixel_values"].to(device, dtype=torch.float16)
        gt = inputs["image_grid_thw"].to(device)
        return pv, gt
    return preprocess


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------
def extract_clip(image_paths, classes):
    cached = load_cache(CLIP_CACHE)
    if cached is not None:
        print(f"  using cached CLIP features: {CLIP_CACHE}")
        return cached
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("  loading CLIP ViT-L/14 (open_clip, OpenAI weights) ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    feats = {c: [] for c in classes}
    for i in tqdm(range(0, len(image_paths), CLIP_BATCH), desc="CLIP"):
        batch = image_paths[i:i + CLIP_BATCH]
        images = [Image.open(p).convert("RGB") for p in batch]
        x = torch.stack([preprocess(im) for im in images]).to(device)
        with torch.no_grad():
            f = F.normalize(model.encode_image(x), dim=-1)
        for p, v in zip(batch, f.cpu().numpy()):
            feats[parse_class(p)].append(v)
    feats = {c: np.array(v) for c, v in feats.items()}
    save_cache(CLIP_CACHE, feats)
    return feats


def extract_qwen(image_paths, classes):
    cached = load_cache(QWEN_CACHE)
    if cached is not None:
        print(f"  using cached Qwen features: {QWEN_CACHE}")
        return cached
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("  loading Qwen2-VL-7B vision encoder ...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", local_files_only=True)
    vision = model.visual
    del model
    import gc; gc.collect(); torch.cuda.empty_cache()
    vision = vision.to(device).eval()
    preprocess = make_qwen_preprocess(processor, device)
    feats = {c: [] for c in classes}
    for i in tqdm(range(0, len(image_paths), QWEN_BATCH), desc="Qwen2-VL"):
        batch = image_paths[i:i + QWEN_BATCH]
        images = [Image.open(p).convert("RGB") for p in batch]
        pv, gt = preprocess(images)
        with torch.no_grad():
            out = vision(pv, gt)  # [total_tokens, dim]
            if hasattr(out, "last_hidden_state"):
                out = out.last_hidden_state
            splits, idx = [], 0
            for t, h, w in gt.cpu().numpy():
                # Processor merges 2x2 patches -> tower emits t*h*w/4 tokens.
                n = int(t * h * w) // 4
                splits.append(out[idx:idx + n].mean(dim=0))
                idx += n
            assert idx == out.shape[0], \
                f"patch count mismatch: consumed {idx} / produced {out.shape[0]}"
            f = F.normalize(torch.stack(splits), dim=-1)
        for p, v in zip(batch, f.cpu().numpy()):
            feats[parse_class(p)].append(v)
    feats = {c: np.array(v) for c, v in feats.items()}
    save_cache(QWEN_CACHE, feats)
    return feats


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def class_similarity_matrix(feat_dict):
    """Cosine sim between L2-normalized per-class centroids."""
    classes_sorted = sorted(feat_dict.keys())
    centroids = []
    for c in classes_sorted:
        mu = np.array(feat_dict[c], dtype=np.float64).mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        centroids.append(mu)
    C = np.array(centroids)
    sim = C @ C.T
    return classes_sorted, sim


def upper_triangle(mat):
    return mat[np.triu_indices_from(mat, k=1)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("Exp 6: Stanford40 cross-encoder class geometry")
    print("=" * 64)
    print(f"RESULTS_DIR = {RESULTS_DIR}")
    print(f"DATA_DIR    = {DATA_DIR}")

    image_paths = ensure_dataset()
    classes = sorted(set(parse_class(p) for p in image_paths))
    print(f"Classes: {len(classes)}")

    print("\n[1/3] Extracting CLIP features")
    clip_feats = extract_clip(image_paths, classes)
    print("\n[2/3] Extracting Qwen2-VL features")
    qwen_feats = extract_qwen(image_paths, classes)

    print("\n[3/3] Computing geometry")
    clip_classes, clip_sim = class_similarity_matrix(clip_feats)
    qwen_classes, qwen_sim = class_similarity_matrix(qwen_feats)
    assert clip_classes == qwen_classes

    rho, pval = spearmanr(upper_triangle(clip_sim), upper_triangle(qwen_sim))
    print(f"\nCLIP vs Qwen2-VL class geometry Spearman r = {rho:.4f}, p = {pval:.2e}")

    out = {
        "dataset": "Stanford40",
        "n_images_total": len(image_paths),
        "n_classes": len(classes),
        "class_names": clip_classes,
        "n_images_per_class": {c: int(len(clip_feats[c])) for c in classes},
        "encoders": ["CLIP_ViT_L_14_OpenAI", "Qwen2_VL_Vision"],
        "similarity_matrices": {
            "CLIP_ViT_L_14_OpenAI": clip_sim.tolist(),
            "Qwen2_VL_Vision": qwen_sim.tolist(),
        },
        "cross_encoder_comparison": [{
            "encoder_a": "CLIP_ViT_L_14_OpenAI",
            "encoder_b": "Qwen2_VL_Vision",
            "spearman_r": float(rho),
            "p_value": float(pval),
            "n_pairs": int(len(upper_triangle(clip_sim))),
        }],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {OUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
