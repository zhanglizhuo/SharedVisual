"""#2 Behavioral difficulty agreement across encoders on Flowers102.

V1 of the "universal visual difficulty" supplement: directly test whether
per-class difficulty (behavioral level) is shared across independently
trained vision encoders, rather than only the geometry (centroid matrices).

Design:
- 5 encoders by default: CLIP ViT-L/14 (OpenAI), CLIP ViT-L/14 (LAION),
  CLIP ViT-B/32 (OpenAI), SigLIP (ViT-SO400M-384), and a clean control
  ResNet-50 (ImageNet-1k only). Optional 6th: Qwen2-VL Vision (QwenViT)
  enable with SHAREDVISUAL_USE_QWEN=1 when the model is cached locally.
- All CLIP/SigLIP/ResNet via the same backends as the geometry pilot
  (official clip / open_clip / torchvision), with weights cached on the V100
  (no network access there; everything is offline).
- Features extracted once per encoder (cached to npz; reuses the Flowers102
  geometry-pilot cache files under RESULTS_DIR where present, so a prior
  geometry run is not recomputed).
- Per-class difficulty: 5-fold CV L2-regularized logistic probe (standard
  scaler), per-class accuracy averaged over the CV folds = difficulty.
- Output: per-class difficulty vectors per encoder, pairwise Spearman for
  all pairs with 10,000-permutation p-values (seed 42), plus overall
  accuracies (behavioral sanity check that probes do learn).

Idempotent: per-encoder features cached to npz (skip extraction if exists).
Dataset auto-downloaded via torchvision to DATA_DIR (same as geometry pilot).
Run on 5090 / V100:
  python experiments/exp_flowers102_difficulty.py
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(os.environ.get(
    "SHAREDVISUAL_ROOT", str(Path(__file__).resolve().parents[1])))
RESULTS_DIR = Path(os.environ.get("SHAREDVISUAL_RESULTS", str(REPO_ROOT / "results")))
DATA_DIR = Path(os.environ.get("SHAREDVISUAL_DATA", "/tmp/flowers102_data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_PER_CLASS = 20
SIGLIP_NAME = "ViT-SO400M-14-SigLIP-384"
SIGLIP_PRETRAINED = "webli"
QWEN_HF_ID = "Qwen/Qwen2-VL-7B-Instruct"
# Same encoders/cache files as exp_flowers102_geometry.py (+ ResNet-50 control).
# Backends mirror the geometry pilot: official `clip` package for OpenAI CLIP,
# open_clip for LAION/SigLIP (weights cached on the V100), torchvision ResNet.
ENCODERS = [
    ("CLIP ViT-L/14 (OpenAI)", "clip", "ViT-L/14", "openai",
     RESULTS_DIR / "flowers_clip_l14_oa.npz"),
    ("CLIP ViT-L/14 (LAION)", "open_clip", "ViT-L-14", "laion2b_s32b_b82k",
     RESULTS_DIR / "flowers_clip_l14_laion.npz"),
    ("CLIP ViT-B/32 (OpenAI)", "clip", "ViT-B/32", "openai",
     RESULTS_DIR / "flowers_clip_b32_oa.npz"),
    ("SigLIP (ViT-SO400M)", "open_clip", SIGLIP_NAME, SIGLIP_PRETRAINED,
     RESULTS_DIR / "flowers_siglip.npz"),
    ("ResNet-50 (ImageNet-1k)", "resnet50", None, None,
     RESULTS_DIR / "flowers_resnet50.npz"),
]
if os.environ.get("SHAREDVISUAL_USE_QWEN") == "1":
    ENCODERS.append(
        ("Qwen2-VL Vision", "qwen", None, None,
         RESULTS_DIR / "flowers_qwen.npz"))
N_FOLDS = 5
N_PERM = 10_000
RNG_SEED = 42
BATCH_SIZE = 64
OUT_PATH = RESULTS_DIR / "flowers102_difficulty.json"


def load_flowers102():
    """Return (image_paths, labels, class_names). Same as geometry pilot."""
    from torchvision.datasets import Flowers102
    paths, labels, seen_count = [], [], {}
    for split in ["train", "val", "test"]:
        ds = Flowers102(root=str(DATA_DIR), split=split, download=True)
        for i in range(len(ds)):
            img, lbl = ds[i]
            if seen_count.get(lbl, 0) >= N_PER_CLASS:
                continue
            p = DATA_DIR / "flowers_imgs" / f"{lbl:03d}_{seen_count.get(lbl, 0):03d}.jpg"
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
          f"{len(class_names)} names, {N_PER_CLASS}/class cap")
    return paths, labels, class_names


def _feats_from_cache(encoder_spec):
    name, _, _, _, cache_path = encoder_spec
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        feats = d["feats"]
        print(f"  {name}: cache hit ({feats.shape})")
        return feats
    return None


def load_features(encoder_spec, image_paths, device):
    feats = _feats_from_cache(encoder_spec)
    if feats is not None:
        return feats
    return extract_features(encoder_spec, image_paths, device, batch=BATCH_SIZE)


def extract_features(encoder_spec, image_paths, device, batch=BATCH_SIZE):
    """Extract one encoder's features; cache to npz. Call only on cache miss."""
    name, backend, mname, pretrained, cache_path = encoder_spec
    print(f"  {name}: extracting ...")
    if backend in ("clip", "open_clip"):
        if backend == "clip":
            import clip
            model, preprocess = clip.load(mname, device=device)
        else:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                mname, pretrained=pretrained, device=device)
        model.eval()
        feats = []
        with torch.no_grad():
            for i in range(0, len(image_paths), batch):
                images = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch]]
                t = torch.stack([preprocess(img) for img in images]).to(device)
                f = model.encode_image(t)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu().numpy())
        del model
    elif backend == "qwen":
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_HF_ID, torch_dtype=torch.float16,
            device_map="auto", attn_implementation="eager")
        processor = AutoProcessor.from_pretrained(QWEN_HF_ID)
        model.eval()
        feats = []
        for p in image_paths:
            img = Image.open(p).convert("RGB").resize((224, 224))
            conv = [{"role": "user", "content": [
                {"type": "image", "max_pixels": 224 * 224},
                {"type": "text", "text": "x"}]}]
            prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[img], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.visual(inputs["pixel_values"], inputs["image_grid_thw"])
                pooled = out.mean(dim=0, keepdim=True)
                pooled = pooled / pooled.norm(dim=-1, keepdim=True)
            feats.append(pooled.squeeze(0).cpu().numpy())
        feats = np.array(feats)
        del model, processor
    elif backend == "resnet50":
        import torchvision.models as models
        import torchvision.transforms as T
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
        preprocess = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # Penultimate avgpool features; fall back to logits if the torchvision
        # build refactored the ResNet internals (module-name changes).
        from torch.nn import Flatten
        seq = None
        for names in (
                [model.conv1, model.bn1, model.relu, model.maxpool,
                 model.layer1, model.layer2, model.layer3, model.layer4,
                 model.avgpool],
                [model.conv1, model.bn1, model.relu, model.maxpool,
                 model.layer1, model.layer2, model.layer3, model.layer4]):
            try:
                seq = torch.nn.Sequential(*names, Flatten())
                fb = "avgpool" if len(names) == 9 else "pool4"
                break
            except AttributeError:
                continue
        if seq is None:
            seq = torch.nn.Sequential(model, lambda x: x)
            fb = "logits"
        print(f"    resnet50 features: {fb}")
        feats = []
        with torch.no_grad():
            for i in range(0, len(image_paths), batch):
                images = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch]]
                t = torch.stack([preprocess(img) for img in images]).to(device)
                f = seq(t)
                f = f / torch.norm(f, dim=-1, keepdim=True)
                feats.append(f.cpu().numpy())
    else:
        raise ValueError(f"unknown backend {backend}")
    feats = feats if isinstance(feats, np.ndarray) else np.concatenate(feats, axis=0)
    np.savez(cache_path, feats=feats)
    print(f"    {feats.shape} -> {cache_path} ({backend})")
    torch.cuda.empty_cache()
    return feats


def linear_probe_per_class_accuracy(feats, labels, n_folds=N_FOLDS, seed=RNG_SEED):
    """L2-regularized logistic probe with 5-fold CV; mean per-class accuracy."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(labels)
    n_classes = max(y) + 1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    per_class_correct = np.zeros(n_classes, dtype=float)
    per_class_total = np.zeros(n_classes, dtype=float)
    fold_accs = []
    for tr_idx, va_idx in skf.split(feats, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(feats[tr_idx])
        Xva = scaler.transform(feats[va_idx])
        clf = LogisticRegression(C=0.05, max_iter=1500, solver="lbfgs")
        clf.fit(Xtr, y[tr_idx])
        pred = clf.predict(Xva)
        fold_correct = np.zeros(n_classes, dtype=float)
        fold_total = np.zeros(n_classes, dtype=float)
        for p, t in zip(pred, y[va_idx]):
            fold_correct[t] += int(p == t)
            fold_total[t] += 1
        per_class_correct += fold_correct
        per_class_total += fold_total
        fold_accs.append(fold_correct.sum() / max(fold_total.sum(), 1.0))
    print(f"    folds: {[f'{a:.3f}' for a in fold_accs]}")
    return per_class_correct / np.maximum(per_class_total, 1.0)


def permutation_spearman(a, b, n_perm=N_PERM, seed=RNG_SEED):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    rho, _ = spearmanr(a, b)
    cnt = 0
    b0 = b.copy()
    for _ in range(n_perm):
        r, _ = spearmanr(a, b0[rng.permutation(len(b))])
        if abs(r) >= abs(rho):
            cnt += 1
    return float(rho), (cnt + 1) / (n_perm + 1)


def main():
    print("=" * 60)
    print("Flowers102 behavioral-difficulty agreement "
          f"({len(ENCODERS)} encoders)")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    image_paths, labels, class_names = load_flowers102()

    feats_by_enc = {}
    for enc in ENCODERS:
        feats_by_enc[enc[0]] = load_features(enc, image_paths, device)

    print("\nLinear probes (5-fold CV) ...")
    per_class_acc = {}
    overall_acc = {}
    for name, feats in feats_by_enc.items():
        print(f"  {name}:")
        acc = linear_probe_per_class_accuracy(feats, labels)
        per_class_acc[name] = acc.tolist()
        overall_acc[name] = float(acc.mean())
        print(f"    overall acc: {overall_acc[name]:.3f}")

    print("\n=== Pairwise Spearman (per-class difficulty) ===")
    names = list(per_class_acc.keys())
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho, p = permutation_spearman(
                np.array(per_class_acc[names[i]]), np.array(per_class_acc[names[j]]))
            print(f"  {names[i][:28]:28s} vs {names[j][:28]:28s} "
                  f"rho={rho:.4f} p={p:.4g}")
            pairs.append({"encoder_a": names[i], "encoder_b": names[j],
                          "spearman_rho": rho, "p_perm10k": p})

    out = {
        "description": ("Behavioral per-class difficulty agreement on "
                        "Flowers102 across independently trained encoders "
                        "(incl. clean ImageNet-1k ResNet-50). Difficulty = "
                        "per-class accuracy of 5-fold CV L2 logistic probe."),
        "n_images": len(image_paths),
        "n_classes": len(set(labels)),
        "n_per_class": N_PER_CLASS,
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "pairwise_spearman": pairs,
        "permutation": f"{N_PERM} permutations, seed {RNG_SEED}",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()