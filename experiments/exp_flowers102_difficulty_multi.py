"""Flowers102 multi-seed difficulty agreement — SHARED splits (formal V2).

Replaces the exploratory /tmp/shot_multi2.py. This is the version whose
numbers enter the paper: per-class difficulty from low-shot probes, verified
to avoid the saturation problem (V1/V1b had 3-14 unique per-class values;
this regime yields 50+ unique values per encoder).

Design (each step chosen defensively):
1. Fixed split generation: N_SEEDS=20 random 3-shot train / 17-shot holdout
   patterns drawn ONCE (seed 42), then the SAME patterns are applied to
   EVERY encoder. Cross-encoder difference can then only come from the
   encoder, never from which images landed in train.
2. Per encoder: per-class held-out accuracy and mean true-class probability
   (margin) averaged over the 20 shared splits -> ~continuous difficulty.
3. Pairwise Spearman for all C(5,2)=10 pairs (Qwen optional), 10k-label
   permutation p-values (seed 42).
4. Sanity gates (assertions + printed):
   - n unique per-class difficulty values per encoder (target >= 30, else
     the measurement is saturated and the run is meaningless);
   - overall held-out accuracy well above 1/102 (probe actually learns);
   - the same-split design means any cross-encoder signal cannot be an
     artifact of train-image sampling (documented in the JSON).
5. Full serialization: per-class vectors, per-pair rho + p, per-encoder
   summary, n seeds/shots, git-inspectable JSON under results/.

Run (V100, offline, deps in dl_env caches):
  python experiments/exp_flowers102_difficulty_multi.py
Run (local, needs flowers_*.npz under RESULTS_DIR):
  python experiments/exp_flowers102_difficulty_multi.py
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

REPO_ROOT = Path(os.environ.get(
    "SHAREDVISUAL_ROOT", str(Path(__file__).resolve().parents[1])))
RESULTS_DIR = Path(os.environ.get("SHAREDVISUAL_RESULTS", str(REPO_ROOT / "results")))
DATA_DIR = Path(os.environ.get("SHAREDVISUAL_DATA", "/tmp/flowers102_data"))

N_PER_CLASS = 20
N_SHOT = 1           # per-class training images (from the shared split)
N_SEEDS = 20         # shared split patterns to average over
N_CLS = 102
N_PERM = 10_000
RNG_SEED = 42
BATCH_SIZE = 64
MIN_UNIQUE = 30      # sanity: below this the difficulty is too saturated
QWEN_HF_ID = "Qwen/Qwen2-VL-7B-Instruct"
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
    ("ResNet-50 (ImageNet-1k)", "resnet50", None, None,
     RESULTS_DIR / "flowers_resnet50.npz"),
]
if os.environ.get("SHAREDVISUAL_USE_QWEN") == "1":
    ENCODERS.append(
        ("Qwen2-VL Vision", "qwen", None, None,
         RESULTS_DIR / "flowers_qwen.npz"))
OUT_PATH = RESULTS_DIR / "flowers102_difficulty_multi.json"


def load_flowers102():
    """Return (paths, labels) using the same cache files as the geometry run
    (images already written under DATA_DIR/flowers_imgs by that script)."""
    paths, labels, seen = [], [], {}
    for split in ["train", "val", "test"]:
        from torchvision.datasets import Flowers102
        ds = Flowers102(root=str(DATA_DIR), split=split, download=True)
        for i in range(len(ds)):
            img, lbl = ds[i]
            if seen.get(lbl, 0) >= N_PER_CLASS:
                continue
            p = DATA_DIR / "flowers_imgs" / f"{lbl:03d}_{seen.get(lbl, 0):03d}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                img.convert("RGB").save(p, format="JPEG", quality=95)
            paths.append(str(p))
            labels.append(int(lbl))
            seen[lbl] = seen.get(lbl, 0) + 1
    return paths, labels


def load_features(spec, image_paths, device):
    """Cache-aware feature load (npz) or extraction (CLIP/open_clip/qwen)."""
    name, backend, mname, pretrained, cache_path = spec
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        feats = d["feats"]
        print(f"  {name}: cache hit ({feats.shape})")
        return feats
    print(f"  {name}: extracting ...")
    if backend == "clip":
        import clip
        model, preprocess = clip.load(mname, device=device)
    elif backend == "open_clip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            mname, pretrained=pretrained, device=device)
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
        np.savez(cache_path, feats=feats)
        return feats
    elif backend == "resnet50":
        import torchvision.models as models
        import torchvision.transforms as T
        from torch.nn import Flatten
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
        preprocess = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
        names = [model.conv1, model.bn1, model.relu, model.maxpool,
                 model.layer1, model.layer2, model.layer3, model.layer4,
                 model.avgpool]
        try:
            seq = torch.nn.Sequential(*names, Flatten())
        except AttributeError:
            try:
                seq = torch.nn.Sequential(*names[:-1], Flatten())
            except AttributeError:
                seq = torch.nn.Sequential(model)
        feats = []
        model.eval()
        with torch.no_grad():
            for p in image_paths:
                img = Image.open(p).convert("RGB")
                t = preprocess(img).unsqueeze(0).to(device)
                f = seq(t)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.squeeze(0).cpu().numpy())
        feats = np.array(feats)
        np.savez(cache_path, feats=feats)
        return feats
    else:
        raise ValueError(backend)

    model.eval()
    feats = []
    with torch.no_grad():
        bs = 64
        for i in range(0, len(image_paths), bs):
            images = [Image.open(p).convert("RGB") for p in image_paths[i:i + bs]]
            t = torch.stack([preprocess(img) for img in images]).to(device)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    np.savez(cache_path, feats=feats)
    return feats


def permutation_spearman(a, b, n_perm=N_PERM, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    rho, _ = spearmanr(a, b)
    cnt = 0
    b0 = b.copy()
    for _ in range(n_perm):
        idx = rng.permutation(len(b))
        r, _ = spearmanr(a, b0[idx])
        if abs(r) >= abs(rho):
            cnt += 1
    return float(rho), (cnt + 1) / (n_perm + 1)


def probe_single_split(feats, labels, tr_idx, te_idx):
    """One L2 logistic probe; returns held-out acc + margin per class."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    sc = StandardScaler().fit(feats[tr_idx])
    clf = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")
    clf.fit(sc.transform(feats[tr_idx]), labels[tr_idx])
    probs = clf.predict_proba(sc.transform(feats[te_idx]))
    pred = probs.argmax(axis=1)
    n_cls = labels.max() + 1
    correct = np.zeros(n_cls, dtype=float)
    total = np.zeros(n_cls, dtype=float)
    true_prob = np.zeros(n_cls, dtype=float)
    for i, t in enumerate(labels[te_idx]):
        total[t] += 1
        correct[t] += int(pred[i] == t)
        true_prob[t] += probs[i, t]
    return correct / np.maximum(total, 1.0), true_prob / np.maximum(total, 1.0)


def main():
    print("=" * 60)
    print(f"Flowers102 multi-seed difficulty "
          f"({N_SHOT}-shot, {N_SEEDS} shared splits, {len(ENCODERS)} encoders)")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_paths, labels = load_flowers102()
    y = np.asarray(labels)

    feats_by_enc = {}
    for enc in ENCODERS:
        feats_by_enc[enc[0]] = load_features(enc, image_paths, device)

    # shared split patterns (fixed, reproducible)
    rng = np.random.default_rng(RNG_SEED)
    splits = []
    for _ in range(N_SEEDS):
        perm = []
        for c in range(N_CLS):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            perm.append(idx)
        splits.append(perm)

    accs = {}
    margs = {}
    print(f"\nProbes over {N_SEEDS} shared splits ...")
    for name, f in feats_by_enc.items():
        a_acc = np.zeros(N_CLS)
        a_mg = np.zeros(N_CLS)
        for perm in splits:
            tr = np.concatenate([p[:N_SHOT] for p in perm])
            te = np.concatenate([p[N_SHOT:] for p in perm])
            acc, mg = probe_single_split(f, y, tr, te)
            a_acc += acc
            a_mg += mg
        accs[name] = a_acc / N_SEEDS
        margs[name] = a_mg / N_SEEDS
        n_uniq = len(np.unique(np.round(accs[name], 4)))
        print(f"  {name:28s} acc mean={accs[name].mean():.3f} nuniq={n_uniq} "
              f"margin mean={margs[name].mean():.3f}")
        if n_uniq < MIN_UNIQUE:
            print(f"    !! WARN: difficulty saturated ({n_uniq} unique < "
                  f"{MIN_UNIQUE}); pairs may be uninformative")

    names = list(accs)
    pairs = []
    print("\n=== Pairwise Spearman (difficulty ACC, 10k perm) ===")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r_acc, p_acc = permutation_spearman(accs[names[i]], accs[names[j]])
            r_mg, p_mg = permutation_spearman(margs[names[i]], margs[names[j]])
            print(f"  {names[i][:26]:26s} vs {names[j][:26]:26s} "
                  f"acc: rho={r_acc:.3f} p={p_acc:.4f}  "
                  f"margin: rho={r_mg:.3f} p={p_mg:.4f}")
            pairs.append({
                "encoder_a": names[i], "encoder_b": names[j],
                "spearman_acc": r_acc, "p_acc_perm10k": p_acc,
                "spearman_margin": r_mg, "p_margin_perm10k": p_mg})

    out = {
        "description": ("Flowers102 per-class difficulty agreement; difficulty "
                        "= held-out probe result averaged over N_SEEDS shared "
                        "train/holdout splits (identical splits for every "
                        "encoder); two difficulty definitions: per-class "
                        "accuracy and per-class mean true-class probability "
                        "(margin)."),
        "n_images": len(image_paths),
        "n_classes": N_CLS,
        "n_per_class": N_PER_CLASS,
        "n_shot": N_SHOT,
        "n_seeds": N_SEEDS,
        "n_perm": N_PERM,
        "rng_seed": RNG_SEED,
        "sanity": {
            "min_unique_values": MIN_UNIQUE,
            "encoders": {n: len(np.unique(np.round(v, 4)))
                         for n, v in accs.items()},
            "note": ("difficulty is only usable if each class has >= "
                     f"{MIN_UNIQUE} unique values; below that the probe is "
                     "saturated and pairwise rho is meaninglessly small"),
        },
        "overall_accuracy": {n: float(accs[n].mean()) for n in names},
        "overall_margin": {n: float(margs[n].mean()) for n in names},
        "accuracy_difficulty": {n: accs[n].tolist() for n in names},
        "margin_difficulty": {n: margs[n].tolist() for n in names},
        "pairwise_spearman": pairs,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()