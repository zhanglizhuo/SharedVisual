#!/usr/bin/env python3
"""Visual-side perturbation synchrony: do encoders' per-class difficulty
changes under Gaussian blur track each other?

Motivation: the paper's geometry claims (CKA/Mantel) are static. The dynamic
question a reviewer will ask: under a visual intervention (blur), do
independently trained encoders' per-class difficulty shifts correlate once the
global "everything got harder" drift is removed?

Protocol (mirrors exp4_clip_blur_recreate.py so blur definitions are shared):
  - TeacherBehavior 8 classes, first 50 images/class in sorted filename order
  - PIL ImageFilter.GaussianBlur(radius=sigma), JPEG quality=90, sigma in
    {0,1,2,4,8}
  - Encoders:
      zero-shot protocol (same "a photo of {class}" text prompts, 4 encoders):
        CLIP ViT-L/14 (OpenAI), CLIP ViT-L/14 (LAION), CLIP ViT-B/32 (OpenAI),
        SigLIP ViT-SO400M-14-384 (webli)
      linear-probe protocol (25/25 split x 3 seeds, closed-form ridge+softmax):
        CLIP ViT-L/14 (OpenAI), Qwen2-VL-7B vision encoder, ResNet-50 (IN-1k)
  - Per class and sigma: accuracy AND mean true-class softmax probability
    (probabilities are smoother than 0/1 accuracy at n=50).

Analysis (analysis/visual_perturbation_sync.py):
  - Delta_t(c) = score_clean(c) - score_blur_t(c) per encoder
  - cross-encoder synchrony of Delta_t per sigma, with the global drift
    removed: Spearman (rank-based, invariant to additive drift) and Pearson on
    mean-centered Deltas; permutation test over class relabelings (seed 42).

Output: results/visual_perturbation_sync_raw.json (checkpointed per encoder;
resumable with --only-encoder NAME).

Usage (V100, dl_env):
  python3 experiments/exp_visual_perturbation_sync.py
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("SHAREDVISUAL_RESULTS", str(REPO_ROOT / "results")))
DATA_ROOT = Path(os.environ.get(
    "SHAREDVISUAL_DATA",
    "/home/broadsense/works/lizhuo/AutoResearchClaw/datasets_scb"))
SCB5_BASE = (DATA_ROOT / "SCB5_TeacherBehavior"
             / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2")
IMG_DIR = SCB5_BASE / "images" / "val"
LBL_DIR = SCB5_BASE / "labels" / "val"

CLASSES = ["guide", "answer", "On-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackBoard"]
SIGMAS = [0, 1, 2, 4, 8]
N_CAP = 50
PROBE_SEEDS = [42, 123, 2024]


def read_label(path):
    with open(path) as f:
        line = f.readline().strip()
    return int(line.split()[0]) if line else None


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


def blur_bytes(img_path, sigma):
    if sigma == 0:
        return open(img_path, "rb").read()
    img = Image.open(img_path).convert("RGB")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    buf = io.BytesIO()
    blurred.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def per_class_scores(preds, probs, cls_of):
    acc = {c: 0.0 for c in CLASSES}
    mprob = {c: 0.0 for c in CLASSES}
    tot = {c: 0 for c in CLASSES}
    for i, p in enumerate(preds):
        c = cls_of[i]
        tot[c] += 1
        if p == c:
            acc[c] += 1.0
    for c in CLASSES:
        acc[c] = 100.0 * acc[c] / tot[c]
    for i, pr in enumerate(probs):
        mprob[cls_of[i]] += pr[CLASSES.index(cls_of[i])]
    for c in CLASSES:
        mprob[c] = mprob[c] / tot[c]
    return {c: round(acc[c], 2) for c in CLASSES}, \
           {c: round(float(mprob[c]), 4) for c in CLASSES}


def ridge_probe(Xtr, ytr, Xte, lam=1e-3):
    """Closed-form multiclass ridge + softmax; deterministic, no deps."""
    Y = np.zeros((len(ytr), len(CLASSES)))
    for i, c in enumerate(ytr):
        Y[i, CLASSES.index(c)] = 1.0
    Xtr = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-12)
    Xte = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-12)
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Y)
    logits = Xte @ W
    return softmax(logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-encoder", default=None,
                    help="resume: process only this encoder name")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--regime", choices=["50", "all"], default="50",
                    help="sample regime: 50 imgs/class (exp2/exp4 protocol) "
                         "or ALL labeled val images (n=3240, low quant noise)")
    ap.add_argument("--zs-only", action="store_true",
                    help="skip the linear-probe encoders (they adapt to blur; "
                         "only zero-shot difficulty responds to the perturbation)")
    args = ap.parse_args()

    import clip
    import open_clip

    device = args.device
    samples = collect_all() if args.regime == "all" else collect_capped()
    tag = "[all]" if args.regime == "all" else ""
    out_path = RESULTS_DIR / (
        "visual_perturbation_sync_raw.json" if args.regime == "50"
        else "visual_perturbation_sync_raw_all.json")
    cls_of = [c for _, c in samples]
    n = len(samples)
    print(f"Images: {n} ({dict({c: cls_of.count(c) for c in CLASSES})})", flush=True)

    out_path = RESULTS_DIR / (
        "visual_perturbation_sync_raw.json" if args.regime == "50"
        else "visual_perturbation_sync_raw_all.json")
    if out_path.exists():
        result = json.loads(out_path.read_text())
    else:
        result = {}

    # ---------- zero-shot encoders ----------
    zs_models = [
        (f"CLIP ViT-L/14 (OpenAI){tag}", "clip", "ViT-L/14", "openai", 77),
        (f"CLIP ViT-L/14 (LAION){tag}", "open_clip", "ViT-L-14",
         "laion2b_s32b_b82k", 77),
        (f"CLIP ViT-B/32 (OpenAI){tag}", "clip", "ViT-B/32", "openai", 77),
        (f"SigLIP ViT-SO400M-14-384{tag}", "open_clip",
         "ViT-SO400M-14-SigLIP-384", "webli", 64),
    ]
    for name, backend, mname, pretrained, ctx in zs_models:
        if args.only_encoder and args.only_encoder != name:
            continue
        if name in result:
            print(f"[skip] {name} already done", flush=True)
            continue
        print(f"[ZS] {name}", flush=True)
        siglip_hf = None
        if backend == "clip":
            model, preprocess = clip.load(mname, device=device)
        elif mname == "ViT-SO400M-14-SigLIP-384":
            # open_clip's bundled tokenizer (CLIP BPE, vocab 49408) produces
            # out-of-range token ids for SigLIP (vocab 32000); use the HF
            # checkpoint (cached on this host) instead.
            from transformers import SiglipModel, SiglipProcessor
            siglip_hf = SiglipModel.from_pretrained(
                "google/siglip-so400m-patch14-384", torch_dtype=torch.float16)
            siglip_hf = siglip_hf.to(device).eval()
            processor = SiglipProcessor.from_pretrained(
                "google/siglip-so400m-patch14-384")
            model = None
            preprocess = None
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                mname, pretrained=pretrained, device=device)
        if model is not None:
            model.eval()
        if siglip_hf is not None:
            import torch as _t
            with _t.no_grad():
                text_feats = siglip_hf.get_text_features(
                    **processor(text=[f"a photo of {c}" for c in CLASSES],
                                padding="max_length", max_length=64,
                                return_tensors="pt").to(device))
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            text_feats = text_feats.float().cpu().numpy()
        else:
            tokens = clip.tokenize([f"a photo of {c}" for c in CLASSES]) if backend == "clip" \
                else open_clip.tokenize([f"a photo of {c}" for c in CLASSES],
                                        context_length=ctx)
            tokens = tokens.to(device)
            with torch.no_grad():
                text_feats = model.encode_text(tokens)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            text_feats = text_feats.float().cpu().numpy()

        result[name] = {}
        for s in SIGMAS:
            feats = []
            for img_path, _ in samples:
                data = blur_bytes(img_path, s)
                img = Image.open(io.BytesIO(data)).convert("RGB")
                if siglip_hf is not None:
                    with torch.no_grad():
                        pv = processor(images=img, return_tensors="pt").to(device)
                        pv["pixel_values"] = pv["pixel_values"].to(torch.float16)
                        f = siglip_hf.get_image_features(**pv)
                        f = f / f.norm(dim=-1, keepdim=True)
                    feats.append(f.float().cpu().numpy()[0])
                else:
                    t = preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        f = model.encode_image(t)
                        f = f / f.norm(dim=-1, keepdim=True)
                    feats.append(f.cpu().numpy()[0])
            feats = np.stack(feats)
            logits = feats @ text_feats.T
            probs = softmax(logits)
            preds = [CLASSES[i] for i in logits.argmax(axis=-1)]
            acc, mprob = per_class_scores(preds, probs, cls_of)
            result[name][f"sigma_{s}"] = {"accuracy": acc, "mean_prob": mprob}
            print(f"  sigma={s}: acc={acc}", flush=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  saved {name} -> {out_path}", flush=True)

    # ---------- probe encoders ----------
    def extract_qwen(paths, device, sigma):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        qwen = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16,
            device_map="auto", attn_implementation="eager")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        qwen.eval()
        feats = []
        for p in paths:
            data = blur_bytes(p, sigma)
            img = Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224))
            conv = [{"role": "user", "content": [
                {"type": "image", "max_pixels": 224 * 224},
                {"type": "text", "text": "x"}]}]
            prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(device)
            with torch.no_grad():
                out = qwen.visual(inputs["pixel_values"], inputs["image_grid_thw"])
                pooled = out.mean(dim=0).squeeze(0)
                pooled = pooled / pooled.norm(dim=-1)
            feats.append(pooled.cpu().numpy())
        return np.stack(feats)

    def extract_resnet(paths, device, sigma):
        import torchvision.transforms as T
        from torchvision import models
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model = model.to(device).eval()
        tfm = T.Compose([
            T.Resize(256), T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        feats = []
        for p in paths:
            data = blur_bytes(p, sigma)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            t = tfm(img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = model(t)
            feats.append(f.cpu().numpy()[0])
        return np.stack(feats)

    probe_models = [
        ("CLIP ViT-L/14 (OpenAI) probe", "clip_l14"),
        ("Qwen2-VL Vision probe", "qwen"),
        ("ResNet-50 (IN-1k) probe", "resnet"),
    ]
    if args.zs_only or args.regime == "all":
        probe_models = []
    clip_probe_model = None
    clip_probe_preprocess = None
    for name, kind in probe_models:
        if args.only_encoder and args.only_encoder != name:
            continue
        if name in result:
            print(f"[skip] {name} already done", flush=True)
            continue
        print(f"[PROBE] {name}", flush=True)
        if kind == "clip_l14":
            if clip_probe_model is None:
                clip_probe_model, clip_probe_preprocess = clip.load(
                    "ViT-L/14", device=device)
                clip_probe_model.eval()
        all_feats = {}
        for s in SIGMAS:
            paths = [p for p, _ in samples]
            if kind == "qwen":
                all_feats[s] = extract_qwen(paths, device, s)
            elif kind == "resnet":
                all_feats[s] = extract_resnet(paths, device, s)
            elif kind == "clip_l14":
                feats = []
                for img_path in paths:
                    data = blur_bytes(img_path, s)
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    t = clip_probe_preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        f = clip_probe_model.encode_image(t)
                        f = f / f.norm(dim=-1, keepdim=True)
                    feats.append(f.cpu().numpy()[0])
                all_feats[s] = np.stack(feats)
            print(f"  features sigma={s}: {all_feats[s].shape}", flush=True)

        result[name] = {}
        for s in SIGMAS:
            rng = np.random.RandomState(7)
            ids = np.arange(n)
            accs = np.zeros(len(CLASSES))
            mps = np.zeros(len(CLASSES))
            for seed in PROBE_SEEDS:
                rng = np.random.RandomState(seed)
                # stratified 25/25 split, same split across all encoders
                tr_ids, te_ids = [], []
                for ci, c in enumerate(CLASSES):
                    idx = [i for i in range(n) if cls_of[i] == c]
                    idx = rng.permutation(idx)
                    tr_ids += list(idx[:25])
                    te_ids += list(idx[25:])
                probs = ridge_probe(all_feats[s][tr_ids], [cls_of[i] for i in tr_ids],
                                    all_feats[s][te_ids])
                preds = [CLASSES[i] for i in probs.argmax(axis=-1)]
                te_cls = [cls_of[i] for i in te_ids]
                a, m = per_class_scores(preds, probs, te_cls)
                for c in CLASSES:
                    accs[CLASSES.index(c)] += a[c]
                    mps[CLASSES.index(c)] += m[c]
            accs /= len(PROBE_SEEDS)
            mps /= len(PROBE_SEEDS)
            result[name][f"sigma_{s}"] = {
                "accuracy": {c: round(float(v), 2) for c, v in zip(CLASSES, accs)},
                "mean_prob": {c: round(float(v), 4) for c, v in zip(CLASSES, mps)}}
            print(f"  sigma={s}: acc={result[name][f'sigma_{s}']['accuracy']}", flush=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  saved {name} -> {out_path}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
