#!/usr/bin/env python3
"""Exp 1B: vision-encoder swap for LLaVA-1.5-7B with label-free linear adapter.

Minimal-intervention test of the shared-encoder hypothesis.

Design
------
- LLaVA-1.5-7B (multi-modal projector + Vicuna-7B) stays FROZEN.
- Its CLIP ViT-L/14-336 vision tower is replaced by a different encoder E.
- A LINEAR adapter W_E: R^{d_E} -> R^{1024} is learned by ridge regression
  (exact least squares via accumulated normal equations) mapping E's patch
  features into LLaVA's CLIP feature space, on SCB5 TRAIN images WITHOUT any
  class labels (label-free regression avoids injecting class difficulty).
- The swapped model is evaluated on SCB5 val with the eval_hf.py protocol
  (13 classes, 100/class, seed 42, class-name prompt, substring matching,
  greedy decoding, max_new_tokens=15); per-class accuracy is correlated with
  AnchorScore (CLIP LAION-2B ViT-L/14 zero-shot).

Prediction
----------
- shared-geometry hypothesis: rho(swapped) ~ rho(baseline) ~ 0.473
- shared-encoder hypothesis: rho(swapped) << rho(baseline)

Variants
--------
baseline : original CLIP ViT-L/14-336 tower (no swap; reproducibility check)
siglip   : SigLIP ViT-SO400M-14-384 @336x336 (grid 24x24 == CLIP's, 1152d)
b32      : CLIP ViT-B/32 @224 + bilinear grid interpolation 7x7 -> 24x24 (512d)
qwen     : Qwen2-VL-7B vision tower @336x336 (grid 24x24, 1024d)
random   : per-image Gaussian noise (negative control; adapter trained but
           features carry no image structure -> per-class acc near chance)

Usage
-----
  python exp1b_encoder_swap.py \
      --data-dir ~/works/lizhuo/AutoResearchClaw/datasets_scb \
      --anchor-json /tmp/exp1b/anchor_scores.json \
      --workdir /tmp/exp1b --out /tmp/exp1b/exp1b_encoder_swap.json \
      [--variants baseline,siglip,b32,qwen,random] [--train-cap 8000]
      [--samples 100] [--seed 42] [--batch 8]
"""
import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SUBDATASETS = [
    {
        "key": "TeacherBehavior",
        "classes": ["guide", "answer", "On-stage interaction", "blackboard-writing",
                    "teacher", "stand", "screen", "blackBoard"],
        "subdir": "SCB5_TeacherBehavior/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2",
    },
    {
        "key": "HandriseReadWrite",
        "classes": ["hand-raising", "read", "write"],
        "subdir": "SCB5_HandriseReadWrite",
    },
    {
        "key": "BowTurnHead",
        "classes": ["BowHead", "TurnHead"],
        "subdir": "SCB_BowTurnHead",
    },
]

GRID = 24
GRID_TOKENS = GRID * GRID


def read_label(path):
    with open(path) as f:
        line = f.readline().strip()
    if not line:
        return None
    return int(line.split()[0])


def load_split(sd, data_dir, split, cap=None, rng=None):
    classes = sd["classes"]
    sd_name = f"{data_dir}/{sd['subdir']}"
    img_dir = f"{sd_name}/images/{split}"
    lbl_dir = f"{sd_name}/labels/{split}"
    if not os.path.isdir(img_dir):
        print(f"  SKIP: {img_dir} not found", flush=True)
        return [], [], classes
    idx_per_class = {i: [] for i in range(len(classes))}
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lp = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        if not os.path.exists(lp):
            continue
        cid = read_label(lp)
        if cid is not None and cid in idx_per_class:
            idx_per_class[cid].append(os.path.join(img_dir, fname))
    paths, labels = [], []
    for ci in range(len(classes)):
        indices = idx_per_class[ci]
        if cap is not None and len(indices) > cap:
            indices = rng.sample(indices, cap)
        for fpath in indices:
            paths.append(fpath)
            labels.append(ci)
    return paths, labels, classes


def spearman(x, y):
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return float(r), float(p)
    except ImportError:
        def rankdata(a):
            a = np.asarray(a, dtype=float)
            order = np.argsort(a, kind="mergesort")
            ranks = np.empty(len(a), dtype=float)
            ranks[order] = np.arange(1, len(a) + 1)
            s = np.argsort(a, kind="mergesort")
            i = 0
            while i < len(a):
                j = i
                while j + 1 < len(a) and a[s[j + 1]] == a[s[i]]:
                    j += 1
                if j > i:
                    ranks[s[i:j + 1]] = ranks[s[i:j + 1]].mean()
                i = j + 1
            return ranks
        rx, ry = rankdata(x), rankdata(y)
        r = float(np.corrcoef(rx, ry)[0, 1])
        n = len(x)
        t = r * np.sqrt((n - 2) / (1 - r * r + 1e-12))
        p = float(2 * (1 - _t_cdf_approx(abs(t), n - 2)))
        return r, p


def _t_cdf_approx(t, df):
    x = df / (df + t * t)
    return 1 - 0.5 * _betainc_approx(0.5 * df, 0.5, x)


def _betainc_approx(a, b, x):
    from math import lgamma, exp
    if x == 0 or x == 1:
        return 0.0 if x == 0 else 1.0
    ln_beta = lgamma(a) + lgamma(b) - lgamma(a + b)
    terms = 0.0
    term = exp(a * np.log(x) + b * np.log(1 - x) - ln_beta) / a
    terms = term
    k = 0
    while k < 200:
        term *= (a + b + k) * x / (a + k + 1)
        terms += term
        k += 1
        if abs(term) < 1e-12:
            break
    return min(1.0, max(0.0, terms))


class VisionOut:
    def __init__(self, feats, hidden_states=None):
        self.last_hidden_state = feats
        self.hidden_states = hidden_states


class SwappedTower(nn.Module):
    def __init__(self, mode, enc, adapter):
        super().__init__()
        self.mode = mode
        self.enc = enc
        self.adapter = adapter
        if adapter is not None:
            self.adapter.to(torch.float16)

    def forward(self, pixel_values=None, output_hidden_states=False, **kwargs):
        mode = self.mode
        if mode == "baseline":
            feats = self.enc(pixel_values, output_hidden_states=False).last_hidden_state[:, 1:]
        elif mode == "b32":
            feats = self.enc(pixel_values).last_hidden_state[:, 1:]
            b, t, d = feats.shape
            g = int(round(t ** 0.5))
            feats = feats.permute(0, 2, 1).reshape(b, d, g, g)
            feats = F.interpolate(feats, size=(GRID, GRID), mode="bilinear", align_corners=False)
            feats = feats.reshape(b, d, -1).permute(0, 2, 1)
        elif mode == "siglip":
            feats = self.enc(pixel_values, interpolate_pos_encoding=True).last_hidden_state
        elif mode == "qwen":
            b = pixel_values.shape[0] // (GRID * GRID)
            assert pixel_values.shape[0] % (GRID * GRID) == 0
            gt = torch.tensor([[1, GRID, GRID]] * b, device=pixel_values.device, dtype=torch.long)
            feats = self.enc(pixel_values, gt)
            feats = feats.reshape(b, GRID, GRID, -1).reshape(b, GRID * GRID, -1)
        elif mode == "random":
            feats = pixel_values
        adtype = next(self.adapter.parameters()).dtype
        feats = self.adapter(feats.to(adtype))
        cls = torch.zeros(feats.shape[0], 1, feats.shape[2], dtype=feats.dtype, device=feats.device)
        feats_cls = torch.cat([cls, feats], dim=1)
        return VisionOut(feats_cls, hidden_states=(feats_cls,) * 5)


def make_preprocessors(args, device):
    from transformers import (
        AutoImageProcessor, AutoProcessor, CLIPVisionModel, LlavaForConditionalGeneration,
        Qwen2VLForConditionalGeneration, SiglipVisionModel,
    )
    print("Loading LLaVA-1.5-7B ...", flush=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map=None)
    model.to(device)
    model.eval()
    llava_proc = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    tokenizer = llava_proc.tokenizer

    print("Loading SigLIP ViT-SO400M-14-384 ...", flush=True)
    siglip = SiglipVisionModel.from_pretrained(
        "google/siglip-so400m-patch14-384", torch_dtype=torch.float16).to(device).eval()
    siglip_proc = AutoImageProcessor.from_pretrained(
        "google/siglip-so400m-patch14-384", size={"height": 336, "width": 336})

    print("Loading CLIP ViT-B/32 ...", flush=True)
    b32 = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32", torch_dtype=torch.float16).to(device).eval()
    b32_proc = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("Loading Qwen2-VL-7B (vision tower only) ...", flush=True)
    qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16,
        device_map=None, low_cpu_mem_usage=True, attn_implementation="eager")
    qwen.eval()
    qwen_visual = qwen.visual.to(device)
    qwen_visual.merger = torch.nn.Identity()
    del qwen
    gc.collect()
    torch.cuda.empty_cache()
    qwen_proc = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    def qwen_pixels(imgs):
        fixed = [img.resize((336, 336), Image.BILINEAR) for img in imgs]
        out = qwen_proc.image_processor(images=fixed, return_tensors="pt")
        pv = out["pixel_values"]
        if isinstance(pv, (list, tuple)):
            pv = torch.cat([p.to(device) for p in pv], dim=0)
        else:
            pv = pv.to(device)
        return pv.to(torch.float16)

    def _canonical(imgs):
        return [img.resize((336, 336), Image.BILINEAR) for img in imgs]

    preprocessors = {
        "baseline": lambda imgs: llava_proc.image_processor(
            images=_canonical(imgs), return_tensors="pt")["pixel_values"].to(device=device, dtype=torch.float16),
        "siglip": lambda imgs: siglip_proc(images=_canonical(imgs), return_tensors="pt")["pixel_values"].to(device=device, dtype=torch.float16),
        "b32": lambda imgs: b32_proc(images=_canonical(imgs), return_tensors="pt")["pixel_values"].to(device=device, dtype=torch.float16),
        "qwen": qwen_pixels,
    }
    towers = {
        "baseline": model.vision_tower,
        "siglip": siglip,
        "b32": b32,
        "qwen": qwen_visual,
    }
    return model, llava_proc, tokenizer, preprocessors, towers


def extract_features(variant, paths, preprocessors, towers, workdir, device, batch=32):
    cache = workdir / f"feats_{variant}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        print(f"  {variant}: cache hit {d['x'].shape}", flush=True)
        return d["x"]
    rng = np.random.RandomState(1234 + len(variant))
    noise = None
    if variant == "random":
        noise = rng.standard_normal((len(paths), GRID_TOKENS, 1024)).astype(np.float16)
    xs = []
    t0 = time.time()
    eff_batch = 8 if variant == "qwen" else batch
    for i in range(0, len(paths), eff_batch):
        chunk = paths[i:i + eff_batch]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        with torch.no_grad():
            if variant == "random":
                pv = torch.from_numpy(noise[i:i + batch]).to(device)
                f = pv
            else:
                pv = preprocessors[variant](imgs)
                if variant == "baseline":
                    f = towers["baseline"](pv, output_hidden_states=False).last_hidden_state[:, 1:]
                elif variant == "b32":
                    raw = towers["b32"](pv).last_hidden_state[:, 1:]
                    b, t, d = raw.shape
                    g = int(round(t ** 0.5))
                    f = F.interpolate(raw.permute(0, 2, 1).reshape(b, d, g, g),
                                      size=(GRID, GRID), mode="bilinear", align_corners=False)
                    f = f.reshape(b, d, -1).permute(0, 2, 1)
                elif variant == "siglip":
                    f = towers["siglip"](pv, interpolate_pos_encoding=True).last_hidden_state
                elif variant == "qwen":
                    b = pv.shape[0] // (GRID * GRID)
                    assert pv.shape[0] % (GRID * GRID) == 0
                    gt = torch.tensor([[1, GRID, GRID]] * b, device=device, dtype=torch.long)
                    f = towers["qwen"](pv, gt)
                    f = f.reshape(b, GRID, GRID, -1).reshape(b, GRID * GRID, -1)
        xs.append(f.float().cpu().numpy())
        if (i // batch) % 20 == 0:
            el = time.time() - t0
            print(f"  {variant} [{i}/{len(paths)}] {el:.0f}s", flush=True)
    x = np.concatenate(xs, axis=0).astype(np.float16)
    np.savez(cache, x=x)
    print(f"  {variant}: {x.shape} -> {cache}", flush=True)
    return x


def train_adapter(x, y, lam=1e-3, val_frac=0.05):
    n, t, din = x.shape
    X = x.reshape(-1, din).astype(np.float32)
    Y = y.reshape(-1, 1024).astype(np.float32)
    idx = np.random.RandomState(42).permutation(n)
    tr_idx = idx[:int(n * (1 - val_frac))]
    va_idx = idx[int(n * (1 - val_frac)):]
    tr_mask = np.zeros(n, dtype=bool)
    tr_mask[tr_idx] = True
    tr = np.where(np.repeat(tr_mask, t))[0]
    va = np.where(np.repeat(~tr_mask, t))[0]
    mx = X[tr].mean(0)
    sx = X[tr].std(0) + 1e-6
    my = Y[tr].mean(0)
    sy = Y[tr].std(0) + 1e-6
    Xn = (X[tr] - mx) / sx
    Yn = (Y[tr] - my) / sy
    din = Xn.shape[1]
    XtX = Xn.T @ Xn + lam * np.eye(din)
    XtY = Xn.T @ Yn
    Wn = np.linalg.solve(XtX, XtY)
    W = (Wn / sx[:, None]) * sy[None, :]
    b = my - mx @ W
    pred = X[va] @ W + b
    ss_res = float(((Y[va] - pred) ** 2).sum())
    ss_tot = float(((Y[va] - my) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    print(f"  adapter {din}->1024: ridge R2 (val 5%) = {r2:.4f}", flush=True)
    return W.astype(np.float16), b.astype(np.float16), r2


class MLPAdapter(nn.Module):
    def __init__(self, din, dout=1024, hidden=2048, p=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(din)
        self.fc1 = nn.Linear(din, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden, dout)

    def forward(self, x):
        x = x.to(self.norm.weight.dtype)
        x = self.norm(x)
        x = self.drop(self.act(self.fc1(x)))
        return self.fc2(x)


def train_mlp_adapter(x, y, device, epochs=12, batch=512, val_frac=0.05, hidden=2048):
    n, t, din = x.shape
    X = torch.from_numpy(x.reshape(-1, din).astype(np.float32))
    Y = torch.from_numpy(y.reshape(-1, 1024).astype(np.float32))
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    ntr = int(n * (1 - val_frac))
    tr_mask = torch.zeros(n, dtype=torch.bool)
    tr_mask[idx[:ntr]] = True
    tok_tr = torch.repeat_interleave(tr_mask, t)
    tok_va = torch.repeat_interleave(~tr_mask, t)
    mlp = MLPAdapter(din, 1024, hidden).to(device).float()
    Xtr = X[tok_tr]
    Ytr = Y[tok_tr]
    Xva = X[tok_va]
    Yva = Y[tok_va]
    my = Ytr.mean(0).to(device).float()
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    ns = Xtr.shape[0]
    nva = Xva.shape[0]
    for ep in range(epochs):
        mlp.train()
        perm = torch.randperm(ns)
        tot = 0.0
        for i in range(0, ns, batch):
            bidx = perm[i:i + batch]
            xb = Xtr[bidx].to(device).float()
            yb = Ytr[bidx].to(device).float()
            pred = mlp(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            opt.step()
            tot += float(loss) * bidx.shape[0]
        mlp.eval()
        ss_res = 0.0
        ss_tot = 0.0
        with torch.no_grad():
            for i in range(0, nva, 4096):
                xb = Xva[i:i + 4096].to(device).float()
                yb = Yva[i:i + 4096].to(device).float()
                pred = mlp(xb)
                ss_res += float(((yb - pred) ** 2).sum())
                ss_tot += float(((yb - my) ** 2).sum())
            r2 = 1 - ss_res / ss_tot
        print(f"  mlp {din}->1024 ep{ep+1}/{epochs} train_mse={tot/ns:.4f} val_R2={r2:.4f}", flush=True)
    mlp.to(torch.float16).eval()
    return mlp, r2


def eval_variant(model, variant, towers, preprocessors, tokenizer, val_paths, val_labels,
                 class_names, device, batch, rng):
    prompt = ("This is a classroom behavior image. Classify the dominant behavior: "
              + ", ".join(class_names) + ". Output only the behavior name.")
    text = f"USER: <image>\n{prompt}\nASSISTANT:"
    tok = tokenizer(text, return_tensors="pt")
    input_ids = tok.input_ids.to(device)
    attn = tok.attention_mask.to(device)

    if variant == "random":
        noise_rng = np.random.RandomState(777)
        noise = noise_rng.standard_normal((len(val_paths), GRID_TOKENS, 1024)).astype(np.float16)

    results = {c: {"correct": 0, "total": 0} for c in class_names}
    n = len(val_paths)
    t0 = time.time()
    for i in range(0, n, batch):
        chunk_idx = range(i, min(i + batch, n))
        paths = [val_paths[j] for j in chunk_idx]
        imgs = [Image.open(p).convert("RGB") for p in paths]
        with torch.no_grad():
            if variant == "random":
                pv = torch.from_numpy(noise[i:i + batch]).to(device)
            else:
                pv = preprocessors[variant](imgs)
            bs = len(chunk_idx)
            out = model.generate(
                input_ids=input_ids.repeat(bs, 1), attention_mask=attn.repeat(bs, 1),
                pixel_values=pv, max_new_tokens=15)
        for k, j in enumerate(chunk_idx):
            resp = tokenizer.decode(out[k], skip_special_tokens=True)
            resp = resp.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in resp else resp
            pred = None
            for ci, c in enumerate(class_names):
                if c.lower().strip() in resp.lower().strip():
                    pred = ci
                    break
            cname = class_names[val_labels[j]]
            results[cname]["total"] += 1
            if pred == val_labels[j]:
                results[cname]["correct"] += 1
        done = min(i + batch, n)
        if done % 200 == 0 or done == n:
            el = time.time() - t0
            acc = sum(r["correct"] for r in results.values()) / max(1, sum(r["total"] for r in results.values()))
            print(f"  [{variant} {done}/{n}] acc={acc:.3f} {el:.0f}s", flush=True)
    return {c: {"correct": r["correct"], "total": r["total"],
                "acc": round(r["correct"] / r["total"] * 100, 2) if r["total"] > 0 else 0.0}
            for c, r in results.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--anchor-json", required=True)
    ap.add_argument("--workdir", default="/tmp/exp1b")
    ap.add_argument("--out", default="/tmp/exp1b/exp1b_encoder_swap.json")
    ap.add_argument("--variants", default="baseline,siglip,b32,qwen,random")
    ap.add_argument("--train-cap", type=int, default=8000)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--feat-batch", type=int, default=32)
    ap.add_argument("--max-train-imgs", type=int, default=0,
                    help="debug: cap total train images (0 = use train-cap/class)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    anchor = json.load(open(args.anchor_json))
    class_map = {sd["key"]: sd["classes"] for sd in SUBDATASETS}
    anchor_vals = []
    class_names_all = []
    for ds_key, classes in class_map.items():
        for cls in classes:
            anchor_vals.append(anchor[ds_key]["per_class_acc"][cls]["acc"])
            class_names_all.append((ds_key, cls))

    train_paths, train_labels = [], []
    for sd in SUBDATASETS:
        p, l, _ = load_split(sd, args.data_dir, "train", cap=args.train_cap, rng=random)
        print(f"  train {sd['key']}: {len(p)} images", flush=True)
        train_paths += p
        train_labels += l
    if args.max_train_imgs:
        train_paths = train_paths[:args.max_train_imgs]

    val_paths, val_labels = [], []
    offset = 0
    for sd in SUBDATASETS:
        p, l, _ = load_split(sd, args.data_dir, "val", rng=random)
        val_paths += p
        val_labels += [x + offset for x in l]
        offset += len(sd["classes"])
    rng = random.Random(args.seed)
    per_class = {}
    for j, lb in enumerate(val_labels):
        per_class.setdefault(lb, []).append(j)
    chosen = []
    for ci, idxs in per_class.items():
        k = min(args.samples, len(idxs))
        chosen.extend(rng.sample(idxs, k))
    chosen.sort()
    val_paths = [val_paths[j] for j in chosen]
    val_labels = [val_labels[j] for j in chosen]
    print(f"  val: {len(val_paths)} images across 13 classes", flush=True)

    model, llava_proc, tokenizer, preprocessors, towers = make_preprocessors(args, device)

    variants = args.variants.split(",")
    out = {
        "experiment": "exp1b_encoder_swap",
        "model": "llava-hf/llava-1.5-7b-hf",
        "protocol": "eval_hf.py-compatible (13 classes, 100/class, seed 42, class-name prompt, substring matching, greedy, max_new_tokens=15)",
        "adapter": {"method": "label-free 2-layer MLP (GELU, dropout) trained by MSE to map patch features to LLaVA CLIP space",
                    "hidden": 2048, "epochs": 12, "val_frac": 0.05, "train_imgs": len(train_paths)},
        "anchor": "CLIP LAION-2B ViT-L/14 per-class zero-shot accuracy (VisualAnchor anchor_scores.json)",
        "variants": {},
    }

    orig_tower = model.vision_tower
    features = {}
    for v in variants:
        print(f"\n=== extract features: {v} ===", flush=True)
        features[v] = extract_features(v, train_paths, preprocessors, towers, workdir, device,
                                       batch=args.feat_batch)
    baseline_x = features["baseline"]

    print("\n=== train MLP adapters ===", flush=True)
    adapters = {}
    for v in variants:
        if v == "baseline":
            continue
        print(f"  {v}: training MLP adapter ...", flush=True)
        mlp, r2 = train_mlp_adapter(features[v], baseline_x, device)
        adapters[v] = (mlp, r2)

    for v in variants:
        print(f"\n=== eval variant: {v} ===", flush=True)
        if v == "baseline":
            model.vision_tower = orig_tower
            adapter_r2 = None
        else:
            mlp, r2 = adapters[v]
            mlp.eval()
            enc = towers[v] if v != "random" else None
            model.vision_tower = SwappedTower(v, enc, mlp).to(device)
            adapter_r2 = r2
        model.to(device)
        accs = eval_variant(model, v, towers, preprocessors, tokenizer, val_paths, val_labels,
                            [c for _, c in class_names_all], device, args.batch, rng)
        acc_vec = [accs[c]["acc"] for _, c in class_names_all]
        rho, p = spearman(np.array(anchor_vals), np.array(acc_vec))
        print(f"  {v}: rho={rho:.3f} p={p:.3f}", flush=True)
        out["variants"][v] = {"per_class_acc": accs, "rho": rho, "p": p,
                              "adapter_r2": adapter_r2}
        json.dump(out, open(args.out, "w"), indent=2)

    print("\n=== SUMMARY ===", flush=True)
    for v, d in out["variants"].items():
        print(f"  {v:8s} rho={d['rho']:.3f} p={d['p']:.3f} r2={d['adapter_r2']}", flush=True)
    print(f"\nSaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
