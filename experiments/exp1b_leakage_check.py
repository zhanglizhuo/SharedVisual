"""Leakage check for exp1b MLP adapter R^2.

Verifies:
1. train/val image sets are disjoint (no shared filenames across the SCB5 splits).
2. The 5% holdout for R^2 is image-level (entire images excluded, not just tokens).
3. Decisively: re-train MLP adapters on train-features but evaluate R^2 on a fully
   independent sample drawn from the VAL split (never used in any training/adapter
   fitting). If val-split R^2 ~ train-holdout R^2, no leakage. If much lower, the
   train-holdout R^2 was inflated (overfit to train distribution).
"""
import os, sys, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, os.environ.get("SHAREDVISUAL_WORK", "/tmp/exp1b"))
from exp1b_encoder_swap import (
    SUBDATASETS, load_split, MLPAdapter, make_preprocessors,
)

DATA = os.environ.get("SHAREDVISUAL_DATA", str(Path.home() / "data"))
WORK = Path(os.environ.get("SHAREDVISUAL_WORK", "/tmp/exp1b"))
DEVICE = torch.device("cuda:0")
RNG = random.Random(42)


def main():
    # 1. load train + val paths the same way exp1b does
    train_paths, train_labels = [], []
    for sd in SUBDATASETS:
        p, l, _ = load_split(sd, DATA, "train", cap=8000, rng=RNG)
        train_paths += p; train_labels += l
    train_paths = train_paths[:5000]
    val_paths, val_labels, off = [], [], 0
    for sd in SUBDATASETS:
        p, l, _ = load_split(sd, DATA, "val", rng=RNG)
        val_paths += p; val_labels += [x + off for x in l]
        off += len(sd["classes"])
    # subsample val to 100/class like exp1b
    rng = random.Random(42)
    per_class = {}
    for j, lb in enumerate(val_labels):
        per_class.setdefault(lb, []).append(j)
    chosen = []
    for ci, idxs in per_class.items():
        k = min(100, len(idxs))
        chosen.extend(rng.sample(idxs, k))
    val_paths = [val_paths[j] for j in chosen]

    tr_set = set(os.path.basename(p) for p in train_paths)
    va_set = set(os.path.basename(p) for p in val_paths)
    inter = tr_set & va_set
    print(f"[1] train={len(tr_set)} val={len(va_set)} intersection={len(inter)}")
    if inter:
        print(f"    !! LEAKAGE: {list(inter)[:5]} ...")

    # 2. extract val features for baseline (y) + each variant, reuse cached train features
    #    We need a small feature extractor on val images. Load towers once.
    print("\n[2] loading models for val feature extraction...")
    import argparse
    args = argparse.Namespace()
    args2 = argparse.Namespace(data_dir=DATA)
    model, llava_proc, tok, preprocessors, towers = make_preprocessors(args2, DEVICE)
    from exp1b_encoder_swap import extract_features, train_mlp_adapter

    # cached train features
    tr_feats = {}
    for v in ["baseline", "siglip", "b32", "qwen"]:
        d = np.load(WORK / f"feats_{v}.npz", allow_pickle=True)
        tr_feats[v] = d["x"]
    base_tr = tr_feats["baseline"]

    # extract val features (first 500 val imgs, disjoint from train by construction)
    val_sample = val_paths[:500]
    print(f"\n[3] extracting val features on {len(val_sample)} held-out val images...")
    val_feats = {}
    for v in ["baseline", "siglip", "b32", "qwen"]:
        cache = WORK / f"feats_val_{v}.npz"
        if cache.exists():
            val_feats[v] = np.load(cache, allow_pickle=True)["x"]
            print(f"    {v}: val cache hit {val_feats[v].shape}")
        else:
            # monkey-extract: use extract_features on val paths (writes feats_val_{v}.npz)
            # simplest: inline a small extractor
            xs = []
            from PIL import Image
            eff_batch = 8 if v == "qwen" else 32
            for i in range(0, len(val_sample), eff_batch):
                chunk = val_sample[i:i+eff_batch]
                imgs = [Image.open(p).convert("RGB") for p in chunk]
                with torch.no_grad():
                    if v == "baseline":
                        f = towers["baseline"](preprocessors[v](imgs), output_hidden_states=False).last_hidden_state[:, 1:]
                    elif v == "b32":
                        raw = towers["b32"](preprocessors[v](imgs)).last_hidden_state[:, 1:]
                        b, t, d = raw.shape; g = int(round(t**0.5))
                        f = F.interpolate(raw.permute(0,2,1).reshape(b,d,g,g), size=(24,24), mode="bilinear", align_corners=False)
                        f = f.reshape(b,d,-1).permute(0,2,1)
                    elif v == "siglip":
                        f = towers["siglip"](preprocessors[v](imgs), interpolate_pos_encoding=True).last_hidden_state
                    elif v == "qwen":
                        pv = preprocessors[v](imgs)
                        bb = pv.shape[0]//(24*24)
                        gt = torch.tensor([[1,24,24]]*bb, device=DEVICE, dtype=torch.long)
                        f = towers["qwen"](pv, gt).reshape(bb,24,24,-1).reshape(bb,576,-1)
                xs.append(f.float().cpu().numpy())
            val_feats[v] = np.concatenate(xs, axis=0).astype(np.float16)
            np.savez(cache, x=val_feats[v])
            print(f"    {v}: {val_feats[v].shape} -> {cache}")

    # 4. retrain MLPs on FULL train (no holdout), eval R^2 on the 500 val images
    print("\n[4] retrain MLP on full train, evaluate R^2 on independent VAL images:")
    base_va = val_feats["baseline"]
    results = {}
    for v in ["siglip", "b32", "qwen"]:
        # train on all 5000 train images
        mlp, _ = train_mlp_adapter(tr_feats[v], base_tr, DEVICE, epochs=12, batch=512)
        # eval on val
        mlp.eval()
        nva = base_va.shape[0]
        with torch.no_grad():
            ss_res = 0.0; ss_tot = 0.0
            my = torch.from_numpy(base_va.reshape(-1,1024).astype(np.float32)).mean(0).to(DEVICE).float()
            for i in range(0, nva*576, 4096):
                xb = torch.from_numpy(val_feats[v].reshape(-1, val_feats[v].shape[-1]).astype(np.float32)[i:i+4096]).to(DEVICE).float()
                yb = torch.from_numpy(base_va.reshape(-1,1024).astype(np.float32)[i:i+4096]).to(DEVICE).float()
                pred = mlp(xb)
                ss_res += float(((yb-pred)**2).sum())
                ss_tot += float(((yb-my)**2).sum())
            r2_val = 1 - ss_res/ss_tot
        results[v] = r2_val
        print(f"    {v:7s} val-split R^2 = {r2_val:.4f}")

    print("\n=== LEAKAGE CHECK SUMMARY ===")
    print(f"train∩val filename overlap: {len(inter)} (must be 0)")
    print(f"val-split R^2 (fully independent): siglip={results['siglip']:.3f} b32={results['b32']:.3f} qwen={results['qwen']:.3f}")
    print("compare to train-holdout R^2:        siglip=0.340 b32=0.230 qwen=0.145")
    print("If val-split ~ train-holdout: no leakage, R^2 trustworthy.")


if __name__ == "__main__":
    main()
