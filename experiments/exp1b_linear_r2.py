"""Linear ridge R^2 on the SAME clean train/val split as exp1b_clean_r2.py.

Ensures the paper table's Linear and MLP columns are computed on identical
features (cached feats_*.npz), identical train/val partition (104 leaked removed,
500 disjoint val images). Only the adapter differs (ridge vs MLP).
"""
import os, sys, json, random
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.environ.get("SHAREDVISUAL_WORK", "/tmp/exp1b"))
from exp1b_encoder_swap import SUBDATASETS, load_split

DATA = os.environ.get("SHAREDVISUAL_DATA", str(Path.home() / "data"))
WORK = Path(os.environ.get("SHAREDVISUAL_WORK", "/tmp/exp1b"))
DEVICE = torch.device("cuda:0")
RNG = random.Random(42)


def ridge_clean_r2(xtr, ytr, xva, yva, lam=1e-3):
    n, t, din = xtr.shape
    Xtr = torch.from_numpy(xtr.reshape(-1, din).astype(np.float32)).to(DEVICE)
    Ytr = torch.from_numpy(ytr.reshape(-1, 1024).astype(np.float32)).to(DEVICE)
    mx = Xtr.mean(0); sx = Xtr.std(0) + 1e-6
    my = Ytr.mean(0); sy = Ytr.std(0) + 1e-6
    Xtr.sub_(mx).div_(sx)          # in-place -> Xn, no copy
    Ytr.sub_(my).div_(sy)          # in-place -> Yn
    XtX = Xtr.t() @ Xtr + lam * torch.eye(din, device=DEVICE)
    XtY = Xtr.t() @ Ytr
    del Xtr, Ytr
    torch.cuda.empty_cache()
    Wn = torch.linalg.solve(XtX, XtY)
    W = (Wn / sx[:, None]) * sy[None, :]
    b = my - mx @ W
    del XtX, XtY, Wn
    torch.cuda.empty_cache()
    Xva = torch.from_numpy(xva.reshape(-1, din).astype(np.float32)).to(DEVICE)
    Yva = torch.from_numpy(yva.reshape(-1, 1024).astype(np.float32)).to(DEVICE)
    pred = Xva @ W + b
    ss_res = float(((Yva - pred) ** 2).sum())
    ss_tot = float(((Yva - my) ** 2).sum())
    return 1 - ss_res / ss_tot


def main():
    # rebuild train/val partition EXACTLY as exp1b_clean_r2.py
    train_paths = []
    for sd in SUBDATASETS:
        p, _, _ = load_split(sd, DATA, "train", cap=8000, rng=RNG)
        train_paths += p
    train_paths = train_paths[:5000]
    val_paths_all, val_labels_all, off = [], [], 0
    for sd in SUBDATASETS:
        p, l, _ = load_split(sd, DATA, "val", rng=RNG)
        val_paths_all += p
        val_labels_all += [x + off for x in l]
        off += len(sd["classes"])
    rng = random.Random(42)
    per_class = {}
    for j, lb in enumerate(val_labels_all):
        per_class.setdefault(lb, []).append(j)
    chosen = []
    for ci, idxs in per_class.items():
        k = min(100, len(idxs))
        chosen.extend(rng.sample(idxs, k))
    chosen.sort()
    val_paths = [val_paths_all[j] for j in chosen]
    val_sample = val_paths[:500]

    train_names = set(os.path.basename(p) for p in train_paths)
    # match clean_r2.py exactly: leaked = train ∩ subsampled-val (104), not full val
    val_names_sub = set(os.path.basename(p) for p in val_paths)
    leaked = train_names & val_names_sub
    clean_mask = np.array([os.path.basename(p) not in leaked for p in train_paths], dtype=bool)
    print(f"clean_train={int(clean_mask.sum())} val_sample={len(val_sample)} "
          f"leaked_in_val={sum(1 for p in val_sample if os.path.basename(p) in train_names)}")

    base_tr = np.load(WORK / "feats_baseline.npz", allow_pickle=True)["x"][clean_mask]
    base_va = np.load(WORK / "feats_val_baseline.npz", allow_pickle=True)["x"]
    results = {}
    for v in ["siglip", "b32", "qwen"]:
        xtr = np.load(WORK / f"feats_{v}.npz", allow_pickle=True)["x"][clean_mask]
        xva = np.load(WORK / f"feats_val_{v}.npz", allow_pickle=True)["x"]
        r2 = ridge_clean_r2(xtr, base_tr, xva, base_va)
        results[v] = r2
        print(f"  {v:7s} linear clean val R^2 = {r2:.4f}")

    # merge into exp1b_clean_r2.json
    clean_path = WORK / "exp1b_clean_r2.json"
    d = json.load(open(clean_path))
    d["linear_clean_val_R2"] = results
    json.dump(d, open(clean_path, "w"), indent=2)
    print(f"\nMerged linear R^2 into {clean_path}")
    print("Final consistent table (all on same clean train/val):")
    for v in ["siglip", "b32", "qwen"]:
        print(f"  {v:7s} linear={results[v]:.3f}  mlp={d['clean_val_R2'][v]:.3f}")


if __name__ == "__main__":
    main()
