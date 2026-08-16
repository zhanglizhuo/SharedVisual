"""Clean R^2 recompute for exp1b: train and val FULLY disjoint.

The SCB5 dataset has 104 filenames appearing in BOTH train and val split dirs.
This script:
1. Identifies the leaked filenames.
2. Excludes them from the TRAIN feature set (cached feats_*.npz rows).
3. Retrains MLP adapters on the clean train set.
4. Evaluates R^2 on the 500 val images (cached feats_val_*.npz) which are now
   guaranteed disjoint from train.

Reports the clean, defensible R^2 for the paper.
"""
import os, sys, json, random
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.environ.get("SHAREDVISUAL_WORK", "/tmp/exp1b"))
from exp1b_encoder_swap import SUBDATASETS, load_split, train_mlp_adapter

DATA = os.environ.get("SHAREDVISUAL_DATA", str(Path.home() / "data"))
WORK = Path(os.environ.get("SHAREDVISUAL_WORK", "/tmp/exp1b"))
DEVICE = torch.device("cuda:0")
RNG = random.Random(42)


def main():
    # rebuild train_paths EXACTLY as exp1b did (same seed/order) to index cached feats
    train_paths = []
    for sd in SUBDATASETS:
        p, _, _ = load_split(sd, DATA, "train", cap=8000, rng=RNG)
        train_paths += p
    train_paths = train_paths[:5000]
    # val paths as exp1b did
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

    # leaked = train filenames that appear in val
    train_names = set(os.path.basename(p) for p in train_paths)
    val_names_all = set(os.path.basename(p) for p in val_paths)
    leaked_names = train_names & val_names_all
    # mask of train rows that are NOT leaked
    train_clean_mask = np.array(
        [os.path.basename(p) not in leaked_names for p in train_paths], dtype=bool)
    n_clean = int(train_clean_mask.sum())
    print(f"train={len(train_paths)} leaked_removed={len(train_paths)-n_clean} clean_train={n_clean}")
    print(f"val_sample={len(val_sample)} leaked_in_val_sample="
          f"{sum(1 for p in val_sample if os.path.basename(p) in train_names)}")

    results = {}
    base_tr_full = np.load(WORK / "feats_baseline.npz", allow_pickle=True)["x"]
    base_tr_clean = base_tr_full[train_clean_mask]
    base_va = np.load(WORK / "feats_val_baseline.npz", allow_pickle=True)["x"]

    for v in ["siglip", "b32", "qwen"]:
        xtr_full = np.load(WORK / f"feats_{v}.npz", allow_pickle=True)["x"]
        xtr_clean = xtr_full[train_clean_mask]
        xva = np.load(WORK / f"feats_val_{v}.npz", allow_pickle=True)["x"]
        # retrain MLP on CLEAN train
        mlp, _ = train_mlp_adapter(xtr_clean, base_tr_clean, DEVICE, epochs=12, batch=512)
        mlp.eval()
        # eval R^2 on val (now fully disjoint)
        din = xva.shape[-1]
        xva_flat = torch.from_numpy(xva.reshape(-1, din).astype(np.float32))
        yva_flat = torch.from_numpy(base_va.reshape(-1, 1024).astype(np.float32))
        my = yva_flat.mean(0).to(DEVICE).float()
        ss_res = 0.0; ss_tot = 0.0
        with torch.no_grad():
            for i in range(0, xva_flat.shape[0], 4096):
                xb = xva_flat[i:i+4096].to(DEVICE).float()
                yb = yva_flat[i:i+4096].to(DEVICE).float()
                pred = mlp(xb)
                ss_res += float(((yb-pred)**2).sum())
                ss_tot += float(((yb-my)**2).sum())
            r2 = 1 - ss_res/ss_tot
        results[v] = r2
        print(f"  {v:7s} clean val R^2 = {r2:.4f}")

    print("\n=== CLEAN R^2 (train and val fully disjoint) ===")
    for v, r2 in results.items():
        print(f"  {v:7s} {r2:.3f}")
    json.dump({"leakage_removed": len(train_paths)-n_clean,
               "clean_train_imgs": n_clean,
               "val_imgs": len(val_sample),
               "clean_val_R2": results,
               "note": "MLP retrained on train excluding 104 filenames that also appear in val; R^2 evaluated on 500 fully-disjoint val images"},
              open(WORK / "exp1b_clean_r2.json", "w"), indent=2)
    print(f"\nSaved -> {WORK/'exp1b_clean_r2.json'}")


if __name__ == "__main__":
    main()
