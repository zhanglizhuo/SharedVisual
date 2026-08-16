#!/usr/bin/env python3
"""A3 — Stanford40 class-resampling Mantel test.

Addresses the "universal geometry" robustness concern (Major Concern #1),
the "how would it look with different classes" question, and the "exact
permutation only covers label-relabeling, not class-set sampling" critique.

1. Bootstrap (10k class resamples w/ replacement): recompute centroid
   similarity matrices for CLIP & QwenViT, compute Mantel ρ, build CI.
2. Subsampling: random k-class subsets (k=10, 15, 20, 30, 40) to map how
   geometry agreement scales with class count.

Input:  results/exp6_clip_features.npz  (40 classes, CLIP ViT-L/14 OpenAI 768-d)
        results/exp6_qwen_features.npz  (40 classes, Qwen2-VL Vision 3584-d)

Output: results/mantel_bootstrap.json
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from sharedvisual.paths import RESULTS_DIR
OUT_DIR = RESULTS_DIR

import numpy as np
from scipy.stats import pearsonr


OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 20260801
N_BOOT = 2000
N_SUBSAMPLE = 500


def precompute_centroids(features, class_names):
    """L2-normalized per-class centroids. Returns (n_class, d) array and list."""
    cents_list = []
    for c in class_names:
        f = np.array(features[c], dtype=np.float64)
        mu = f.mean(axis=0)
        norm = np.linalg.norm(mu)
        cents_list.append(mu / norm if norm > 1e-12 else mu)
    return np.stack(cents_list), class_names


def upper_triangle_from_matrix(mat):
    """Upper triangle (off-diagonal) of a square matrix."""
    return mat[np.triu_indices(mat.shape[0], k=1)]


def cosine_matrix(cents):
    """cents: (k, d) matrix of L2-normalized centroids. Returns (k, k) cosine mat."""
    return cents @ cents.T


def mantel_rho(u1, u2):
    r, _ = pearsonr(u1, u2)
    return float(r)


def bootstrap_mantel(clip_cents, qwen_cents, class_names, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    n = len(class_names)
    rho_samples = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        # unique classes → indices into precomputed centroids
        unique_idx = np.unique(idx)
        if len(unique_idx) < 3:
            continue
        c1 = clip_cents[unique_idx]
        c2 = qwen_cents[unique_idx]
        mat1 = c1 @ c1.T
        mat2 = c2 @ c2.T
        u1 = upper_triangle_from_matrix(mat1)
        u2 = upper_triangle_from_matrix(mat2)
        rho_samples.append(mantel_rho(u1, u2))
    rho_samples = np.array(rho_samples)
    return {
        "mean": float(np.mean(rho_samples)),
        "median": float(np.median(rho_samples)),
        "ci_lo": float(np.percentile(rho_samples, 2.5)),
        "ci_hi": float(np.percentile(rho_samples, 97.5)),
        "n_valid": len(rho_samples),
    }


def subsample_mantel(clip_cents, qwen_cents, class_names, k, n_subs=N_SUBSAMPLE, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    n = len(class_names)
    rho_samples = []
    for _ in range(n_subs):
        idx = rng.choice(n, k, replace=False)
        c1 = clip_cents[idx]
        c2 = qwen_cents[idx]
        mat1 = c1 @ c1.T
        mat2 = c2 @ c2.T
        u1 = upper_triangle_from_matrix(mat1)
        u2 = upper_triangle_from_matrix(mat2)
        rho_samples.append(mantel_rho(u1, u2))
    rho_samples = np.array(rho_samples)
    return {
        "k": k,
        "mean": float(np.mean(rho_samples)),
        "median": float(np.median(rho_samples)),
        "ci_lo": float(np.percentile(rho_samples, 2.5)),
        "ci_hi": float(np.percentile(rho_samples, 97.5)),
        "n_subsamples": n_subs,
    }


def main():
    print("Loading features...", flush=True)
    clip = np.load(BASE / "results" / "exp6_clip_features.npz")
    qwen = np.load(BASE / "results" / "exp6_qwen_features.npz")
    class_names = sorted(clip.keys())
    print(f"  {len(class_names)} classes", flush=True)

    # Precompute L2-normalized centroids
    print("Precomputing centroids...", flush=True)
    clip_cents, _ = precompute_centroids(clip, class_names)
    qwen_cents, _ = precompute_centroids(qwen, class_names)
    print(f"  CLIP: {clip_cents.shape}  QwenViT: {qwen_cents.shape}", flush=True)

    # Full-point Mantel
    mat1 = clip_cents @ clip_cents.T
    mat2 = qwen_cents @ qwen_cents.T
    u1_all = upper_triangle_from_matrix(mat1)
    u2_all = upper_triangle_from_matrix(mat2)
    rho_full = mantel_rho(u1_all, u2_all)
    print(f"Full Mantel ρ = {rho_full:.4f} ({len(u1_all)} class pairs)", flush=True)

    # Bootstrap
    print(f"\nBootstrap ({N_BOOT} class resamples)...", flush=True)
    boot = bootstrap_mantel(clip_cents, qwen_cents, class_names)
    print(f"  mean={boot['mean']:.4f}  CI=[{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}]", flush=True)

    # Subsampling
    print("\nSubsampling...", flush=True)
    subs = []
    for k in [8, 10, 15, 20, 25, 30, 40]:
        s = subsample_mantel(clip_cents, qwen_cents, class_names, k)
        subs.append(s)
        print(f"  k={k:2d}: mean={s['mean']:.4f}  CI=[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]", flush=True)

    out = {
        "full_mantel_rho": rho_full,
        "n_class_pairs": len(u1_all),
        "bootstrap_class_resampling": boot,
        "subsampling": subs,
        "n_boot": N_BOOT,
        "n_subsample": N_SUBSAMPLE,
        "seed": RNG_SEED,
    }
    path = OUT_DIR / "mantel_bootstrap.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
