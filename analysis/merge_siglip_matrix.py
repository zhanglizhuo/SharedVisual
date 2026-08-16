"""Merge SigLIP class-similarity matrix into class_similarity_matrices.json.

Reproduction glue: experiments/exp2_class_similarity_matrices.py regenerates
the 4-encoder matrix set; experiments/exp_siglip_geometry.py produces
the SigLIP 8x8 matrix in results/siglip_geometry.json. This script
merges the SigLIP matrix (reordered to match the existing class_names order)
into the canonical class_similarity_matrices.json, so that
analysis/mantel_test.py then produces the full 5-encoder Mantel results.

Idempotent: if SigLIP is already present, it is overwritten in place.

Run locally after both feature-extraction scripts have produced their outputs:
  python analysis/merge_siglip_matrix.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAT_path = ROOT / "results" / "class_similarity_matrices.json"
SIG_path = ROOT / "results" / "siglip_geometry.json"
SIG_KEY = "SigLIP (ViT-SO400M)"


def main():
    mats = json.load(open(MAT_path))
    sig = json.load(open(SIG_path))

    target_order = mats["class_names"]
    sig_order = sig["class_names"]
    if set(target_order) != set(sig_order):
        raise SystemExit(f"class-name mismatch: {set(target_order)} vs {set(sig_order)}")

    # Reorder SigLIP matrix rows/cols to match the canonical class order
    sig_mat = sig["siglip_mantel_matrix"]
    idx = [sig_order.index(c) for c in target_order]
    reordered = [[sig_mat[i][j] for j in idx] for i in idx]

    mats["matrices"][SIG_KEY] = reordered
    with open(MAT_path, "w") as f:
        json.dump(mats, f, indent=2)
    print(f"Merged SigLIP matrix into {MAT_path}")
    print(f"Encoders now: {list(mats['matrices'].keys())}")
    print("Next: python analysis/mantel_test.py")


if __name__ == "__main__":
    main()
