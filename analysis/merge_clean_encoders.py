"""Merge ResNet-50 + DINOv2 class-similarity matrices into class_similarity_matrices.json.

Reproduction glue (companion to merge_siglip_matrix.py): after
experiments/exp_clean_encoder_geometry.py produces the two clean-encoder
8x8 matrices in results/clean_encoder_geometry.json, this script merges
them (reordered to the canonical class_names order) into
results/class_similarity_matrices.json, so that analysis/mantel_test.py then
produces the full 7-encoder Mantel results including the H2 test.

Idempotent: if either encoder is already present, it is overwritten in place.

Run locally after V100 feature extraction:
  python analysis/merge_clean_encoders.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAT_path = ROOT / "results" / "class_similarity_matrices.json"
CLEAN_path = ROOT / "results" / "clean_encoder_geometry.json"
NEW_KEYS = ["ResNet-50 (ImageNet-1k)", "DINOv2 ViT-B/14 (LVD-142M)"]


def main():
    mats = json.load(open(MAT_path))
    clean = json.load(open(CLEAN_path))

    target_order = mats["class_names"]
    clean_order = clean["class_names"]
    if set(target_order) != set(clean_order):
        raise SystemExit(f"class-name mismatch: {set(target_order)} vs {set(clean_order)}")

    idx = [clean_order.index(c) for c in target_order]
    for key in NEW_KEYS:
        mat = clean["mantel_matrices"][key]
        reordered = [[mat[i][j] for j in idx] for i in idx]
        mats["matrices"][key] = reordered

    with open(MAT_path, "w") as f:
        json.dump(mats, f, indent=2)
    print(f"Merged {NEW_KEYS} into {MAT_path}")
    print(f"Encoders now: {list(mats['matrices'].keys())}")
    print("Next: python analysis/mantel_test.py")


if __name__ == "__main__":
    main()
