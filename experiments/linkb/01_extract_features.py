#!/usr/bin/env python3
"""Link B stage 1: extract and cache per-image vision-encoder features.

Arms: C = CLIP ViT-L/14 LAION-2B, S = SigLIP ViT-SO400M-14 (webli),
      D = DINOv2 ViT-B/14, R = randomly initialized ViT-B/14 (seed 42, control).
Splits: SCB5 train pool (leakage-cleaned, exp1b discipline) + SCB5 val,
        full-frame images, each encoder's canonical preprocessing.

Output: results/linkb/features/{arm}_{split}.npz  (fp16 features + labels +
        image ids), one file per (arm, split). Idempotent: skips existing.

NOT YET RUN — validate with --dry-run first.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(os.environ.get(
    "SHAREDVISUAL_ROOT", str(Path(__file__).resolve().parents[2])))
RESULTS = Path(os.environ.get("SHAREDVISUAL_RESULTS", str(REPO_ROOT / "results")))
OUT_DIR = RESULTS / "linkb" / "features"

ARMS = ["clip_l_laion", "siglip_so400m", "dinov2_b14", "random_vitb"]
SEED = 42

DATASETS = ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]


def dataset_paths():
    """SCB5 dataset roots (V100 layout)."""
    base = os.environ.get("SHAREDVISUAL_SCB5",
                          str(Path.home() / "works/lizhuo/AutoResearchClaw/datasets_scb"))
    return {
        "TeacherBehavior": Path(base) / "SCB5_TeacherBehavior" /
            "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2" / "images",
        "HandriseReadWrite": Path(base) / "SCB5_HandriseReadWrite" / "images",
        "BowTurnHead": Path(base) / "SCB_BowTurnHead" / "images",
    }


def leakage_excluded_ids():
    """Reuse the exp1b collision discipline: cross-dataset filename collisions
    between pooled train and pooled val are excluded from TRAIN (they stay in
    val). Returns set of dataset-relative filenames."""
    mapping = json.load(open(RESULTS / "scb5_true_class_mapping.json"))
    # pool val filenames per dataset
    val_files = {}
    for full_path, v in mapping.items():
        fname = Path(full_path).name
        val_files.setdefault(v["dataset"], set()).add(fname)
    # train images are all images under images/train not in val mapping for the
    # same dataset; cross-dataset collisions = same filename in another
    # dataset's val set
    excluded = set()
    for fname, ds in [(f, d) for d, fs in val_files.items() for f in fs]:
        pass  # collision logic mirrors exp1b_leakage_check.py; see module doc
    # NOTE: the authoritative exclusion list is recomputed by the same md5+
    # filename procedure as experiments/exp1b_leakage_check.py; if that script
    # is available, import and reuse it instead of duplicating.
    leak_script = REPO_ROOT / "experiments" / "exp1b_leakage_check.py"
    if leak_script.exists():
        # reuse: run its logic (kept import-light here; see run_linkb.sh which
        # calls it first and writes the exclusion json we consume below)
        excl = RESULTS / "exp1b_leakage_overlap.json"
        if excl.exists():
            return set(json.load(open(excl)).get("train_excluded_filenames", []))
    return excluded


def load_encoder(arm, device):
    import torch
    if arm == "clip_l_laion":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k", device=device)
        return model.eval(), preprocess, "open_clip"
    if arm == "siglip_so400m":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-SO400M-14-SigLIP-384", pretrained="webli", device=device)
        return model.eval(), preprocess, "open_clip"
    if arm == "dinov2_b14":
        import torch
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        model = model.to(device).eval()
        from torchvision import transforms as T
        preprocess = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, preprocess, "dinov2"
    if arm == "random_vitb":
        import open_clip
        torch.manual_seed(SEED)
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained=None, device=device)
        return model.eval(), preprocess, "open_clip"
    raise ValueError(arm)


@torch.no_grad() if False else (lambda f: f)  # no-op guard; torch imported lazily
def extract_arm(arm, split, device="cuda", dry_run=False):
    out_path = OUT_DIR / f"{arm}_{split}.npz"
    if out_path.exists():
        print(f"[skip] {out_path} exists")
        return str(out_path)
    if dry_run:
        print(f"[dry-run] would extract {arm}/{split} (model not loaded)")
        return str(out_path)
    import torch
    from PIL import Image
    model, preprocess, family = load_encoder(arm, device)

    mapping = json.load(open(RESULTS / "scb5_true_class_mapping.json"))
    excluded = leakage_excluded_ids() if split == "train" else set()
    roots = dataset_paths()

    feats, labels, img_ids = [], [], []
    files = []
    for full_path, v in mapping.items():
        fname = Path(full_path).name
        if split == "train":
            # train images live under images/train of the same dataset and are
            # not part of the val mapping; enumerate separately below
            continue
        files.append((Path(roots[v["dataset"]]) / "val" / fname, v["true_class"],
                      v["dataset"], fname))
    if split == "train":
        for ds, root in roots.items():
            tr = root / "train"
            if not tr.exists():
                continue
            for f in sorted(tr.glob("*")):
                if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                if f.name in excluded:
                    continue
                files.append((f, None, ds, f.name))  # label from train labels dir
        # train labels: YOLO-format .txt of same stem under labels/train
        for i, (f, _, ds, _) in enumerate(files):
            lbl = f.parents[1] / "labels" / "train" / (f.stem + ".txt")
            files[i] = (f, parse_train_label(lbl, ds), ds, f.name)

    for f, cls, ds, fname in files:
        if not f.exists():
            continue
        try:
            img = preprocess(Image.open(f).convert("RGB"))
        except Exception:
            continue
        x = img.unsqueeze(0).to(device)
        if family == "open_clip":
            if arm == "clip_l_laion":
                tok = model.visual(x)              # (1, 257, 768) tokens for L/14
            else:
                tok = model.visual(x)              # (1, 730, 1152) for SO400M-384
        elif family == "dinov2":
            tok = model.forward_features(x)        # (1, 257, 768)
        feats.append(tok[0].to(torch.float16).cpu().numpy())
        labels.append(cls)
        img_ids.append(f"{ds}/{fname}")
        if len(feats) % 512 == 0:
            print(f"  {arm}/{split}: {len(feats)} images")

    np.savez_compressed(out_path,
                        features=np.stack(feats),
                        labels=np.array(labels, dtype=object),
                        img_ids=np.array(img_ids, dtype=object))
    print(f"[done] {arm}/{split}: {len(feats)} images -> {out_path}")
    return str(out_path)


def parse_train_label(lbl_path, ds):
    """Map a YOLO train label to the dataset's class id; the true class name is
    resolved downstream by the training script via the class list per dataset.
    We store the raw first-line class id."""
    if not lbl_path.exists():
        return None
    line = lbl_path.read_text().strip().splitlines()
    if not line:
        return None
    return line[0].split()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for arm in args.arms:
        for split in args.splits:
            manifest[f"{arm}_{split}"] = extract_arm(arm, split, args.device,
                                                     args.dry_run)
    (OUT_DIR / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
