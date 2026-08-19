#!/usr/bin/env python3
"""Encoder-swap stage 1: extract and cache per-image vision-encoder features.

Arms: C = CLIP ViT-L/14 LAION-2B, S = SigLIP ViT-SO400M-14 (webli),
      D = DINOv2 ViT-B/14, R = randomly initialized ViT-B/14 (seed 42, control).
Splits: SCB5 train pool (leakage-cleaned, exp1b discipline) + SCB5 val,
        full-frame images, each encoder's canonical preprocessing.

Output: results/encoder_swap/features/{arm}_{split}.npz  (fp16 features + labels +
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
OUT_DIR = RESULTS / "encoder_swap" / "features"

ARMS = ["clip_l_laion", "siglip_so400m", "dinov2_b14", "random_vitb"]
SEED = 42

DATASETS = ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]


def dataset_paths():
    """SCB5 dataset roots (V100 layout, verified 2026-08-15 against both the
    mapping keys and the server directory structure)."""
    base = os.environ.get("SHAREDVISUAL_SCB5",
                          str(Path.home() / "works/lizhuo/AutoResearchClaw/datasets_scb"))
    return {
        "TeacherBehavior": Path(base) / "SCB5_TeacherBehavior" /
            "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2" / "images",
        "HandriseReadWrite": Path(base) / "SCB5_HandriseReadWrite" /
            "SCB5-Handrise-Read-write-2024-9-17" / "images",
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
        # reuse: run its logic (kept import-light here; see run_encoder_swap.sh which
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
        # local load (V100: repo + weights pre-cached; avoids network flakiness
        # and the py3.8 union-type incompatibility of the hub repo)
        hub_dir = Path(os.environ.get("TORCH_HOME", str(Path.home()/".cache/torch"))) / "hub"
        repo = hub_dir / "facebookresearch_dinov2_main"
        ckpt = hub_dir / "checkpoints" / "dinov2_vitb14_pretrain.pth"
        if repo.exists() and ckpt.exists():
            model = torch.hub.load(str(repo), "dinov2_vitb14",
                                   source="local", pretrained=False)
            model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        else:
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
            lbl = f.parents[2] / "labels" / "train" / (f.stem + ".txt")
            files[i] = (f, parse_train_label(lbl, ds), ds, f.name)
        n_labeled = sum(1 for (_, c, _, _) in files if c is not None)
        if split == "train" and n_labeled == 0:
            raise RuntimeError(
                f"train labels all None for this enumeration — label path "
                f"resolution broken; refusing to write a useless npz "
                f"(first file: {files[0][0]}, tried: "
                f"{files[0][0].parents[2] / 'labels' / 'train' / (files[0][0].stem + '.txt')})")

    with torch.no_grad():
        for f, y_cls, ds, fname in files:
            if not f.exists():
                continue
            try:
                img = preprocess(Image.open(f).convert("RGB"))
            except Exception:
                continue
            x = img.unsqueeze(0).to(device)
            if family == "open_clip":
                v = model.visual
                if hasattr(v, "trunk"):
                    # TimmModel wrapper (newer open_clip): trunk.__call__
                    # returns pooled; forward_features gives pre-pool tokens
                    tok = v.trunk.forward_features(x)   # (1,730,1152) SO400M
                else:
                    # old-style VisionTransformer returns pooled; rebuild
                    # the token stream: conv1->flatten->CLS->pos->ln_pre->
                    # transformer->ln_post
                    y = v.conv1(x)                             # (1,D,14,14)
                    y = y.reshape(y.shape[0], y.shape[1], -1).permute(0, 2, 1)
                    cls_emb = v.class_embedding.to(y.dtype).view(1, 1, -1)
                    y = torch.cat([cls_emb, y], dim=1)         # prepend CLS
                    y = y + v.positional_embedding.to(y.dtype)
                    y = v.patch_dropout(y) if hasattr(v, "patch_dropout") else y
                    y = v.ln_pre(y)
                    y = v.transformer(y)
                    tok = v.ln_post(y)                          # (1,257,1024)
            elif family == "dinov2":
                out = model.forward_features(x)   # dict in this repo version
                tok = out["x_norm_patchtokens"] if isinstance(out, dict) else out
                # (1, 256, 768) patch tokens; prepend CLS for parity with
                # open_clip token streams (257 total)
                if isinstance(out, dict):
                    cls_tok = out["x_norm_clstoken"].unsqueeze(1)
                    tok = torch.cat([cls_tok, tok], dim=1)
            feats.append(tok[0].to(torch.float16).cpu().numpy())
            assert not torch.is_tensor(y_cls), f"label must not be a tensor: {y_cls}"
            labels.append(y_cls)
            img_ids.append(f"{ds}/{fname}")
            if len(feats) % 512 == 0:
                print(f"  {arm}/{split}: {len(feats)} images", flush=True)

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
