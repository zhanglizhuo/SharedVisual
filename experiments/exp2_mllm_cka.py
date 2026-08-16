import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
import clip
import open_clip
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from scipy.stats import spearmanr

CLASSES = ["guide", "answer", "On-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackBoard"]
REPO_ROOT = Path(os.environ.get(
    "SHAREDVISUAL_ROOT",
    str(Path(__file__).resolve().parents[1]),
))
RESULTS_DIR = Path(os.environ.get(
    "SHAREDVISUAL_RESULTS",
    str(REPO_ROOT / "results"),
))
SCB5_BASE = Path(os.environ.get(
    "SHAREDVISUAL_SCB5_TEACHER",
    str(Path(os.environ.get("SHAREDVISUAL_DATA", str(Path.home() / "data")))
        / "SCB5_TeacherBehavior"
        / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2"
        / "images" / "val"),
))
N_PER_CLASS = 20


def linear_cka(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    XXT = X @ X.T
    YYT = Y @ Y.T
    hsic = (XXT * YYT).sum()
    hsic_xx = (XXT * XXT).sum()
    hsic_yy = (YYT * YYT).sum()
    return (hsic / (hsic_xx.sqrt() * hsic_yy.sqrt())).item()


def main():
    print("=" * 60)
    print("CKA: CLIP variants vs Qwen2-VL Vision Encoder")
    print("=" * 60)

    device = "cuda"

    with open(RESULTS_DIR / "scb5_true_class_mapping.json") as f:
        all_mapping = json.load(f)

    image_paths = []
    img_to_label = {}
    counts = {c: 0 for c in CLASSES}
    for p, v in all_mapping.items():
        cls = v["true_class"]
        if cls not in CLASSES or counts[cls] >= N_PER_CLASS:
            continue
        img_path = str(SCB5_BASE / Path(p).name)
        if Path(img_path).exists():
            image_paths.append(img_path)
            img_to_label[img_path] = cls
            counts[cls] += 1

    print("Images:", len(image_paths), "per class:", dict(counts))

    # Extract CLIP features
    print("\nExtracting CLIP features...")
    clip_models = [
        ("CLIP ViT-L/14 (OpenAI)", "clip", "ViT-L/14", "openai"),
        ("CLIP ViT-L/14 (LAION)", "open_clip", "ViT-L-14", "laion2b_s32b_b82k"),
        ("CLIP ViT-B/32 (OpenAI)", "clip", "ViT-B/32", "openai"),
    ]

    clip_feats = {}
    for name, backend, mname, pretrained in clip_models:
        print("  Loading %s..." % name)
        if backend == "clip":
            model, preprocess = clip.load(mname, device=device)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                mname, pretrained=pretrained, device=device)

        feats = []
        for i in range(0, len(image_paths), 32):
            batch_paths = image_paths[i:i+32]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            image_tensors = torch.stack([preprocess(img) for img in images]).to(device)
            with torch.no_grad():
                f = model.encode_image(image_tensors)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu())
        clip_feats[name] = torch.cat(feats)
        print("    shape:", clip_feats[name].shape)

    # Extract Qwen2-VL vision features
    print("\nLoading Qwen2-VL-7B...")
    qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    qwen.eval()

    print("Extracting Qwen2-VL vision features...")
    qwen_feats_list = []
    for p in tqdm(image_paths):
        img = Image.open(p).convert("RGB")
        # Force small size to avoid OOM
        img = img.resize((224, 224))
        conv = [{"role": "user", "content": [
            {"type": "image", "max_pixels": 224 * 224},
            {"type": "text", "text": "x"},
        ]}]
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(device)

        with torch.no_grad():
            vision_out = qwen.visual(inputs["pixel_values"], inputs["image_grid_thw"])
            pooled = vision_out.mean(dim=0, keepdim=True)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
            qwen_feats_list.append(pooled.cpu())

    qwen_feats = torch.cat(qwen_feats_list)
    print("  Qwen features:", qwen_feats.shape)

    # CKA matrix
    print("\n=== CKA MATRIX ===")
    all_names = list(clip_feats.keys()) + ["Qwen2-VL Vision"]
    all_feats = list(clip_feats.values()) + [qwen_feats]
    n = len(all_names)

    cka_matrix = {}
    header = "  " + " " * 35
    for name in all_names:
        header += name[:20].center(20)
    print(header)
    for i in range(n):
        name_i = all_names[i][:33]
        line = "  " + name_i.ljust(33) + " "
        cka_matrix[all_names[i]] = {}
        for j in range(n):
            if i == j:
                score = 1.0
            else:
                score = linear_cka(all_feats[i], all_feats[j])
            cka_matrix[all_names[i]][all_names[j]] = round(score, 4)
            line += ("%.4f" % score).rjust(10) + "  "
        print(line)

    # Per-class within-similarity
    print("\n=== Per-class within-feature similarity ===")
    sim_matrices = {}
    for name, feats in list(clip_feats.items()) + [("Qwen2-VL Vision", qwen_feats)]:
        sim_matrices[name] = feats @ feats.T

    labels = [CLASSES.index(img_to_label[p]) for p in image_paths]
    labels_t = torch.tensor(labels)

    header2 = "  " + "Class".ljust(25)
    for name in all_names:
        header2 += "within_sim".rjust(14)
    print(header2)
    for c_idx, cls in enumerate(CLASSES):
        mask = labels_t == c_idx
        if mask.sum() <= 1:
            continue
        line = "  " + cls.ljust(25)
        for name in all_names:
            ws = sim_matrices[name][mask][:, mask].mean().item()
            line += ("%.4f" % ws).rjust(14)
        print(line)

    # RSA of per-class within-similarities
    print("\n=== Per-class within-sim Spearman correlation ===")
    for i in range(n):
        for j in range(i + 1, n):
            ws_i = []
            ws_j = []
            for c_idx in range(len(CLASSES)):
                mask = labels_t == c_idx
                if mask.sum() > 1:
                    ws_i.append(sim_matrices[all_names[i]][mask][:, mask].mean().item())
                    ws_j.append(sim_matrices[all_names[j]][mask][:, mask].mean().item())
            if len(ws_i) >= 3:
                r, p = spearmanr(ws_i, ws_j)
                ni = all_names[i][:30]
                nj = all_names[j][:30]
                print("  %-30s vs %-30s  r=%.3f (p=%.4f)" % (ni, nj, r, p))

    # Save
    out = {
        "n_images": len(image_paths),
        "n_per_class": N_PER_CLASS,
        "cka": cka_matrix,
    }
    out_path = RESULTS_DIR / "exp2_cka_all_encoders.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to", out_path)

    del qwen
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
