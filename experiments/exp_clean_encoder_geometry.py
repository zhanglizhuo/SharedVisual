"""Clean-encoder geometry test: address H2 (shared-data hypothesis).

Why this exists: the paper's main open weakness (Limitations, point 2) is that
all five encoders tested (CLIP x3, SigLIP, QwenViT) were trained on Internet-scale
data that may overlap, so the shared class geometry could in principle be an
artifact of training-data overlap rather than a property of the natural image
distribution. This script tests that confound by adding two encoders with
constrained, well-characterized training data:

  - ResNet-50 (supervised, ImageNet-1k only, 1.2M images / 1000 classes):
    training boundary fully public, ~0 overlap with CLIP's WIT-400M / MLLM web
    corpora. The cleanest available test of "is it the data distribution?"
  - DINOv2 ViT-B/14 (self-supervised, LVD-142M, partially derived from LAION):
    modern self-supervised, partially overlaps with web data; a useful
    intermediate point between fully-clean (ResNet) and fully-web (CLIP).

If ResNet-50 (clean) still shows high Mantel rho with CLIP/QwenViT/SigLIP on
SCB5 class geometry, the shared-data alternative is substantially weakened.

Outputs (results/clean_encoder_geometry.json):
  - ResNet-50 8x8 class-centroid matrix (50 imgs/class)
  - DINOv2 8x8 class-centroid matrix (50 imgs/class)
  - 7-encoder CKA matrix (5 prior + ResNet-50 + DINOv2) on 160 imgs (20/class)

Run on V100 (dl_env):
  python experiments/exp_clean_encoder_geometry.py
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import torch
import clip
import open_clip
import numpy as np
import torchvision
from torchvision import transforms as T
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

CLASSES = ["guide", "answer", "On-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackBoard"]
RESULTS_DIR = Path(os.environ.get("SHAREDVISUAL_RESULTS", str(Path(__file__).resolve().parents[1] / "results")))
SCB5_BASE = (Path(os.environ.get("SHAREDVISUAL_DATA", str(Path.home() / "data")))
             / "SCB5_TeacherBehavior"
             / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2"
             / "images" / "val")
N_MANTEL = 50
N_CKA = 20
SIGLIP_NAME = "ViT-SO400M-14-SigLIP-384"
SIGLIP_PRETRAINED = "webli"


def linear_cka(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    XXT = X @ X.T
    YYT = Y @ Y.T
    hsic = (XXT * YYT).sum()
    hsic_xx = (XXT * XXT).sum()
    hsic_yy = (YYT * YYT).sum()
    return (hsic / (hsic_xx.sqrt() * hsic_yy.sqrt())).item()


def class_similarity_matrix(feat_dict):
    classes_sorted = sorted(feat_dict.keys())
    centroids = []
    for c in classes_sorted:
        mu = np.array(feat_dict[c], dtype=np.float64).mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        centroids.append(mu)
    C = np.array(centroids)
    return classes_sorted, (C @ C.T).tolist()


def select_images(n_per_class):
    with open(RESULTS_DIR / "scb5_true_class_mapping.json") as f:
        all_mapping = json.load(f)
    image_paths, img_to_label = [], {}
    counts = {c: 0 for c in CLASSES}
    for p, v in all_mapping.items():
        cls = v["true_class"]
        if cls not in CLASSES or counts[cls] >= n_per_class:
            continue
        img_path = str(SCB5_BASE / Path(p).name)
        if Path(img_path).exists():
            image_paths.append(img_path)
            img_to_label[img_path] = cls
            counts[cls] += 1
    return image_paths, img_to_label, counts


def extract_clip(name, backend, mname, pretrained, image_paths, device, batch=32):
    if backend == "clip":
        model, preprocess = clip.load(mname, device=device)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            mname, pretrained=pretrained, device=device)
    feats = []
    for i in range(0, len(image_paths), batch):
        batch_paths = image_paths[i:i + batch]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        t = torch.stack([preprocess(img) for img in images]).to(device)
        with torch.no_grad():
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    return torch.cat(feats)


def extract_qwen(image_paths, device):
    qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16,
        device_map="auto", attn_implementation="eager")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    qwen.eval()
    feats = []
    for p in tqdm(image_paths, desc="QwenViT"):
        img = Image.open(p).convert("RGB").resize((224, 224))
        conv = [{"role": "user", "content": [
            {"type": "image", "max_pixels": 224 * 224},
            {"type": "text", "text": "x"}]}]
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            out = qwen.visual(inputs["pixel_values"], inputs["image_grid_thw"])
            pooled = out.mean(dim=0, keepdim=True)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        feats.append(pooled.cpu())
    del qwen
    torch.cuda.empty_cache()
    return torch.cat(feats)


def extract_siglip(image_paths, device, batch=16):
    print("  Loading SigLIP ViT-SO400M-384 (open_clip, webli weights)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        SIGLIP_NAME, pretrained=SIGLIP_PRETRAINED, device=device)
    model.eval()
    feats = []
    for i in tqdm(range(0, len(image_paths), batch), desc="SigLIP"):
        batch_paths = image_paths[i:i + batch]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        t = torch.stack([preprocess(img) for img in images]).to(device)
        with torch.no_grad():
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(feats)


def extract_resnet50(image_paths, device, batch=32):
    """Supervised ResNet-50 on ImageNet-1k (clean training boundary)."""
    print("  Loading ResNet-50 (torchvision IMAGENET1K_V2)...")
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights).to(device).eval()
    preprocess = weights.transforms()  # standard ImageNet V2 transform
    feats = []
    for i in range(0, len(image_paths), batch):
        batch_paths = image_paths[i:i + batch]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        t = torch.stack([preprocess(img) for img in images]).to(device)
        with torch.no_grad():
            # Forward up to avgpool, take 2048-d feature (drop fc)
            x = model.conv1(t)
            x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
            x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
            x = model.avgpool(x)
            f = torch.flatten(x, 1)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(feats)


def extract_dinov2(image_paths, device, batch=8):
    """Self-supervised DINOv2 ViT-B/14 (LVD-142M, partial LAION overlap).

    Built via timm architecture (vit_base_patch14_dinov2.lvd142m) and loaded
    from the cached original dinov2_vitb14_pretrain.pth weights, because the
    facebookresearch/dinov2 hub code uses PEP 604 type unions (Python 3.10+)
    that fail on the V100's Python 3.8 environment. timm's architecture is
    key-compatible (174/174) with the original DINOv2 state dict.
    """
    import timm
    print("  Loading DINOv2 ViT-B/14 (timm arch + cached dinov2_vitb14 weights, native 518)...")
    # Native 518x518 (DINOv2 training resolution); 518/14=37 -> 1369 patches + 1 cls = 1370 tokens.
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False).to(device).eval()
    weights_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth")
    sd = torch.load(weights_path, map_location=device)
    sd.pop("mask_token", None)
    model.load_state_dict(sd, strict=True)
    # DINOv2 native preprocessing: resize 518, ImageNet norm
    tfm = T.Compose([
        T.Resize(518, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(518),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    feats = []
    for i in tqdm(range(0, len(image_paths), batch), desc="DINOv2"):
        batch_paths = image_paths[i:i + batch]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        t = torch.stack([tfm(img) for img in images]).to(device)
        with torch.no_grad():
            f = model.forward_features(t)
            f = f.mean(dim=1) if f.dim() == 3 else f  # pool patches -> (B, D)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(feats)


def main():
    print("=" * 60)
    print("Clean-encoder geometry: ResNet-50 (ImageNet) + DINOv2 (LVD-142M)")
    print("Tests H2: does shared class geometry require shared training data?")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img50, lab50, cnt50 = select_images(N_MANTEL)
    print(f"Mantel: {len(img50)} images, per class: {cnt50}")
    img20, lab20, cnt20 = select_images(N_CKA)
    print(f"CKA:    {len(img20)} images, per class: {cnt20}")
    assert set(img20).issubset(set(img50)), "CKA 20-set must be subset of Mantel 50-set"

    # === Mantel: extract ResNet-50 + DINOv2 on 50/class ===
    print("\nExtracting clean-encoder features (50/class for Mantel)...")
    clean_feats = {
        "ResNet-50 (ImageNet-1k)": extract_resnet50(img50, device),
        "DINOv2 ViT-B/14 (LVD-142M)": extract_dinov2(img50, device),
    }

    mantel_matrices = {}
    for name, feats in clean_feats.items():
        by_class = {c: [] for c in CLASSES}
        for p, v in zip(img50, feats.numpy()):
            by_class[lab50[p]].append(v)
        by_class = {c: np.array(v) for c, v in by_class.items()}
        classes_sorted, matrix = class_similarity_matrix(by_class)
        assert classes_sorted == sorted(CLASSES)
        mantel_matrices[name] = matrix
        print(f"  {name}: 8x8 matrix computed")

    # === CKA: re-extract all 7 encoders on 20/class ===
    print("\nExtracting features (20/class for full 7-encoder CKA)...")
    feats_cka = {}
    feats_cka["CLIP ViT-L/14 (OpenAI)"] = extract_clip("CLIP ViT-L/14 (OpenAI)", "clip", "ViT-L/14", "openai", img20, device)
    feats_cka["CLIP ViT-L/14 (LAION)"] = extract_clip("CLIP ViT-L/14 (LAION)", "open_clip", "ViT-L-14", "laion2b_s32b_b82k", img20, device)
    feats_cka["CLIP ViT-B/32 (OpenAI)"] = extract_clip("CLIP ViT-B/32 (OpenAI)", "clip", "ViT-B/32", "openai", img20, device)
    feats_cka["Qwen2-VL Vision"] = extract_qwen(img20, device)
    feats_cka["SigLIP (ViT-SO400M)"] = extract_siglip(img20, device)
    feats_cka["ResNet-50 (ImageNet-1k)"] = extract_resnet50(img20, device)
    feats_cka["DINOv2 ViT-B/14 (LVD-142M)"] = extract_dinov2(img20, device)

    names = list(feats_cka.keys())
    cka_matrix = {}
    print("\n=== 7-encoder CKA matrix ===")
    for i, ni in enumerate(names):
        cka_matrix[ni] = {}
        row = []
        for j, nj in enumerate(names):
            if i == j:
                s = 1.0
            else:
                s = round(linear_cka(feats_cka[ni], feats_cka[nj]), 4)
            cka_matrix[ni][nj] = s
            row.append(f"{s:.4f}")
        print(f"  {ni[:28]:<28} " + "  ".join(f"{x:>7}" for x in row))

    out = {
        "description": "Clean-encoder geometry test for H2 (shared-data hypothesis)",
        "encoders": {
            "ResNet-50 (ImageNet-1k)": "supervised, ImageNet-1k only (1.2M imgs, ~0 overlap with WIT-400M)",
            "DINOv2 ViT-B/14 (LVD-142M)": "self-supervised, LVD-142M (partial LAION overlap)",
        },
        "n_per_class_mantel": N_MANTEL,
        "n_per_class_cka": N_CKA,
        "class_names": sorted(CLASSES),
        "mantel_matrices": mantel_matrices,
        "cka_matrix": cka_matrix,
    }
    out_path = RESULTS_DIR / "clean_encoder_geometry.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("\nNext (local): merge both matrices via analysis/merge_clean_encoders.py,")
    print("rerun analysis/mantel_test.py -> see if ResNet-50 (clean) shares class geometry.")


if __name__ == "__main__":
    main()
