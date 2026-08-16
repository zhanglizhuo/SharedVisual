"""
Class Geometry Analysis: Are class-neighborhood structures shared across encoders?
Computes class centroid similarity matrices for each encoder, then compares across encoders.

Outputs:
  - results/class_geometry_analysis.json   (cross-encoder geometry + image-pair RSA)

Note: results/class_similarity_matrices.json (the Mantel test input) is produced
exclusively by experiments/exp2_class_similarity_matrices.py; this script does
NOT write it to avoid conflicting producers.
"""
import os, json, torch, numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
import clip
import open_clip
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

CLASSES = ["guide", "answer", "On-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackBoard"]
N_PER_CLASS = 50
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


def extract_features():
    """Extract features from all encoders for same 160 images (20/class)."""
    device = "cuda"

    with open(RESULTS_DIR / "scb5_true_class_mapping.json") as f:
        all_mapping = json.load(f)

    # 20 images per class
    image_paths, img_labels = [], {}
    counts = {c: 0 for c in CLASSES}
    for p, v in all_mapping.items():
        cls = v["true_class"]
        if cls not in CLASSES or counts[cls] >= N_PER_CLASS:
            continue
        pth = str(SCB5_BASE / Path(p).name)
        if Path(pth).exists():
            image_paths.append(pth)
            img_labels[pth] = cls
            counts[cls] += 1
    print("Images:", len(image_paths), "per class:", dict(counts))

    # Labels array aligned with image_paths
    labels = [CLASSES.index(img_labels[p]) for p in image_paths]
    labels_t = torch.tensor(labels)

    # CLIP variants
    clip_models = [
        ("CLIP ViT-L/14 (OpenAI)", "clip", "ViT-L/14", "openai"),
        ("CLIP ViT-L/14 (LAION)", "open_clip", "ViT-L-14", "laion2b_s32b_b82k"),
        ("CLIP ViT-B/32 (OpenAI)", "clip", "ViT-B/32", "openai"),
    ]

    all_feats, all_names = {}, []

    for name, backend, mname, pretrained in clip_models:
        print("\nLoading %s..." % name)
        if backend == "clip":
            model, preprocess = clip.load(mname, device=device)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(mname, pretrained=pretrained, device=device)

        feats = []
        for i in range(0, len(image_paths), 32):
            batch = image_paths[i:i+32]
            images = [Image.open(p).convert("RGB") for p in batch]
            tensors = torch.stack([preprocess(img) for img in images]).to(device)
            with torch.no_grad():
                f = model.encode_image(tensors)
                f = (f / f.norm(dim=-1, keepdim=True)).cpu()
            feats.append(f)
        all_feats[name] = torch.cat(feats)
        all_names.append(name)
        print("  shape:", all_feats[name].shape)

    # Qwen2-VL vision features
    print("\nLoading Qwen2-VL-7B...")
    qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16,
        device_map="auto", attn_implementation="eager", local_files_only=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", local_files_only=True)
    qwen.eval()

    print("Extracting Qwen2-VL vision features...")
    qwen_feats = []
    for p in tqdm(image_paths):
        img = Image.open(p).convert("RGB").resize((224, 224))
        conv = [{"role": "user", "content": [
            {"type": "image", "max_pixels": 224 * 224},
            {"type": "text", "text": "x"},
        ]}]
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            vision_out = qwen.visual(inputs["pixel_values"], inputs["image_grid_thw"])
            pooled = vision_out.mean(dim=0, keepdim=True)
            qwen_feats.append((pooled / pooled.norm(dim=-1, keepdim=True)).cpu())
    qwen_feats = torch.cat(qwen_feats)
    all_feats["Qwen2-VL Vision"] = qwen_feats
    all_names.append("Qwen2-VL Vision")
    print("  shape:", qwen_feats.shape)

    del qwen
    torch.cuda.empty_cache()

    return all_feats, all_names, labels_t


def compute_class_centroids(feats, labels, n_classes):
    """Compute class centroid (mean feature) per class."""
    centroids = []
    for c in range(n_classes):
        mask = labels == c
        centroids.append(feats[mask].mean(dim=0))
    return torch.stack(centroids)


def compute_similarity_matrix(centroids):
    """Compute N×N cosine similarity matrix from N class centroids."""
    centroids = centroids / centroids.norm(dim=-1, keepdim=True)
    return centroids @ centroids.T


def compare_matrices(m1, m2):
    """Compare two similarity matrices using Spearman correlation of upper triangle."""
    n = m1.shape[0]
    triu_idx = torch.triu_indices(n, n, offset=1)
    v1 = m1[triu_idx[0], triu_idx[1]].numpy()
    v2 = m2[triu_idx[0], triu_idx[1]].numpy()
    r, p = spearmanr(v1, v2)
    return r, p


def compute_image_pair_rsa(feats_a, feats_b):
    """Standard RSA: correlate upper triangles of pairwise image similarity matrices."""
    n = feats_a.shape[0]
    sim_a = feats_a @ feats_a.T
    sim_b = feats_b @ feats_b.T
    triu = torch.triu_indices(n, n, offset=1)
    v_a = sim_a[triu[0], triu[1]].numpy()
    v_b = sim_b[triu[0], triu[1]].numpy()
    r, p = spearmanr(v_a, v_b)
    return r, p


def main():
    print("=" * 60)
    print("Class Geometry Analysis")
    print("=" * 60)

    all_feats, all_names, labels = extract_features()
    n_classes = len(CLASSES)

    # Compute class centroids and similarity matrices
    sim_matrices = {}
    for name in all_names:
        centroids = compute_class_centroids(all_feats[name], labels, n_classes)
        sim_matrices[name] = compute_similarity_matrix(centroids)

    # Print per-class similarity matrices
    print("\n=== Class Similarity Matrices (upper triangle) ===")
    for name in all_names:
        sm = sim_matrices[name]
        print(f"\n{name}:")
        header = "          " + "  ".join(f"{c[:10]:>10s}" for c in CLASSES)
        print(header)
        for i, ci in enumerate(CLASSES):
            row = f"{ci[:10]:>10s}"
            for j in range(n_classes):
                row += f"  {sm[i,j]:.3f}"
            print(row)

    # Cross-encoder comparison
    print("\n=== Cross-Encoder Class Geometry Consistency ===")
    print(f"{'Encoder A':35s} {'Encoder B':35s} {'RSA':>8s} {'p':>8s}")
    print("-" * 90)
    results = []
    for i in range(len(all_names)):
        for j in range(i + 1, len(all_names)):
            r, p = compare_matrices(sim_matrices[all_names[i]], sim_matrices[all_names[j]])
            results.append({"encoder_a": all_names[i], "encoder_b": all_names[j],
                           "spearman_r": round(r, 3), "p_value": round(p, 4)})
            print(f"{all_names[i]:35s} {all_names[j]:35s} {r:>8.3f} {p:>8.4f}")

    # Compare full (diag+off-diag) vs off-diagonal only
    print("\n=== Cross-Encoder Consistency (off-diagonal only) ===")
    for i in range(len(all_names)):
        for j in range(i + 1, len(all_names)):
            sm_i, sm_j = sim_matrices[all_names[i]], sim_matrices[all_names[j]]
            n = len(CLASSES)
            triu = torch.triu_indices(n, n, offset=1)
            v_i = sm_i[triu[0], triu[1]].numpy()
            v_j = sm_j[triu[0], triu[1]].numpy()
            r, p = spearmanr(v_i, v_j)
            print(f"  {all_names[i][:30]:30s} vs {all_names[j][:30]:30s}  ρ={r:.3f} (p={p:.4f})")

    # Standard image-pair RSA
    print("\n=== Standard Image-Pair RSA ===")
    rsa_values = {}
    for i in range(len(all_names)):
        for j in range(i + 1, len(all_names)):
            r, p = compute_image_pair_rsa(all_feats[all_names[i]], all_feats[all_names[j]])
            rsa_values[f"{all_names[i]} vs {all_names[j]}"] = {"spearman_r": round(r, 4), "p_value": round(p, 4)}
            print(f"  {all_names[i]:35s} vs {all_names[j]:35s}  ρ={r:.3f} (p={p:.4f})")

    # Save
    out = {
        "n_classes": n_classes,
        "n_per_class": N_PER_CLASS,
        "class_names": CLASSES,
        "encoders": all_names,
        "cross_encoder_comparison": results,
        "image_pair_rsa": rsa_values,
    }
    out_path = RESULTS_DIR / "class_geometry_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
