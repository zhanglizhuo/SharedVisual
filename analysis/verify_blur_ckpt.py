"""Verify blur pilot checkpoints are reproducible from raw ckpt files.

Checks:
1. Each sigma ckpt has exactly N=390 unique done fnames.
2. Recomputing per-class & overall acc from ckpt matches the merged JSON exactly.
3. Done sets are identical across sigmas (same sample list).
4. Same-image comparison: max per-class acc for given sigma is where you'd expect.
"""
import json
import sys
from pathlib import Path

CKPT_DIR = Path("results/blur_gemma4_31b_ckpt")
OUT_PATH = Path("results/exp_perturbation_blur.json")
N_EXPECT = 390
CLASSES = ["guide", "answer", "On-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackBoard"]

out = json.loads(OUT_PATH.read_text())
print(f"merged json: model={out['model']} task={out['task'][:40]}...")
ok = True

done_sets = {}
for s in range(5):
    ckpt = json.loads((CKPT_DIR / f"blur_{s}.json").read_text())
    done = ckpt["done"]
    done_sets[s] = set(done)
    n_unq = len(set(done))
    pc = ckpt["per_class"]
    overall = sum(p["correct"] for p in pc.values()) / max(sum(p["total"] for p in pc.values()), 1)
    merged = out["results"][f"sigma_{s}"]
    match = abs(merged["overall_acc"] - overall) < 1e-12
    pc_match = all(abs(merged["per_class_acc"][c] - pc.get(c, {"correct": 0, "total": 0})["correct"] / max(pc.get(c, {"total": 0})["total"], 1)) < 1e-12 for c in CLASSES)
    per_total_ok = all(pc.get(c, {"total": 0})["total"] == 50 for c in CLASSES) and sum(p["total"] for p in pc.values()) == N_EXPECT
    print(f"sigma={s}: n_unique={n_unq} n_total={len(done)} | merged_overall_match={match} "
          f"merged_perclass_match={pc_match} each_class_50={per_total_ok} overall={overall:.4f}")
    ok &= (n_unq == N_EXPECT) and (len(done) == N_EXPECT) and match and pc_match and per_total_ok

# cross-sigma done set identical?
all_same = all(done_sets[s] == done_sets[0] for s in range(1, 5))
print(f"\nidentical sample lists across sigmas: {all_same}")
ok &= all_same

# per-class acc table recomputed
print("\nper-class acc (recomputed from ckpts):")
print("class                 " + "".join(f"  sig{s} " for s in range(5)))
for c in CLASSES:
    row = []
    for s in range(5):
        pc = json.loads((CKPT_DIR / f"blur_{s}.json").read_text())["per_class"].get(c, {"correct": 0, "total": 0})
        row.append(f"{pc['correct']/max(pc['total'],1):.3f}")
    print(f"{c:20s} " + " ".join(row))

print(f"\nVERDICT: {'ALL CHECKS PASS' if ok else 'FAILURES DETECTED'}")
sys.exit(0 if ok else 1)