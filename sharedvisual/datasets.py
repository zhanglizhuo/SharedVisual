"""Dataset definitions and loaders shared across all scripts.

SCB5 (Smart Classroom Behavior v5) has 13 classes across 3 sub-datasets.
Stanford40 has 40 action classes.
"""
import json
from pathlib import Path

import numpy as np

from .paths import va_results

# SCB5 class structure: sub-dataset -> ordered class list.
# This is the single source of truth (was duplicated across 9 scripts).
CLASS_MAP = {
    "TeacherBehavior": [
        "guide", "answer", "On-stage interaction", "blackboard-writing",
        "teacher", "stand", "screen", "blackBoard",
    ],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}

# Flattened list of (sub-dataset, class) pairs in canonical order.
SCB5_CLASSES = [(ds, cls) for ds, classes in CLASS_MAP.items() for cls in classes]
N_SCB5_CLASSES = len(SCB5_CLASSES)  # = 13


# Sub-directory layout for SCB5 image/label directories (matches eval_hf.py).
SCB5_SUBDIRS = {
    "TeacherBehavior": "SCB5_TeacherBehavior/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2",
    "HandriseReadWrite": "SCB5_HandriseReadWrite",
    "BowTurnHead": "SCB_BowTurnHead",
}


def load_anchor_scores():
    """Load CLIP AnchorScore per-class accuracy on SCB5.

    Returns:
        dict mapping (sub_dataset, class) -> accuracy (float in [0, 1]).
    """
    va = va_results()
    a = json.load(open(va / "01_core" / "anchor_score_scb5" / "anchor_scores.json"))
    out = {}
    for ds, classes in CLASS_MAP.items():
        for cls in classes:
            out[(ds, cls)] = a[ds]["per_class_acc"][cls]["acc"]
    return out


def load_mllm_full():
    """Load the per-class accuracy of the 9 core MLLMs on SCB5.

    Returns:
        dict {sub_dataset: {model_name: {class_name: accuracy_float}}}.
        Accuracy values are floats (e.g. 0.04 meaning 4%).
    """
    va = va_results()
    return json.load(open(va / "01_core" / "paper_data" / "mllm_full.json"))


def load_scb5_model_accs():
    """Load SCB5 per-class accuracies for all 9 core MLLMs.

    Returns:
        (classes, anchor, model_accs) where
          classes: list of (sub_dataset, class) tuples (length 13),
          anchor: np.ndarray of AnchorScore per class,
          model_accs: list of (model_name, np.ndarray of per-class accuracy).
    """
    anchor_map = load_anchor_scores()
    mllm = load_mllm_full()
    classes = SCB5_CLASSES
    anchor = np.array([anchor_map[(ds, cls)] for ds, cls in classes])

    model_accs = []
    # Model list is the same across all sub-datasets; take from TeacherBehavior.
    for model_name in mllm["TeacherBehavior"]:
        vals = []
        ok = True
        for ds, cls in classes:
            v = mllm.get(ds, {}).get(model_name, {}).get(cls)
            if v is None:
                ok = False
                break
            vals.append(v)
        if ok:
            model_accs.append((model_name, np.array(vals, dtype=float)))
    return classes, anchor, model_accs


def load_stanford40():
    """Load Stanford40 per-class AnchorScores and MLLM accuracies.

    Returns:
        (classes, anchor, model_accs) where model_accs is
        {model_name: np.ndarray of per-class accuracy (40,)}.
    """
    va = va_results()
    a = json.load(open(va / "02_robustness" / "stanford40" / "anchor_scores.json"))
    o = json.load(open(va / "02_robustness" / "stanford40" / "ollama_results.json"))
    classes = sorted(a["per_class_acc"].keys())
    anchor = np.array([a["per_class_acc"][c]["acc"] for c in classes])
    model_accs = {
        m: np.array([o["models"][m]["per_class_acc"][c]["acc"] for c in classes],
                    dtype=float)
        for m in o["models"]
    }
    return classes, anchor, model_accs
