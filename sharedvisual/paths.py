"""Centralized path configuration for all SharedVisual scripts.

Every script imports from here so that:
  (1) there is a single source of truth for paths,
  (2) external users override locations via environment variables,
  (3) no script contains hardcoded usernames (/home/lizhuo, etc.).

Environment variables (all optional):
  SHAREDVISUAL_ROOT    Repository root (default: auto-detected from this file).
  SHAREDVISUAL_DATA    Dataset root (default: ~/data).
  SHAREDVISUAL_RESULTS Results dir (default: $ROOT/results).
"""
import os
from pathlib import Path

# Repository root: this file lives at <root>/sharedvisual/paths.py
REPO_ROOT = Path(__file__).resolve().parents[1]

# Allow explicit override (useful when running from a copied script elsewhere)
REPO_ROOT = Path(os.environ.get("SHAREDVISUAL_ROOT", str(REPO_ROOT)))

# Results directory (committed JSON outputs)
RESULTS_DIR = Path(os.environ.get("SHAREDVISUAL_RESULTS", str(REPO_ROOT / "results")))

# Dataset root: where SCB5 / Stanford40 / Flowers102 live after download.
# External users set SHAREDVISUAL_DATA; default is ~/data for convenience.
DATA_DIR = Path(os.environ.get("SHAREDVISUAL_DATA", str(Path.home() / "data")))

# VisualAnchor snapshot (committed upstream per-class accuracies).
# Lives inside the repo so analyses run without a sibling VisualAnchor checkout.
VA_SNAPSHOT = REPO_ROOT / "data" / "visualanchor_snapshot"

# Fallback to a sibling VisualAnchor checkout (development environment only).
VA_FALLBACK = REPO_ROOT.parent / "VisualAnchor" / "results"


def va_results():
    """Return the VisualAnchor results dir to read upstream data from.

    Prefers the committed snapshot (so a fresh clone works); falls back to a
    sibling VisualAnchor repository for developers who have it checked out.
    """
    if VA_SNAPSHOT.exists():
        return VA_SNAPSHOT
    if VA_FALLBACK.exists():
        return VA_FALLBACK
    raise FileNotFoundError(
        f"Neither {VA_SNAPSHOT} nor {VA_FALLBACK} exists. "
        "The data/visualanchor_snapshot directory should be committed."
    )


# Standard seeds used across the project (documented in README / paper).
SEED_BOOTSTRAP = 20260801   # mediation, cluster bootstrap, kruskal permutation
SEED_PERM = 42              # Mantel permutations, image sampling for MLLM eval
SEED_SIMULATION = 20260802  # mediation simulation appendix
