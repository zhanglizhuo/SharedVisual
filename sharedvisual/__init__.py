"""SharedVisual: shared utilities for the SharedVisual research project.

This package centralizes paths, dataset definitions, and reusable helpers
so that every script resolves configuration the same way and external users
can run the code without editing hardcoded paths.

Usage in scripts:
    from sharedvisual.paths import REPO_ROOT, RESULTS_DIR, DATA_DIR, SEED_BOOTSTRAP
    from sharedvisual.datasets import CLASS_MAP, SUBDATASETS, load_scb5_metadata
"""
