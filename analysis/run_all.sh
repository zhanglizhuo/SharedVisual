#!/usr/bin/env bash
# Run all SharedVisual analyses that require no GPU.
# GPU experiments (feature extraction, prompt evaluation) run via
# experiments/exp*.py on a GPU server; their outputs are committed in results/.
set -e

cd "$(dirname "$0")/.."

echo "=== 1. Permutation significance thresholds (n=8/13/40) ==="
python analysis/permutation_critical_rho.py

echo ""
echo "=== 2. Exp 1C: Natural MLLM Groups by Encoder Family ==="
python analysis/exp1_group_analysis.py

echo ""
echo "=== 3. Exp 1C deep analysis (LOO partial correlation, SCB5) ==="
python analysis/exp1_deep_analysis.py

echo ""
echo "=== 4. Exact Mantel tests (SCB5 8x8, Stanford40 40x40) ==="
python analysis/mantel_test.py

echo ""
echo "=== 5. SCB5 mediation bootstrap (leave-one-out covariate) ==="
python analysis/compute_partial_ci.py

echo ""
echo "=== 6. Stanford40 mediation + aggregated phenomenon ==="
python analysis/stanford40_partial_correlation.py

echo ""
echo "Done. All no-GPU analyses regenerated."
