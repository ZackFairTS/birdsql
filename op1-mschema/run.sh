#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

LIMIT="${1:-200}"

echo "=== MSchema: Baseline + M-Schema ==="
echo "Limit: $LIMIT entries"
echo ""

python op1-mschema/run_inference.py \
    --limit "$LIMIT" \
    --output_path op1-mschema/results/predict_dev.json

echo ""
echo "=== Running Evaluation ==="

python scripts/run_evaluation.py \
    --predicted_sql_path op1-mschema/results/predict_dev.json \
    --output_path op1-mschema/results/eval_report.txt
