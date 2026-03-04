#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

LIMIT="${1:-200}"

export BIRD_DB_ROOTS=data/dev_databases

echo "=== OP1: Multi-Turn Agentic Text-to-SQL ==="
echo "Limit: $LIMIT entries"
echo ""

# Run inference
python op1_prompt_and_tools/run_inference.py \
    --limit "$LIMIT" \
    --save_traces \
    --output_path results/op1_predict_dev.json \
    --trace_dir results/op1_traces

echo ""
echo "=== Running Evaluation ==="

# Run evaluation using existing evaluation script
python scripts/run_evaluation.py \
    --predicted_sql_path results/op1_predict_dev.json \
    --output_path results/op1_eval_report.txt
