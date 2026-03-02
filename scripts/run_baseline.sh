#!/bin/bash
# End-to-end BIRD baseline: download data -> inference -> evaluation
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_DIR}"

echo "========================================"
echo "  BIRD Baseline: Qwen3.5-35B-A3B-FP8"
echo "========================================"

# Step 1: Check data
if [ ! -f "data/dev.json" ]; then
    echo "=== Step 1: Download data ==="
    bash scripts/download_data.sh
else
    echo "=== Step 1: Data already exists ==="
fi

# Step 2: Install dependencies
echo "=== Step 2: Install dependencies ==="
pip install -q func_timeout openai tqdm

# Step 3: Check model service
echo "=== Step 3: Check model service ==="
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "ERROR: Model service not running on port 8000"
    echo "Start it with: cd qwen3.5-35b-a3b-fp8-dep && ./serve.sh"
    exit 1
fi
echo "Model service is running"

# Step 4: Run inference
echo "=== Step 4: Run inference ==="
python scripts/run_inference.py \
    --dev_json_path data/dev.json \
    --db_root_path data/dev_databases \
    --output_path results/predict_dev.json \
    --model "Qwen/Qwen3.5-35B-A3B-FP8" \
    --api_base "http://localhost:8000/v1" \
    --max_tokens 4096 \
    --temperature 0 \
    --concurrency 8

# Step 5: Run evaluation
echo "=== Step 5: Run evaluation ==="
python scripts/run_evaluation.py \
    --predicted_sql_path results/predict_dev.json \
    --dev_json_path data/dev.json \
    --db_root_path data/dev_databases \
    --num_cpus 8 \
    --timeout 30

echo ""
echo "========================================"
echo "  Baseline complete!"
echo "========================================"
