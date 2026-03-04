#!/bin/bash
# Qwen3.5-35B-A3B-FP8 推理服务启动脚本
# 使用 SGLang 框架，OpenAI 兼容 API

set -e

MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
MEM_FRACTION=${MEM_FRACTION:-0.92}

echo "=== 启动 Qwen3.5-35B-A3B-FP8 推理服务 ==="
echo "模型: ${MODEL}"
echo "端口: ${PORT}"
echo "Context: ${CONTEXT_LENGTH}"
echo "显存占用比例: ${MEM_FRACTION}"

python3 -m sglang.launch_server \
  --model ${MODEL} \
  --port ${PORT} \
  --host ${HOST} \
  --context-length ${CONTEXT_LENGTH} \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --mem-fraction-static ${MEM_FRACTION}
