#!/bin/bash
# Qwen3.5-35B-A3B-FP8 推理测试脚本

PORT=${PORT:-8000}
BASE_URL="http://localhost:${PORT}/v1"

echo "=== 检查模型状态 ==="
curl -s ${BASE_URL}/models | python3 -m json.tool

echo ""
echo "=== 测试推理请求 ==="
curl -s ${BASE_URL}/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B-FP8",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己"}],
    "max_tokens": 512
  }' | python3 -m json.tool
