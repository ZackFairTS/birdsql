#!/bin/bash
# Qwen3.5-35B-A3B-FP8 部署环境安装脚本
# 推理框架: SGLang (vLLM 0.16.0 不支持 Qwen3.5 架构)
# 硬件要求: NVIDIA L40S 46GB 或同级以上 GPU

set -e

echo "=== 安装 SGLang ==="
pip install "sglang[all]"

echo "=== 升级 CuDNN (PyTorch 2.9.1 兼容性修复) ==="
pip install nvidia-cudnn-cu12==9.16.0.29

echo "=== 预下载模型权重 ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-35B-A3B-FP8')
print('模型下载完成')
"

echo "=== 安装完成 ==="
echo "运行 ./serve.sh 启动服务"
