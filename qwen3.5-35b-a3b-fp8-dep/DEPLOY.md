# Qwen3.5-35B-A3B-FP8 部署文档

## 模型信息

| 项目 | 详情 |
|------|------|
| 模型 | Qwen/Qwen3.5-35B-A3B-FP8 |
| 架构 | MoE (Mixture of Experts) |
| 总参数 | 35B |
| 激活参数 | 3B |
| 量化精度 | FP8 |
| 权重大小 | ~35 GB |

## 硬件要求

| 项目 | 最低要求 | 当前环境 |
|------|----------|----------|
| GPU | 单卡 40GB+ VRAM | NVIDIA L40S 46GB |
| 系统内存 | 32GB+ | 61GB |
| 磁盘 | 50GB+ 可用 | 401GB 可用 |
| CUDA | 12.0+ | 13.0 |

## 推理框架选择

**使用 SGLang 0.5.9**（而非 vLLM）

原因：vLLM 0.16.0 不支持 `Qwen3_5MoeForConditionalGeneration` 架构，且 transformers 5.x 与 vLLM 0.16.0 存在依赖冲突。SGLang 0.5.9 原生支持该模型。

## 部署步骤

### 1. 安装环境

```bash
chmod +x install.sh serve.sh test.sh
./install.sh
```

主要依赖：
- sglang[all] >= 0.5.9
- nvidia-cudnn-cu12 == 9.16.0.29（修复 PyTorch 2.9.1 + CuDNN < 9.15 的兼容性问题）

### 2. 启动服务

```bash
./serve.sh
```

支持环境变量覆盖默认参数：

```bash
PORT=8080 CONTEXT_LENGTH=8192 ./serve.sh
```

### 3. 测试验证

```bash
./test.sh
```

## API 使用

服务兼容 OpenAI API 格式：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B-FP8",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 2048
  }'
```

响应中 `reasoning_content` 字段包含思维链内容（由 `--reasoning-parser qwen3` 启用）。

## Context 长度说明

| Context | 显存需求 | 单卡 L40S (46GB) |
|---------|----------|-------------------|
| 8192 | ~37 GB | 可用，KV cache 充裕 |
| 16384 | ~39 GB | 可用，推荐配置 |
| 32768 | ~43 GB | 勉强可用 |
| 262144 | ~90 GB+ | 不可用，需多卡 |

模型权重固定占用 ~35GB，剩余显存分配给 KV cache。单卡 L40S 推荐 context-length 设为 16384。

## 多卡部署（262K Context）

如需 262K 长上下文，需要多卡实例：

```bash
python3 -m sglang.launch_server \
  --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --port 8000 \
  --host 0.0.0.0 \
  --context-length 262144 \
  --tp-size 4 \
  --trust-remote-code \
  --reasoning-parser qwen3
```

推荐 AWS 实例：g6e.12xlarge (4x L40S 192GB) 或 p5.48xlarge (8x H100)。

## 踩坑记录

1. **vLLM 0.16.0 不支持 Qwen3.5**：`Qwen3_5MoeForConditionalGeneration` 不在支持列表中，`--model-impl transformers` 通用后端也不兼容
2. **transformers 版本冲突**：Qwen3.5 需要 transformers >= 5.2.0，但 vLLM 0.16.0 要求 < 5.0
3. **CuDNN 兼容性**：PyTorch 2.9.1 + CuDNN < 9.15 存在已知 bug，SGLang 启动时会检查并报错，需升级到 9.16+
4. **262K Context OOM**：单卡 46GB 下，模型权重 35GB + 262K KV cache 远超显存，CUDA graph capture 阶段 `capture_bs=[0]` 断言失败
5. **端口冲突**：如有 kubectl port-forward 等占用 8000 端口，需先清理
