# BirdSQL

基于大语言模型的 Text-to-SQL 项目，使用 Qwen3.5-35B-A3B-FP8 作为推理引擎。

## 项目结构

```
birdsql/
├── README.md
├── qwen3.5-35b-a3b-fp8-dep/       # 模型部署目录
│   ├── DEPLOY.md                   # 详细部署文档
│   ├── install.sh                  # 环境安装脚本
│   ├── serve.sh                    # 服务启动脚本
│   └── test.sh                     # 推理测试脚本
└── qwen3.5_sql_test.txt            # 推理测试结果样例
```

## 模型

| 项目 | 详情 |
|------|------|
| 模型 | [Qwen/Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8) |
| 架构 | MoE（总参数 35B，激活参数 3B） |
| 量化 | FP8 |
| 推理框架 | SGLang 0.5.9 |
| API | OpenAI 兼容（`/v1/chat/completions`） |

## 快速开始

### 1. 环境要求

- NVIDIA GPU，40GB+ VRAM（测试环境：L40S 46GB）
- CUDA 12.0+
- Python 3.10+

### 2. 安装

```bash
cd qwen3.5-35b-a3b-fp8-dep
./install.sh
```

### 3. 启动服务

```bash
./serve.sh
```

### 4. 测试

```bash
./test.sh
```

或直接调用 API：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B-FP8",
    "messages": [{"role": "user", "content": "将以下自然语言转换为SQL：查询销售额最高的前10个商品"}],
    "max_tokens": 2048
  }'
```

## 部署说明

详见 [qwen3.5-35b-a3b-fp8-dep/DEPLOY.md](qwen3.5-35b-a3b-fp8-dep/DEPLOY.md)，包含：

- 硬件要求与推荐 AWS 机型
- Context 长度与显存对照表
- 多卡部署方案（262K 长上下文）
- 踩坑记录（vLLM 兼容性、CuDNN 修复等）
