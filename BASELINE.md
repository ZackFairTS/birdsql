# BIRD Benchmark Baseline 文档

## 概述

使用 Qwen3.5-35B-A3B-FP8 模型在 BIRD (BIg Bench for LaRge-scale Database Grounded Text-to-SQL Evaluation) dev 集上完成 baseline 评测。

## Baseline 结果

| 难度 | 正确 | 总数 | EX (%) |
|------|------|------|--------|
| simple | 633 | 925 | 68.43% |
| moderate | 233 | 464 | 50.22% |
| challenging | 75 | 145 | 51.72% |
| **ALL** | **941** | **1534** | **61.34%** |

## 评测指标：Execution Accuracy (EX)

EX 是 BIRD 的主要评测指标，核心逻辑如下：

1. 将模型生成的 SQL 和标准答案 SQL **分别在对应 SQLite 数据库上执行**
2. 比较两个执行结果集是否相同（`set(predicted_res) == set(gold_res)`）
3. 结果集无序比较——行的顺序不影响判定
4. 每条 SQL 执行超时 30 秒，超时判错
5. SQL 执行出错（语法错误等）判错

这种评测方式的优势：同一个问题可能有多种正确 SQL 写法，只要最终查询结果一致就算正确。

## 推理配置

| 配置项 | 值 |
|--------|-----|
| 模型 | Qwen/Qwen3.5-35B-A3B-FP8 |
| 推理框架 | SGLang 0.5.9 |
| GPU | NVIDIA L40S 46GB |
| Context Length | 16384 |
| Temperature | 0（确定性输出） |
| Max Tokens | 4096 |
| 并发数 | 8 |

## Prompt 设计

### System Prompt

```
You are an expert SQL assistant. Given a database schema and a question, generate a valid SQLite query that answers the question.

Rules:
- Output ONLY the SQL query, nothing else
- Do not include markdown code fences or backticks
- Do not include any explanations or comments
- The query must be valid SQLite syntax
- Use the exact table and column names from the schema
```

### User Prompt 模板

```
### Database Schema:
{CREATE TABLE DDL + 每表 3 条 sample rows}

### External Knowledge:
{evidence，来自 BIRD 数据集的外部知识提示}

### Question:
{自然语言问题}

### SQL:
```

### Schema 策略

- 从 SQLite 的 `sqlite_master` 提取完整 `CREATE TABLE` DDL
- 每表附加 3 条 sample rows，帮助模型理解数据格式和值域
- 按 `db_id` 缓存 schema（dev 集仅 11 个数据库）

## 数据集

BIRD dev 集包含：

- **1534** 条 text-to-SQL 数据
- **11** 个真实世界数据库
- **3** 个难度级别：simple (925), moderate (464), challenging (145)
- 每条数据包含：`question`（自然语言问题）、`SQL`（标准答案）、`db_id`（数据库）、`evidence`（外部知识）、`difficulty`（难度）

数据库涵盖领域：
california_schools, card_games, codebase_community, debit_card_specializing, european_football_2, financial, formula_1, student_club, superhero, thrombosis_prediction, toxicology

## Pipeline 架构

```
dev.json ──> schema_extractor ──> prompt_builder ──> Qwen3.5 API ──> sql_parser ──> predict_dev.json
                                                                                         │
                                                                                         v
dev.json (gold SQL) + dev_databases/ ──────────────────────────────────────────> evaluation ──> EX score
```

### 模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| Schema 提取 | `src/schema_extractor.py` | 从 SQLite 提取 DDL 和 sample rows |
| Prompt 构建 | `src/prompt_builder.py` | 组装 system/user 消息 |
| SQL 解析 | `src/sql_parser.py` | 从模型输出提取 SQL（处理 markdown/解释文本） |
| 评测 | `src/evaluation.py` | EX 指标计算，多进程并行 |
| 推理脚本 | `scripts/run_inference.py` | 异步并发推理，支持断点续传 |
| 评测脚本 | `scripts/run_evaluation.py` | 评测入口，生成报告 |

### 关键设计

- **异步并发**：使用 `asyncio` + `openai.AsyncOpenAI` + `Semaphore(8)` 控制并发
- **断点续传**：每 50 条保存中间结果，中断后可 `--resume` 继续
- **Thinking Mode**：SGLang 配置 `--reasoning-parser qwen3`，模型先输出 reasoning_content 再输出 content，SQL 从 content 字段提取
- **评测并行**：`multiprocessing.Pool(8)` 并行执行 SQL 并比较

## 运行方式

### 前置条件

- 模型服务已启动（`cd qwen3.5-35b-a3b-fp8-dep && ./serve.sh`）
- Python 环境已安装依赖（`pip install -r requirements.txt`）

### 一键运行

```bash
bash scripts/run_baseline.sh
```

### 分步运行

```bash
# 1. 下载数据
bash scripts/download_data.sh

# 2. 运行推理（约 45 分钟）
python scripts/run_inference.py \
    --dev_json_path data/dev.json \
    --db_root_path data/dev_databases \
    --output_path results/predict_dev.json \
    --concurrency 8

# 3. 运行评测
python scripts/run_evaluation.py \
    --predicted_sql_path results/predict_dev.json \
    --dev_json_path data/dev.json \
    --db_root_path data/dev_databases
```

### 快速验证（少量数据）

```bash
python scripts/run_inference.py --limit 10 --output_path results/predict_test.json --no-resume
python scripts/run_evaluation.py --predicted_sql_path results/predict_test.json
```

## 可能的改进方向

1. **Few-shot prompting**：在 prompt 中添加同数据库的示例 question-SQL 对
2. **Schema linking**：只提供与问题相关的表和列，减少 prompt 长度
3. **Self-consistency**：多次采样取多数结果
4. **Database description**：使用 BIRD 提供的 CSV 列描述信息
5. **更长 context**：多卡部署支持 262K context，容纳更大 schema
6. **Fine-tuning**：使用 BIRD train 集对模型进行微调
