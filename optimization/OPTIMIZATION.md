# CHASE-SQL 优化实现文档

## 1. 概述

基于论文 *"CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL"* (BIRD SOTA 73.01% EX) 实现多路径生成 + 自一致性投票的优化方案。

### 验证结果（dev 前 100 条，同数据对比）

| 难度 | Baseline | CHASE-SQL | Delta |
|------|----------|-----------|-------|
| simple | 66.10% (39/59) | 67.80% (40/59) | **+1.70%** |
| moderate | 34.29% (12/35) | 37.14% (13/35) | **+2.85%** |
| challenging | 0.00% (0/6) | 0.00% (0/6) | 0.00% |
| **Overall** | **51.00% (51/100)** | **53.00% (53/100)** | **+2.00%** |

净变化：5 条改对，3 条改错，净 +2 条。

### 运行配置

```
模型: Qwen3.5-35B-A3B-FP8 (SGLang, reasoning mode)
生成器: 3 个 (Divide&Conquer / QueryPlan / SyntheticExamples)
每生成器候选数: 3 (论文建议 7，为节省时间减半)
Query Fixer: 开启，最多 2 轮修复
选择策略: self-consistency (执行结果多数投票)
Value Retrieval: 开启
耗时: ~2 小时 (79s/entry, concurrency=2)
```

---

## 2. 架构

```
Question
  │
  ├─ Value Retrieval (LLM 关键词提取 → 模糊匹配 DB 值)
  │
  ├─ Generator 1: Divide & Conquer CoT ──→ 3 candidates
  ├─ Generator 2: Query Plan CoT       ──→ 3 candidates    (并发执行)
  ├─ Generator 3: Synthetic Examples    ──→ 3 candidates
  │
  ├─ 过滤 SELECT 1 / 解析失败
  │
  ├─ Query Fixer (执行 SQL → 报错 → LLM 修复 → 重试，最多 2 轮)
  │
  ├─ Self-Consistency 投票 (按执行结果分组，选最大组)
  │
  └─ Final SQL
      └─ 若全部失败则 fallback 到 baseline prompt (temp=0)
```

---

## 3. 文件清单

```
optimization/
├── src/
│   ├── value_retrieval.py        # 180 行 - 关键词提取 + 编辑距离匹配
│   ├── generators/
│   │   ├── divide_conquer.py     #  75 行 - 分解子问题 CoT
│   │   ├── query_plan.py         #  74 行 - 模拟执行计划 CoT
│   │   └── synthetic_examples.py # 159 行 - 动态生成 few-shot 示例
│   ├── query_fixer.py            # 147 行 - SQL 自修复循环
│   ├── selector.py               # 342 行 - 一致性投票 / LLM pairwise / 训练模型
│   └── pipeline.py               # 277 行 - 端到端编排
├── prompts/
│   ├── divide_conquer.txt        # Divide & Conquer 提示模板
│   ├── query_plan.txt            # Query Plan 提示模板
│   ├── synthetic_gen.txt         # Synthetic Examples 提示模板
│   └── query_fixer.txt           # Query Fixer 提示模板
├── scripts/
│   ├── run_optimization.py       # 169 行 - 推理入口，支持 resume / feature flags
│   └── run_eval.py               # 110 行 - 评测入口，支持 baseline 对比
├── training/
│   ├── generate_training_data.py # 233 行 - 生成 pairwise 训练数据
│   ├── train_selector.py         # 150 行 - LoRA 微调 Qwen2.5-1.5B
│   └── ray_train_job.yaml        #  66 行 - EKS Ray Job 配置
└── results/
    ├── predict_optimized_100.json
    ├── eval_report_100.json
    └── error_analysis_optimized.json
```

---

## 4. 各模块说明

### 4.1 Value Retrieval (`value_retrieval.py`)

**作用**：从问题中提取可能出现在 WHERE 条件中的具体值，然后在数据库列值中模糊匹配。

**流程**：
1. LLM 提取关键词（`extract_keywords`）：给模型问题和 evidence，让它列出可能的过滤值
2. 获取数据库所有 TEXT 列的 distinct 值（`get_column_values`，按 db_id 缓存）
3. 对每个关键词在所有列值中模糊匹配（`fuzzy_match`）：
   - 精确匹配 → score=1.0
   - 包含匹配 → score≥0.8
   - 编辑距离 SequenceMatcher → threshold=0.6
4. 返回 top-k 匹配，格式化为 `### Matched Database Values:` 注入 prompt

**已知问题**：在本次验证中 value_hint 经常返回空（关键词提取在 reasoning mode 下效率较低），实际帮助有限。

### 4.2 Divide & Conquer CoT (`generators/divide_conquer.py`)

**作用**：将复杂问题拆分为子问题 → 分别写子 SQL → 组装 → 优化。

**Prompt 策略**：
- 引导模型做 5 步推理：理解问题 → 分解 → 求解子问题 → 组装 → 验证
- 第一个候选用原始 schema 顺序，后续候选随机 shuffle 表顺序增加多样性
- temperature=0.8

**并发**：所有候选通过 `asyncio.gather` 并发生成（之前是串行，已修复）。

### 4.3 Query Plan CoT (`generators/query_plan.py`)

**作用**：模拟数据库执行计划的推理过程。

**Prompt 策略**：
- 7 步推理：确定表 → 访问路径 → JOIN 计划 → 过滤条件 → 聚合 → 输出格式 → 写 SQL
- 同样支持 schema shuffle + temperature sampling

### 4.4 Online Synthetic Examples (`generators/synthetic_examples.py`)

**作用**：为每个测试问题动态生成相关的 few-shot 示例。

**流程**：
1. `_detect_sql_patterns`：从问题文本检测可能的 SQL 模式（GROUP BY、JOIN、ORDER BY+LIMIT 等）
2. `generate_examples`：让 LLM 基于 schema + 检测到的模式生成 3 个 Q-SQL 示例对
3. 将示例注入 prompt，再生成实际 SQL 候选

**特点**：这是最慢的生成器（需要先生成示例再生成候选），但理论上对复杂查询帮助最大。

### 4.5 Query Fixer (`query_fixer.py`)

**作用**：执行 SQL → 发现错误 → 反馈给 LLM 修复 → 循环最多 β 次。

**修复触发条件**：
- SQL 语法错误（`execute_sql_check` 返回 error message）
- 查询超时（简化提示）
- 首轮空结果集（提示可能的逻辑错误：错误列名、过严 WHERE、缺失 JOIN 等）

**注意**：空结果修复只在第一轮触发，避免死循环。所有候选的修复通过 `asyncio.gather` 并发执行。

### 4.6 Selector (`selector.py`)

实现了 3 种选择策略：

**Self-Consistency（当前使用）**：
- 执行所有候选 SQL，按结果集分组（`set(result)` 去重后比较）
- 选最大组的第一个候选
- 过滤 ERROR/TIMEOUT 结果
- 变体 `consistency_empty_penalty`：对空结果集降权

**LLM Pairwise（备选）**：
- 对候选两两比较，让 LLM 判断哪个更可能正确
- 最多 15 次比较（候选多时随机采样），统计胜场选胜者
- API 开销大，适合候选数少的场景

**Trained Selector（待实现）**：
- LoRA 微调 Qwen2.5-1.5B-Instruct 做二分类
- 通过 `TrainedSelector` 类加载，支持 pairwise tournament
- 需要先在 EKS 上训练

### 4.7 Pipeline (`pipeline.py`)

**编排逻辑**：
1. Value Retrieval（可选）
2. 3 个生成器并发运行 → 收集候选 → `parse_sql` 提取 SQL
3. 过滤 `SELECT 1`（parse 失败的兜底值）
4. Query Fixer 并发修复所有候选
5. 再次过滤 `SELECT 1`
6. Self-Consistency 投票
7. 若全部候选失败 → `_baseline_fallback`：用原始 baseline prompt (temp=0) 兜底

**关键修复**：
- 初版无 SELECT 1 过滤，导致 parse 失败的候选参与投票，大量 entry 被投成 SELECT 1
- 修复后仅 2/100 为 SELECT 1（均为 baseline 也无法生成有效 SQL 的 entry）

---

## 5. 验证过程中发现的问题

### 5.1 Reasoning Mode 的影响

SGLang 开启了 `--reasoning-parser qwen3`，模型输出分为 `reasoning_content`（思考）和 `content`（回答）。CoT prompt 导致模型在 content 中也输出大量推理文本而非纯 SQL，需要 `parse_sql` 从中提取。

影响：
- 每个 API 调用耗时长（reasoning 占据大量 token），~6-15s/call
- 部分请求 content 为空（所有 token 被 reasoning 消耗），导致 parse 失败
- 总 API 调用量：每 entry ~15-25 calls（value_retrieval + synthetic_gen + 9 candidates + fixer）

### 5.2 SELECT 1 泛滥（已修复）

**初版问题**：`parse_sql` 对无法解析的内容返回 `SELECT 1`。当多个候选 parse 失败时，它们都变成 `SELECT 1`，在 self-consistency 投票中形成最大组，导致最终选择 `SELECT 1`。

初版 100 条中 36% 为 SELECT 1。

**修复方案**：
1. 在投票前过滤掉所有 `SELECT 1` 候选
2. 若过滤后无有效候选，fallback 到 baseline prompt
3. 修复后 SELECT 1 降至 2%

### 5.3 吞吐量瓶颈

| 配置 | 耗时/entry | 100 条总耗时 |
|------|-----------|-------------|
| 1 gen x 2 cands, no fixer | ~28s | ~47min |
| 3 gen x 7 cands, no fixer | ~133s | ~3.7h |
| 3 gen x 3 cands + fixer (验证配置) | ~79s | ~2.1h |
| Baseline (temp=0, 1 call) | ~1.8s | ~3min |

单 L40S GPU 上，CHASE-SQL 方案比 baseline 慢 ~44 倍。主要原因：
- 每 entry 15-25 个 API 调用（vs baseline 的 1 个）
- Reasoning mode 下每个调用 6-15 秒
- 生成器内部已改为并发，但 GPU 是瓶颈

### 5.4 回归分析

3 条回归（baseline 对，优化错）均非 SELECT 1，而是 self-consistency 投了错误结果：

| Entry | 类型 | 原因 |
|-------|------|------|
| #43 | moderate | 多个候选都错误地加了 GROUP BY AVG，投票选了错误多数 |
| #67 | simple | 候选对列名理解不同，多数选了错误列 |
| #80 | simple | JOIN 条件不同，多数候选用了错误的 JOIN |

**结论**：self-consistency 的核心假设是"正确答案更容易被多次独立生成"，但当模型对某个问题有系统性偏差时，多路径反而强化错误。

---

## 6. 与论文差距分析

| 论文配置 | 本次验证 | 差异 |
|----------|---------|------|
| GPT-4o / Claude 3.5 Sonnet | Qwen3.5-35B-A3B (3B active) | 模型能力差距显著 |
| 每生成器 7 候选 (21 total) | 3 候选 (9 total) | 候选多样性不足 |
| 训练过的 Selection Agent | Self-Consistency 投票 | 无训练选择器 |
| 无 reasoning mode overhead | SGLang reasoning parser | 额外延迟 + parse 复杂度 |
| 全量 dev 集 (1534 条) | 前 100 条 | 样本量小 |

论文中 self-consistency 本身就有 ~69% (vs baseline ~63%)，而训练选择器额外 +4% 到 73%。本次 +2% 的提升幅度偏低，核心原因是模型能力和候选数量。

---

## 7. 待改进方向

### 高优先级

1. **关闭 reasoning mode 或增大 max_tokens**
   - 当前 reasoning 消耗大量 token budget，导致 content 质量下降
   - 考虑对 CoT prompt 单独关闭 reasoning（`/no_think`），让模型在 content 中做 CoT
   - 或将 max_tokens 增大到 8192

2. **增加候选数到 7**
   - 论文用 7 候选/生成器，多样性更充分
   - 需要更高并发或更长等待时间

3. **改进 SQL 解析**
   - 当前 `parse_sql` 对 CoT 输出的 SQL 提取不够鲁棒
   - 可以添加更多 pattern：`**Final SQL:**` / `**Optimized Query:**` 等标记

### 中优先级

4. **训练选择模型**
   - 用 `generate_training_data.py` 在 dev 集生成 pairwise 训练数据
   - 在 EKS Ray 集群 LoRA 微调 Qwen2.5-1.5B
   - 预期在 selection 阶段额外 +3-4%

5. **优化 Value Retrieval**
   - 当前关键词提取在 reasoning mode 下效果差
   - 可以改用规则提取（NER / 正则）替代 LLM 调用

### 低优先级

6. **Prompt 迭代**
   - 对 3 个生成器的 prompt 做 few-shot 优化
   - 根据错误分析调整指令（如强调 column name 精确匹配）

7. **成本优化**
   - Query Fixer 对已成功执行的 SQL 跳过修复
   - Synthetic Examples 生成器的"先生成示例"步骤可以缓存

---

## 8. 运行命令

```bash
# 完整运行（3 gen x 3 cands + fixer）
python optimization/scripts/run_optimization.py \
  --limit 100 --n_candidates 3 --concurrency 2

# 单生成器快速测试
python optimization/scripts/run_optimization.py \
  --limit 10 --n_candidates 3 --no_qp --no_se --no_fixer

# 评测并对比 baseline
python optimization/scripts/run_eval.py \
  --predicted_sql_path optimization/results/predict_optimized_100.json \
  --baseline_path results/eval_report.json

# 生成训练数据（用于训练 selector）
python optimization/training/generate_training_data.py --limit 100 --n_candidates 3

# EKS 训练（需先 port-forward）
ray job submit --address http://localhost:8265 \
  --working-dir optimization/training \
  -- python train_selector.py
```
