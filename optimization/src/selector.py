"""Candidate Selection: self-consistency voting and trained model selection.

Based on CHASE-SQL Section 3.5.
Phase 1: Self-consistency - execute all candidates, group by result, pick majority.
Phase 2: Trained selector - pairwise comparison using fine-tuned model.
"""

import sqlite3
from collections import Counter
from pathlib import Path

from func_timeout import FunctionTimedOut, func_timeout


def execute_sql_safe(db_path: str, sql: str, timeout: int = 30) -> tuple[str, list | None]:
    """Execute SQL and return a hashable result key + raw results.

    Returns (result_key, results) where result_key is a string
    representation of the result set for grouping.
    """
    def _exec():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return result

    try:
        result = func_timeout(timeout, _exec)
        # Create a hashable key from the result set (order-independent)
        result_key = str(sorted(set(str(r) for r in result))) if result else "EMPTY"
        return result_key, result
    except FunctionTimedOut:
        return "TIMEOUT", None
    except Exception as e:
        return f"ERROR:{e}", None


def select_by_consistency(
    candidates: list[str],
    db_path: str,
    timeout: int = 30,
) -> tuple[str, dict]:
    """Self-consistency selection: execute all candidates, pick majority result.

    Returns (selected_sql, stats) where stats contains grouping info.
    """
    # Execute all candidates and group by result
    result_groups: dict[str, list[int]] = {}
    results_map: dict[int, tuple[str, list | None]] = {}

    for i, sql in enumerate(candidates):
        result_key, result = execute_sql_safe(db_path, sql, timeout)
        results_map[i] = (result_key, result)

        if result_key not in result_groups:
            result_groups[result_key] = []
        result_groups[result_key].append(i)

    # Filter out error/timeout groups
    valid_groups = {
        k: v for k, v in result_groups.items()
        if not k.startswith("ERROR:") and k != "TIMEOUT"
    }

    if not valid_groups:
        # All candidates failed - return the first one as fallback
        return candidates[0], {
            "method": "fallback",
            "total_candidates": len(candidates),
            "valid_groups": 0,
            "all_errors": True,
        }

    # Find the largest group
    largest_group_key = max(valid_groups, key=lambda k: len(valid_groups[k]))
    largest_group = valid_groups[largest_group_key]

    # Pick the first candidate from the largest group (deterministic)
    selected_idx = largest_group[0]

    stats = {
        "method": "self_consistency",
        "total_candidates": len(candidates),
        "valid_groups": len(valid_groups),
        "largest_group_size": len(largest_group),
        "largest_group_indices": largest_group,
        "selected_idx": selected_idx,
        "group_sizes": {k: len(v) for k, v in sorted(valid_groups.items(), key=lambda x: -len(x[1]))},
        "error_count": sum(1 for k in result_groups if k.startswith("ERROR:") or k == "TIMEOUT"),
    }

    return candidates[selected_idx], stats


def select_by_consistency_with_empty_penalty(
    candidates: list[str],
    db_path: str,
    timeout: int = 30,
) -> tuple[str, dict]:
    """Enhanced self-consistency that penalizes empty results.

    Empty results are less likely to be correct, so we prefer
    non-empty result groups when counts are close.
    """
    result_groups: dict[str, list[int]] = {}
    result_data: dict[str, list | None] = {}

    for i, sql in enumerate(candidates):
        result_key, result = execute_sql_safe(db_path, sql, timeout)
        if result_key not in result_groups:
            result_groups[result_key] = []
            result_data[result_key] = result
        result_groups[result_key].append(i)

    valid_groups = {
        k: v for k, v in result_groups.items()
        if not k.startswith("ERROR:") and k != "TIMEOUT"
    }

    if not valid_groups:
        return candidates[0], {"method": "fallback", "all_errors": True}

    # Score groups: size, with penalty for EMPTY
    def group_score(key: str) -> tuple[int, int]:
        size = len(valid_groups[key])
        is_nonempty = 0 if key == "EMPTY" else 1
        return (size, is_nonempty)

    best_key = max(valid_groups, key=group_score)
    selected_idx = valid_groups[best_key][0]

    stats = {
        "method": "consistency_with_empty_penalty",
        "total_candidates": len(candidates),
        "valid_groups": len(valid_groups),
        "largest_group_size": len(valid_groups[best_key]),
        "selected_idx": selected_idx,
    }

    return candidates[selected_idx], stats


PAIRWISE_PROMPT = """Given the following question and two SQL candidates, determine which SQL query is more likely to be correct.

### Database Schema:
{schema}

### Question:
{question}

### External Knowledge:
{evidence}

### SQL Candidate A:
{sql_a}

### SQL Candidate B:
{sql_b}

Which candidate is more likely correct? Answer with ONLY "A" or "B"."""


async def select_by_pairwise_llm(
    client,
    model: str,
    candidates: list[str],
    schema: str,
    question: str,
    evidence: str = "",
    db_path: str = "",
) -> tuple[str, dict]:
    """LLM-based pairwise selection (fallback before trained model is ready).

    Runs a tournament: compare pairs, track wins, select highest-win candidate.
    """
    import asyncio
    import random

    if len(candidates) <= 1:
        return candidates[0] if candidates else "SELECT 1", {"method": "single_candidate"}

    # Deduplicate candidates by SQL text
    unique = list(dict.fromkeys(candidates))
    if len(unique) == 1:
        return unique[0], {"method": "all_identical"}

    # Run pairwise comparisons (sample if too many)
    wins = Counter()
    pairs = []
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            pairs.append((i, j))

    # Limit comparisons for efficiency
    if len(pairs) > 15:
        random.shuffle(pairs)
        pairs = pairs[:15]

    async def compare_pair(idx_a: int, idx_b: int) -> int | None:
        prompt = PAIRWISE_PROMPT.format(
            schema=schema,
            question=question,
            evidence=evidence,
            sql_a=unique[idx_a],
            sql_b=unique[idx_b],
        )
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
                temperature=0,
            )
            answer = (response.choices[0].message.content or "").strip().upper()
            if "A" in answer:
                return idx_a
            elif "B" in answer:
                return idx_b
        except Exception:
            pass
        return None

    tasks = [compare_pair(i, j) for i, j in pairs]
    results = await asyncio.gather(*tasks)

    for winner in results:
        if winner is not None:
            wins[winner] += 1

    if wins:
        best_idx = wins.most_common(1)[0][0]
    else:
        best_idx = 0

    stats = {
        "method": "pairwise_llm",
        "total_candidates": len(candidates),
        "unique_candidates": len(unique),
        "comparisons": len(pairs),
        "wins": dict(wins),
        "selected_idx": best_idx,
    }

    return unique[best_idx], stats


class TrainedSelector:
    """Selector using a fine-tuned classification model for pairwise comparison.

    Loaded after training on EKS. Falls back to self-consistency if model not available.
    """

    def __init__(self, model_path: str | None = None):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load the fine-tuned LoRA model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from peft import PeftModel

            base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name, num_labels=2
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.model.eval()
        except Exception as e:
            print(f"[WARN] Failed to load trained selector: {e}")
            self.model = None

    def is_available(self) -> bool:
        return self.model is not None

    def compare_pair(self, schema: str, question: str, evidence: str, sql_a: str, sql_b: str) -> str:
        """Compare two SQL candidates, return the better one."""
        if not self.is_available():
            raise RuntimeError("Trained selector model not loaded")

        import torch

        prompt = (
            f"Question: {question}\n"
            f"Evidence: {evidence}\n"
            f"Schema: {schema[:1000]}\n"  # Truncate schema for 1.5B model
            f"SQL A: {sql_a}\n"
            f"SQL B: {sql_b}\n"
            f"Which is correct?"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # label 0 = A is better, label 1 = B is better
            pred = torch.argmax(logits, dim=-1).item()

        return sql_a if pred == 0 else sql_b

    def select_best(
        self,
        candidates: list[str],
        schema: str,
        question: str,
        evidence: str = "",
    ) -> tuple[str, dict]:
        """Select the best candidate through pairwise tournament."""
        from collections import Counter

        if len(candidates) <= 1:
            return candidates[0] if candidates else "SELECT 1", {"method": "single"}

        unique = list(dict.fromkeys(candidates))
        if len(unique) == 1:
            return unique[0], {"method": "all_identical"}

        wins = Counter()
        comparisons = 0
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                winner = self.compare_pair(schema, question, evidence, unique[i], unique[j])
                if winner == unique[i]:
                    wins[i] += 1
                else:
                    wins[j] += 1
                comparisons += 1

        best_idx = wins.most_common(1)[0][0] if wins else 0

        return unique[best_idx], {
            "method": "trained_selector",
            "comparisons": comparisons,
            "wins": dict(wins),
            "selected_idx": best_idx,
        }
