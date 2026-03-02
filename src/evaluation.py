"""BIRD benchmark evaluation - Execution Accuracy (EX) metric.

Based on the official BIRD evaluation logic:
- Execute predicted and gold SQL against SQLite databases
- Compare result sets (order-independent)
- 30-second timeout per query
"""

import json
import multiprocessing as mp
import sqlite3
from collections import defaultdict
from pathlib import Path

from func_timeout import FunctionTimedOut, func_timeout


def execute_sql(db_path: str, sql: str, timeout: int = 30) -> list:
    """Execute SQL against a SQLite database with timeout."""
    def _exec():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return result

    try:
        return func_timeout(timeout, _exec)
    except FunctionTimedOut:
        return ["TIMEOUT"]
    except Exception as e:
        return ["ERROR", str(e)]


def compare_results(predicted_res: list, gold_res: list) -> bool:
    """Compare two SQL result sets (order-independent)."""
    if not predicted_res or not gold_res:
        return predicted_res == gold_res
    # Check for error/timeout markers
    if isinstance(predicted_res[0], str) and predicted_res[0] in ("TIMEOUT", "ERROR"):
        return False
    if isinstance(gold_res[0], str) and gold_res[0] in ("TIMEOUT", "ERROR"):
        return False
    return set(predicted_res) == set(gold_res)


def evaluate_single(args: tuple) -> dict:
    """Evaluate a single prediction. Used with multiprocessing.Pool."""
    idx, predicted_sql, gold_sql, db_path, difficulty, timeout = args
    predicted_res = execute_sql(db_path, predicted_sql, timeout)
    gold_res = execute_sql(db_path, gold_sql, timeout)
    correct = compare_results(predicted_res, gold_res)
    return {
        "idx": idx,
        "correct": correct,
        "difficulty": difficulty,
        "predicted_sql": predicted_sql,
        "gold_sql": gold_sql,
        "predicted_res_preview": str(predicted_res[:3]) if correct else str(predicted_res[:1]),
    }


def run_evaluation(
    predictions_path: str,
    dev_json_path: str,
    db_root_path: str,
    num_cpus: int = 8,
    timeout: int = 30,
) -> dict:
    """
    Run full evaluation on predictions.

    Returns dict with overall and per-difficulty EX scores.
    """
    # Load predictions
    with open(predictions_path) as f:
        predictions = json.load(f)

    # Load dev data (for gold SQL and difficulty)
    with open(dev_json_path) as f:
        dev_data = json.load(f)

    db_root = Path(db_root_path)

    # Build evaluation tasks
    eval_tasks = []
    for i, entry in enumerate(dev_data):
        idx_str = str(i)
        if idx_str not in predictions:
            continue

        pred_line = predictions[idx_str]
        # Parse prediction format: "SQL\t----- bird -----\tdb_id"
        parts = pred_line.split("\t----- bird -----\t")
        predicted_sql = parts[0].strip() if parts else "SELECT 1"

        gold_sql = entry["SQL"]
        db_id = entry["db_id"]
        difficulty = entry.get("difficulty", "unknown")
        db_path = str(db_root / db_id / f"{db_id}.sqlite")

        eval_tasks.append((i, predicted_sql, gold_sql, db_path, difficulty, timeout))

    print(f"Evaluating {len(eval_tasks)} predictions...")

    # Run evaluation in parallel
    with mp.Pool(processes=num_cpus) as pool:
        results = list(pool.map(evaluate_single, eval_tasks))

    # Aggregate results
    total_correct = sum(1 for r in results if r["correct"])
    total = len(results)

    # Per-difficulty breakdown
    diff_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        d = r["difficulty"]
        diff_stats[d]["total"] += 1
        if r["correct"]:
            diff_stats[d]["correct"] += 1

    # Build report
    report = {
        "overall": {
            "total": total,
            "correct": total_correct,
            "ex_accuracy": round(total_correct / total * 100, 2) if total > 0 else 0,
        },
        "by_difficulty": {},
    }

    for diff in ["simple", "moderate", "challenging"]:
        if diff in diff_stats:
            stats = diff_stats[diff]
            report["by_difficulty"][diff] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "ex_accuracy": round(stats["correct"] / stats["total"] * 100, 2) if stats["total"] > 0 else 0,
            }

    # Collect errors for analysis
    errors = [r for r in results if not r["correct"]]

    return report, errors
