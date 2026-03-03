"""Generate training data for the pairwise selection model.

Runs the 3 generators on dev data, executes all candidates against gold SQL,
and creates pairwise (correct, incorrect) training samples.
"""

import argparse
import asyncio
import json
import random
import sqlite3
import sys
from pathlib import Path

from func_timeout import FunctionTimedOut, func_timeout
from tqdm import tqdm

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import importlib
SchemaCache = importlib.import_module("src.schema_extractor").SchemaCache
parse_sql = importlib.import_module("src.sql_parser").parse_sql

from optimization.src.generators import divide_conquer, query_plan, synthetic_examples


def execute_sql(db_path: str, sql: str, timeout: int = 30) -> list | None:
    """Execute SQL, return results or None on error."""
    def _exec():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return result

    try:
        return func_timeout(timeout, _exec)
    except (FunctionTimedOut, Exception):
        return None


def is_correct(predicted_res: list | None, gold_res: list | None) -> bool:
    """Check if predicted result matches gold result (order-independent)."""
    if predicted_res is None or gold_res is None:
        return False
    return set(predicted_res) == set(gold_res)


async def generate_candidates_for_entry(
    client, model, schema, question, evidence, value_hint, n_candidates=5
):
    """Generate candidates from all 3 generators for one entry."""
    tasks = [
        divide_conquer.generate_candidates(
            client, model, schema, question, evidence, value_hint, n_candidates, 0.8
        ),
        query_plan.generate_candidates(
            client, model, schema, question, evidence, value_hint, n_candidates, 0.8
        ),
        synthetic_examples.generate_candidates(
            client, model, schema, question, evidence, value_hint, n_candidates, 0.8
        ),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_sqls = []
    for result in results:
        if isinstance(result, Exception):
            continue
        for raw in result:
            if not raw.startswith("ERROR:"):
                sql = parse_sql(raw)
                all_sqls.append(sql)

    return list(set(all_sqls))  # Deduplicate


async def main_async(args):
    from openai import AsyncOpenAI

    with open(args.dev_json_path) as f:
        dev_data = json.load(f)

    if args.limit:
        dev_data = dev_data[:args.limit]

    print(f"Processing {len(dev_data)} entries for training data generation")

    client = AsyncOpenAI(base_url=args.api_base, api_key="dummy")
    schema_cache = SchemaCache(args.db_root_path)
    db_root = Path(args.db_root_path)

    training_samples = []
    semaphore = asyncio.Semaphore(args.concurrency)

    async def process_entry(i, entry):
        async with semaphore:
            db_id = entry["db_id"]
            question = entry["question"]
            evidence = entry.get("evidence", "")
            gold_sql = entry["SQL"]
            schema = schema_cache.get(db_id)
            db_path = str(db_root / db_id / f"{db_id}.sqlite")

            # Generate candidates
            candidates = await generate_candidates_for_entry(
                client, args.model, schema, question, evidence, "",
                args.n_candidates,
            )

            # Execute gold SQL
            gold_res = execute_sql(db_path, gold_sql)
            if gold_res is None:
                return []

            # Classify candidates
            correct_sqls = []
            incorrect_sqls = []
            for sql in candidates:
                pred_res = execute_sql(db_path, sql)
                if is_correct(pred_res, gold_res):
                    correct_sqls.append(sql)
                else:
                    incorrect_sqls.append(sql)

            # Generate pairwise samples
            samples = []
            for c in correct_sqls:
                for ic in incorrect_sqls:
                    samples.append({
                        "question_id": i,
                        "question": question,
                        "evidence": evidence,
                        "schema": schema[:2000],  # Truncate for 1.5B model
                        "sql_correct": c,
                        "sql_incorrect": ic,
                        "db_id": db_id,
                    })

            return samples

    tasks = [process_entry(i, entry) for i, entry in enumerate(dev_data)]
    pbar = tqdm(total=len(tasks), desc="Generating training data")

    for coro in asyncio.as_completed(tasks):
        samples = await coro
        training_samples.extend(samples)
        pbar.update(1)

    pbar.close()

    # Balance and shuffle
    random.shuffle(training_samples)

    # Limit samples if too many
    if args.max_samples and len(training_samples) > args.max_samples:
        training_samples = training_samples[:args.max_samples]

    # Convert to training format
    train_data = []
    for sample in training_samples:
        # Format for binary classification
        # Randomly assign correct/incorrect to A/B to avoid position bias
        if random.random() < 0.5:
            text = (
                f"Question: {sample['question']}\n"
                f"Evidence: {sample['evidence']}\n"
                f"Schema: {sample['schema']}\n"
                f"SQL A: {sample['sql_correct']}\n"
                f"SQL B: {sample['sql_incorrect']}\n"
                f"Which is correct?"
            )
            label = 0  # A is correct
        else:
            text = (
                f"Question: {sample['question']}\n"
                f"Evidence: {sample['evidence']}\n"
                f"Schema: {sample['schema']}\n"
                f"SQL A: {sample['sql_incorrect']}\n"
                f"SQL B: {sample['sql_correct']}\n"
                f"Which is correct?"
            )
            label = 1  # B is correct

        train_data.append({"text": text, "label": label})

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(train_data)} training samples")
    print(f"From {len(dev_data)} questions")
    print(f"Saved to: {output_path}")

    # Save split for validation
    split_idx = int(len(train_data) * 0.9)
    train_split = train_data[:split_idx]
    val_split = train_data[split_idx:]

    train_path = output_path.with_stem(output_path.stem + "_train")
    val_path = output_path.with_stem(output_path.stem + "_val")

    with open(train_path, "w") as f:
        json.dump(train_split, f, indent=2, ensure_ascii=False)
    with open(val_path, "w") as f:
        json.dump(val_split, f, indent=2, ensure_ascii=False)

    print(f"Train split: {len(train_split)} samples → {train_path}")
    print(f"Val split: {len(val_split)} samples → {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate selector training data")
    parser.add_argument("--dev_json_path", default="data/dev.json")
    parser.add_argument("--db_root_path", default="data/dev_databases")
    parser.add_argument("--output_path", default="optimization/training/selector_data.json")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    parser.add_argument("--api_base", default="http://localhost:8000/v1")
    parser.add_argument("--n_candidates", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=10000)
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
