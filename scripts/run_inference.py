"""Run Text-to-SQL inference on BIRD dev set using Qwen3.5-35B-A3B-FP8."""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prompt_builder import build_prompt
from src.schema_extractor import SchemaCache
from src.sql_parser import parse_sql


async def infer_single(
    client: AsyncOpenAI,
    model: str,
    entry: dict,
    idx: int,
    schema_cache: SchemaCache,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> tuple[int, str, str]:
    """Run inference for a single entry. Returns (idx, predicted_sql, db_id)."""
    async with semaphore:
        db_id = entry["db_id"]
        schema = schema_cache.get(db_id)
        messages = build_prompt(schema, entry["question"], entry.get("evidence", ""))

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            sql = parse_sql(content)
        except Exception as e:
            print(f"\n[ERROR] Entry {idx} (db={db_id}): {e}")
            sql = "SELECT 1"

        return idx, sql, db_id


async def run_inference(args):
    # Load dev data
    with open(args.dev_json_path) as f:
        dev_data = json.load(f)

    # Optionally limit entries
    if args.limit:
        dev_data = dev_data[: args.limit]

    print(f"Total entries: {len(dev_data)}")
    print(f"Model: {args.model}")
    print(f"API base: {args.api_base}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max tokens: {args.max_tokens}")

    # Load existing results for resume
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if output_path.exists() and args.resume:
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} entries already completed")

    schema_cache = SchemaCache(args.db_root_path, num_sample_rows=3)
    client = AsyncOpenAI(base_url=args.api_base, api_key="dummy")
    semaphore = asyncio.Semaphore(args.concurrency)

    # Filter out already completed entries
    pending = []
    for i, entry in enumerate(dev_data):
        if str(i) not in results:
            pending.append((i, entry))

    print(f"Pending: {len(pending)} entries")
    if not pending:
        print("All entries already completed.")
        return

    # Create tasks
    tasks = [
        infer_single(client, args.model, entry, idx, schema_cache, args.max_tokens, args.temperature, semaphore)
        for idx, entry in pending
    ]

    # Run with progress bar
    completed = 0
    save_interval = 50
    start_time = time.time()
    pbar = tqdm(total=len(pending), desc="Inference")

    for coro in asyncio.as_completed(tasks):
        idx, sql, db_id = await coro
        results[str(idx)] = f"{sql}\t----- bird -----\t{db_id}"
        completed += 1
        pbar.update(1)

        # Periodic save
        if completed % save_interval == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    pbar.close()

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\nDone! {len(pending)} entries in {elapsed:.1f}s ({elapsed/len(pending):.1f}s/entry)")
    print(f"Results saved to: {output_path}")
    print(f"Total results: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Run BIRD Text-to-SQL inference")
    parser.add_argument("--dev_json_path", default="data/dev.json")
    parser.add_argument("--db_root_path", default="data/dev_databases")
    parser.add_argument("--output_path", default="results/predict_dev.json")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    parser.add_argument("--api_base", default="http://localhost:8000/v1")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of entries (for testing)")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing results")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    asyncio.run(run_inference(args))


if __name__ == "__main__":
    main()
