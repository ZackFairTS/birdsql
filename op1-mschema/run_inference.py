"""Run Text-to-SQL inference on BIRD dev set with M-Schema."""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mschema_extractor import MSchemaCache
from prompt_builder import build_prompt
from src.sql_parser import parse_sql


async def infer_single(
    client: AsyncOpenAI,
    model: str,
    entry: dict,
    idx: int,
    schema_cache: MSchemaCache,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    enable_thinking: bool = True,
) -> tuple[int, str, str]:
    """Run inference for a single entry. Returns (idx, predicted_sql, db_id)."""
    async with semaphore:
        db_id = entry["db_id"]
        schema = schema_cache.get(db_id)
        messages = build_prompt(schema, entry["question"], entry.get("evidence", ""))

        extra_body = {}
        if not enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body if extra_body else None,
            )
            content = response.choices[0].message.content or ""
            sql = parse_sql(content)
        except Exception as e:
            print(f"\n[ERROR] Entry {idx} (db={db_id}): {e}")
            sql = "SELECT 1"

        return idx, sql, db_id


async def run_inference(args):
    with open(args.dev_json_path) as f:
        dev_data = json.load(f)

    if args.limit:
        dev_data = dev_data[:args.limit]

    print(f"Total entries: {len(dev_data)}")
    print(f"Model: {args.model}")
    print(f"API base: {args.api_base}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Thinking: {not args.no_thinking}")
    print(f"Schema format: M-Schema")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if output_path.exists() and args.resume:
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} entries already completed")

    schema_cache = MSchemaCache(args.db_root_path, num_examples=3)
    client = AsyncOpenAI(base_url=args.api_base, api_key="dummy")
    semaphore = asyncio.Semaphore(args.concurrency)

    pending = []
    for i, entry in enumerate(dev_data):
        if str(i) not in results:
            pending.append((i, entry))

    print(f"Pending: {len(pending)} entries")
    if not pending:
        print("All entries already completed.")
        return

    enable_thinking = not args.no_thinking
    tasks = [
        infer_single(client, args.model, entry, idx, schema_cache,
                     args.max_tokens, args.temperature, semaphore, enable_thinking)
        for idx, entry in pending
    ]

    completed = 0
    save_interval = 50
    start_time = time.time()
    pbar = tqdm(total=len(pending), desc="MSchema Inference")

    for coro in asyncio.as_completed(tasks):
        idx, sql, db_id = await coro
        results[str(idx)] = f"{sql}\t----- bird -----\t{db_id}"
        completed += 1
        pbar.update(1)

        if completed % save_interval == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    pbar.close()

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\nDone! {len(pending)} entries in {elapsed:.1f}s ({elapsed/len(pending):.1f}s/entry)")
    print(f"Results saved to: {output_path}")
    print(f"Total results: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Run BIRD Text-to-SQL inference with M-Schema")
    parser.add_argument("--dev_json_path", default="data/dev.json")
    parser.add_argument("--db_root_path", default="data/dev_databases")
    parser.add_argument("--output_path", default="op1-mschema/results/predict_dev.json")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    parser.add_argument("--api_base", default="http://localhost:8000/v1")
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--no_thinking", action="store_true", default=True,
                        help="Disable model thinking/reasoning (default: on, M-Schema triggers excessive reasoning)")
    parser.add_argument("--enable_thinking", dest="no_thinking", action="store_false")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    asyncio.run(run_inference(args))


if __name__ == "__main__":
    main()
