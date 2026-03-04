"""Multi-turn agentic Text-to-SQL inference on BIRD dev set."""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

# Ensure project root is on path for src.* imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema_extractor import SchemaCache
from op1_prompt_and_tools.prompt_builder import build_messages
from op1_prompt_and_tools.tool_executor import execute_tool_call, parse_tool_call
from op1_prompt_and_tools.sql_parser import extract_sql

MAX_TURNS = 10
DEFAULT_MAX_CONTEXT_TOKENS = 30000  # Leave headroom below model's 32768 limit
CHARS_PER_TOKEN = 3.5  # Conservative estimate for mixed English/SQL content
NUDGE_MSG = "Continue your analysis. If you have enough information, proceed to generate sql_candidates, execute your chosen SQL, and emit <final_answer>."
FORCE_FINAL_MSG = "You must now emit your <final_answer> with <sql_code>. Use your best SQL based on what you know so far."


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: total chars / CHARS_PER_TOKEN."""
    return int(sum(len(m.get("content", "")) for m in messages) / CHARS_PER_TOKEN)


def _trim_messages(messages: list[dict], max_context_tokens: int, max_tokens: int) -> list[dict]:
    """Trim middle messages to fit within context budget.

    Keeps: system message (idx 0), initial user message (idx 1), and as many
    recent messages as possible. Drops oldest middle messages first.
    """
    budget = max_context_tokens - max_tokens  # Reserve space for model output
    if budget <= 0:
        budget = max_context_tokens // 2

    if _estimate_tokens(messages) <= budget:
        return messages

    # Always keep first 2 messages (system + initial user with schema)
    pinned_head = messages[:2]
    tail = messages[2:]

    # Keep removing oldest tail messages until we fit
    while tail and _estimate_tokens(pinned_head + tail) > budget:
        tail.pop(0)

    return pinned_head + tail


async def infer_single(
    client: AsyncOpenAI,
    model: str,
    entry: dict,
    idx: int,
    schema_cache: SchemaCache,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    save_traces: bool = False,
    trace_dir: Path | None = None,
) -> tuple[int, str, str]:
    """Run multi-turn agentic inference for a single entry."""
    async with semaphore:
        db_id = entry["db_id"]
        schema = schema_cache.get(db_id)
        messages = build_messages(schema, entry["question"], entry.get("evidence", ""), db_id)

        consecutive_nudges = 0

        for turn in range(MAX_TURNS):
            # Trim context to stay within model limits
            trimmed = _trim_messages(messages, max_context_tokens, max_tokens)

            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=trimmed,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content or ""
            except Exception as e:
                print(f"\n[ERROR] Entry {idx} turn {turn} (db={db_id}): {e}")
                break

            messages.append({"role": "assistant", "content": content})

            # Check for <final_answer>
            if "<final_answer>" in content:
                break

            # Check for <tool_call>
            if parse_tool_call(content) is not None:
                consecutive_nudges = 0
                tool_result = await execute_tool_call(content)
                if tool_result:
                    messages.append({"role": "user", "content": tool_result})
                else:
                    messages.append({"role": "user", "content": "<tool_result>\n{\"error\": \"failed to parse tool call\"}\n</tool_result>"})
                continue

            # Neither final_answer nor tool_call — nudge
            consecutive_nudges += 1
            if consecutive_nudges >= 2:
                messages.append({"role": "user", "content": FORCE_FINAL_MSG})
            else:
                messages.append({"role": "user", "content": NUDGE_MSG})

        # Extract SQL from conversation
        sql = extract_sql(messages)

        # Save trace if requested
        if save_traces and trace_dir:
            trace_path = trace_dir / f"{idx:04d}_{db_id}.json"
            with open(trace_path, "w") as f:
                json.dump({
                    "idx": idx,
                    "db_id": db_id,
                    "question": entry["question"],
                    "evidence": entry.get("evidence", ""),
                    "extracted_sql": sql,
                    "turns": len([m for m in messages if m["role"] == "assistant"]),
                    "messages": messages,
                }, f, indent=2, ensure_ascii=False)

        return idx, sql, db_id


async def run_inference(args):
    # Set BIRD_DB_ROOTS for tool resolution
    os.environ["BIRD_DB_ROOTS"] = args.db_root_path

    # Load dev data
    with open(args.dev_json_path) as f:
        dev_data = json.load(f)

    if args.limit:
        dev_data = dev_data[:args.limit]

    print(f"Total entries: {len(dev_data)}")
    print(f"Model: {args.model}")
    print(f"API base: {args.api_base}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Max context tokens: {args.max_context_tokens}")
    print(f"Max turns per entry: {MAX_TURNS}")

    # Load existing results for resume
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if output_path.exists() and args.resume:
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} entries already completed")

    # Set up trace dir
    trace_dir = None
    if args.save_traces:
        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving traces to: {trace_dir}")

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
        infer_single(
            client, args.model, entry, idx, schema_cache,
            args.max_tokens, args.temperature, semaphore,
            args.max_context_tokens, args.save_traces, trace_dir,
        )
        for idx, entry in pending
    ]

    # Run with progress bar
    completed = 0
    save_interval = 20
    start_time = time.time()
    pbar = tqdm(total=len(pending), desc="OP1 Inference")

    for coro in asyncio.as_completed(tasks):
        idx, sql, db_id = await coro
        results[str(idx)] = f"{sql}\t----- bird -----\t{db_id}"
        completed += 1
        pbar.update(1)

        if completed % save_interval == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    pbar.close()

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\nDone! {len(pending)} entries in {elapsed:.1f}s ({elapsed / max(len(pending), 1):.1f}s/entry)")
    print(f"Results saved to: {output_path}")
    print(f"Total results: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="OP1: Multi-turn agentic Text-to-SQL inference")
    parser.add_argument("--dev_json_path", default="data/dev.json")
    parser.add_argument("--db_root_path", default="data/dev_databases")
    parser.add_argument("--output_path", default="results/op1_predict_dev.json")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    parser.add_argument("--api_base", default="http://localhost:8000/v1")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max_context_tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS,
                        help="Max context tokens budget (default: 30000, model limit is 32768)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--save_traces", action="store_true", default=False)
    parser.add_argument("--trace_dir", default="results/op1_traces")
    args = parser.parse_args()

    asyncio.run(run_inference(args))


if __name__ == "__main__":
    main()
