"""Run CHASE-SQL optimized inference on BIRD dev set."""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OPTIMIZATION_ROOT = PROJECT_ROOT / "optimization"
sys.path.insert(0, str(PROJECT_ROOT))

from optimization.src.pipeline import ChaseSQLPipeline, PipelineConfig


async def run(args):
    # Load dev data
    with open(args.dev_json_path) as f:
        dev_data = json.load(f)

    if args.limit:
        dev_data = dev_data[:args.limit]

    print(f"Total entries: {len(dev_data)}")
    print(f"Model: {args.model}")
    print(f"Generators: DC={args.enable_dc}, QP={args.enable_qp}, SE={args.enable_se}")
    print(f"Candidates per generator: {args.n_candidates}")
    print(f"Query Fixer: {args.enable_fixer} (max_iter={args.fixer_iterations})")
    print(f"Selection: {args.selection_method}")
    print(f"Value Retrieval: {args.enable_value_retrieval}")
    print(f"Concurrency: {args.concurrency}")

    # Build config
    config = PipelineConfig(
        model=args.model,
        api_base=args.api_base,
        n_candidates_per_generator=args.n_candidates,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_value_retrieval=args.enable_value_retrieval,
        enable_divide_conquer=args.enable_dc,
        enable_query_plan=args.enable_qp,
        enable_synthetic_examples=args.enable_se,
        enable_query_fixer=args.enable_fixer,
        fixer_max_iterations=args.fixer_iterations,
        selection_method=args.selection_method,
        trained_selector_path=args.trained_selector_path,
        db_root_path=args.db_root_path,
        concurrency=args.concurrency,
    )

    pipeline = ChaseSQLPipeline(config)

    # Load existing results for resume
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_results = {}
    if output_path.exists() and args.resume:
        with open(output_path) as f:
            existing_results = json.load(f)
        print(f"Resuming: {len(existing_results)} entries already completed")

    # Filter pending entries
    pending = [(i, entry) for i, entry in enumerate(dev_data) if str(i) not in existing_results]
    print(f"Pending: {len(pending)} entries")

    if not pending:
        print("All entries already completed.")
        return

    # Process with progress bar
    results = dict(existing_results)
    pbar = tqdm(total=len(pending), desc="CHASE-SQL Inference")
    start_time = time.time()
    save_interval = 5
    completed = 0

    semaphore = asyncio.Semaphore(args.concurrency)

    async def process_one(idx: int, entry: dict):
        async with semaphore:
            return await pipeline.process_single(entry, idx)

    tasks = {
        asyncio.ensure_future(process_one(idx, entry)): (idx, entry)
        for idx, entry in pending
    }

    for coro in asyncio.as_completed(tasks.keys()):
        try:
            result = await coro
            db_id = result.db_id
            results[str(result.question_id)] = f"{result.selected_sql}\t----- bird -----\t{db_id}"
            completed += 1
            pbar.update(1)

            # Periodic save
            if completed % save_interval == 0:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            # Find which entry failed
            pbar.update(1)
            completed += 1
            print(f"\n[ERROR] {e}")

    pbar.close()

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\nDone! {len(pending)} entries in {elapsed:.1f}s ({elapsed/len(pending):.1f}s/entry)")
    print(f"Results saved to: {output_path}")
    print(f"Total results: {len(results)}")

    # Save detailed results if requested
    if args.save_details:
        details_path = output_path.with_suffix(".details.json")
        print(f"Detailed results would be saved to: {details_path}")


def main():
    parser = argparse.ArgumentParser(description="Run CHASE-SQL optimized inference")
    parser.add_argument("--dev_json_path", default="data/dev.json")
    parser.add_argument("--db_root_path", default="data/dev_databases")
    parser.add_argument("--output_path", default="optimization/results/predict_optimized.json")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    parser.add_argument("--api_base", default="http://localhost:8000/v1")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--n_candidates", type=int, default=7, help="Candidates per generator")

    # Feature flags
    parser.add_argument("--enable_value_retrieval", action="store_true", default=True)
    parser.add_argument("--no_value_retrieval", dest="enable_value_retrieval", action="store_false")
    parser.add_argument("--enable_dc", action="store_true", default=True, help="Divide & Conquer")
    parser.add_argument("--no_dc", dest="enable_dc", action="store_false")
    parser.add_argument("--enable_qp", action="store_true", default=True, help="Query Plan")
    parser.add_argument("--no_qp", dest="enable_qp", action="store_false")
    parser.add_argument("--enable_se", action="store_true", default=True, help="Synthetic Examples")
    parser.add_argument("--no_se", dest="enable_se", action="store_false")
    parser.add_argument("--enable_fixer", action="store_true", default=True, help="Query Fixer")
    parser.add_argument("--no_fixer", dest="enable_fixer", action="store_false")
    parser.add_argument("--fixer_iterations", type=int, default=3)

    # Selection
    parser.add_argument("--selection_method", default="consistency",
                        choices=["consistency", "consistency_empty_penalty", "pairwise_llm", "trained"])
    parser.add_argument("--trained_selector_path", default=None)

    # Other
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--save_details", action="store_true", default=False)

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
