"""Run evaluation on optimized predictions.

Reuses the base evaluation module from birdsql/src/evaluation.py.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate CHASE-SQL optimized predictions")
    parser.add_argument("--predicted_sql_path", default="optimization/results/predict_optimized.json")
    parser.add_argument("--dev_json_path", default="data/dev.json")
    parser.add_argument("--db_root_path", default="data/dev_databases")
    parser.add_argument("--num_cpus", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output_path", default="optimization/results/eval_report.txt")
    parser.add_argument("--baseline_path", default=None, help="Baseline results for comparison")
    args = parser.parse_args()

    report, errors = run_evaluation(
        predictions_path=args.predicted_sql_path,
        dev_json_path=args.dev_json_path,
        db_root_path=args.db_root_path,
        num_cpus=args.num_cpus,
        timeout=args.timeout,
    )

    # Print report
    print("\n" + "=" * 60)
    print("CHASE-SQL Optimized Evaluation Results")
    print("=" * 60)

    overall = report["overall"]
    print(f"\nOverall EX Accuracy: {overall['ex_accuracy']}%  ({overall['correct']}/{overall['total']})")

    print(f"\n{'Difficulty':<15} {'Correct':>8} {'Total':>8} {'EX (%)':>8}")
    print("-" * 45)
    for diff in ["simple", "moderate", "challenging"]:
        if diff in report["by_difficulty"]:
            d = report["by_difficulty"][diff]
            print(f"{diff:<15} {d['correct']:>8} {d['total']:>8} {d['ex_accuracy']:>7.2f}%")

    print("-" * 45)
    print(f"{'ALL':<15} {overall['correct']:>8} {overall['total']:>8} {overall['ex_accuracy']:>7.2f}%")

    # Compare with baseline if provided
    if args.baseline_path:
        baseline_path = Path(args.baseline_path)
        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline = json.load(f)
            baseline_overall = baseline["overall"]
            delta = overall["ex_accuracy"] - baseline_overall["ex_accuracy"]
            print(f"\n{'Baseline EX:':<25} {baseline_overall['ex_accuracy']}%")
            print(f"{'Optimized EX:':<25} {overall['ex_accuracy']}%")
            print(f"{'Delta:':<25} {'+' if delta >= 0 else ''}{delta:.2f}%")

            print(f"\n{'Difficulty':<15} {'Baseline':>8} {'Optimized':>10} {'Delta':>8}")
            print("-" * 45)
            for diff in ["simple", "moderate", "challenging"]:
                if diff in report["by_difficulty"] and diff in baseline.get("by_difficulty", {}):
                    opt = report["by_difficulty"][diff]["ex_accuracy"]
                    base = baseline["by_difficulty"][diff]["ex_accuracy"]
                    d = opt - base
                    print(f"{diff:<15} {base:>7.2f}% {opt:>9.2f}% {'+' if d >= 0 else ''}{d:>6.2f}%")

    print("=" * 60)

    # Save report
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("CHASE-SQL Optimized Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Predictions: {args.predicted_sql_path}\n\n")
        f.write(f"Overall EX Accuracy: {overall['ex_accuracy']}%  ({overall['correct']}/{overall['total']})\n\n")
        f.write(f"{'Difficulty':<15} {'Correct':>8} {'Total':>8} {'EX (%)':>8}\n")
        f.write("-" * 45 + "\n")
        for diff in ["simple", "moderate", "challenging"]:
            if diff in report["by_difficulty"]:
                d = report["by_difficulty"][diff]
                f.write(f"{diff:<15} {d['correct']:>8} {d['total']:>8} {d['ex_accuracy']:>7.2f}%\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'ALL':<15} {overall['correct']:>8} {overall['total']:>8} {overall['ex_accuracy']:>7.2f}%\n")

    # Save JSON report
    json_report_path = output_path.with_suffix(".json")
    with open(json_report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save error analysis
    if errors:
        error_path = output_path.parent / "error_analysis_optimized.json"
        with open(error_path, "w") as f:
            json.dump(errors[:50], f, indent=2, ensure_ascii=False)
        print(f"\nError analysis saved: {error_path} ({len(errors)} errors, first 50 saved)")

    print(f"Report saved: {output_path}")
    print(f"JSON report saved: {json_report_path}")


if __name__ == "__main__":
    main()
