"""Run BIRD evaluation on predictions."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate BIRD Text-to-SQL predictions")
    parser.add_argument("--predicted_sql_path", default="results/predict_dev.json")
    parser.add_argument("--dev_json_path", default="data/dev.json")
    parser.add_argument("--db_root_path", default="data/dev_databases")
    parser.add_argument("--num_cpus", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output_path", default="results/eval_report.txt")
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
    print("BIRD Benchmark Evaluation Results")
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
    print("=" * 60)

    # Save report
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("BIRD Benchmark Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: Qwen3.5-35B-A3B-FP8\n")
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

    # Save detailed JSON report
    json_report_path = output_path.with_suffix(".json")
    with open(json_report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save error analysis (first 50 errors)
    if errors:
        error_path = output_path.parent / "error_analysis.json"
        with open(error_path, "w") as f:
            json.dump(errors[:50], f, indent=2, ensure_ascii=False)
        print(f"\nError analysis saved: {error_path} ({len(errors)} total errors, first 50 saved)")

    print(f"Report saved: {output_path}")
    print(f"JSON report saved: {json_report_path}")


if __name__ == "__main__":
    main()
