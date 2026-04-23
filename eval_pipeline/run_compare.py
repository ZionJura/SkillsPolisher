"""
run_compare.py — Run all baselines on a dataset and print a comparison table.

Usage:
    python eval_pipeline/run_compare.py --dataset gsm8k --split test --n_samples 100 --mock

Run from project root: /mnt/d/Code/AI4R/Skills-Learning2/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is in path
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval_pipeline.run_eval import (
    load_config,
    build_llm_client,
    build_dataset,
    build_train_dataset,
    build_baseline,
    build_evaluator,
)

ALL_BASELINES = ["zero_shot", "random_kshot", "case_bandit", "bpo_rewrite"]


def print_comparison_table(compare_results: list) -> None:
    """Print a formatted comparison table."""
    # Header
    col_widths = {
        "baseline": 18,
        "n": 8,
        "accuracy": 12,
        "avg_score": 12,
        "n_calls": 10,
        "tokens": 12,
        "time_s": 10,
    }

    def fmt(s, w):
        return str(s)[:w].ljust(w)

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths.values()) + "+"
    header = (
        "| "
        + " | ".join(fmt(k, w) for k, w in col_widths.items())
        + " |"
    )

    print(sep)
    print(header)
    print(sep)

    for r in compare_results:
        row = (
            "| "
            + " | ".join([
                fmt(r.get("baseline", ""), col_widths["baseline"]),
                fmt(r.get("n_samples", 0), col_widths["n"]),
                fmt(f"{r.get('accuracy', 0):.1%}", col_widths["accuracy"]),
                fmt(f"{r.get('avg_score', 0):.3f}", col_widths["avg_score"]),
                fmt(r.get("n_calls", 0), col_widths["n_calls"]),
                fmt(r.get("total_tokens", 0), col_widths["tokens"]),
                fmt(f"{r.get('elapsed_s', 0):.1f}s", col_widths["time_s"]),
            ])
            + " |"
        )
        print(row)

    print(sep)


def run_single_baseline(
    baseline_name: str,
    dataset,
    samples: list,
    train_samples,
    llm_client,
    evaluator,
    dataset_name: str,
    k: int,
    seed: int,
    verbose: bool,
    rewrite_mode: str = "mock",
) -> dict:
    """Run a single baseline and return results summary."""
    print(f"\n  Running {baseline_name}...")

    # Build baseline
    try:
        baseline = build_baseline(
            baseline_name,
            llm_client=llm_client,
            dataset_name=dataset_name,
            k=k,
            seed=seed,
            train_samples=list(train_samples) if train_samples else [],
            rewrite_mode=rewrite_mode,
        )
    except Exception as e:
        print(f"  [warn] Could not build baseline {baseline_name}: {e}")
        return {
            "baseline": baseline_name,
            "error": str(e),
            "n_samples": 0,
            "accuracy": 0.0,
            "avg_score": 0.0,
            "n_calls": 0,
            "total_tokens": 0,
            "elapsed_s": 0.0,
        }

    # Reset cost tracker for this baseline
    llm_client.reset_cost_tracker()

    start = time.time()
    n_correct = 0
    total_score = 0.0
    result_records = []

    for i, sample in enumerate(samples):
        if verbose and (i % 20 == 0):
            print(f"    [{i+1}/{len(samples)}]")

        try:
            prediction = baseline.predict(sample)
        except ConnectionError as e:
            print(f"\n  ERROR: Network unreachable: {e}")
            prediction = ""
        except Exception as e:
            prediction = ""

        try:
            score_result = evaluator.score(prediction, sample)
        except Exception as e:
            score_result = {"correct": False, "score": 0.0, "details": str(e)}

        correct = score_result.get("correct", False)
        score = score_result.get("score", 0.0)

        if correct:
            n_correct += 1
        total_score += score

        result_records.append({
            "id": sample.id,
            "question": sample.question[:150],
            "prediction": (prediction or "")[:300],
            "answer": (sample.answer or "")[:150],
            "correct": correct,
            "score": score,
        })

    elapsed = time.time() - start
    n = len(samples)
    accuracy = n_correct / n if n > 0 else 0.0
    avg_score = total_score / n if n > 0 else 0.0
    cost = llm_client.cost_tracker.to_dict()

    print(
        f"    Done: {n_correct}/{n} correct ({accuracy:.1%}), "
        f"time={elapsed:.1f}s, tokens={cost['total_tokens']}"
    )

    return {
        "baseline": baseline_name,
        "n_samples": n,
        "accuracy": accuracy,
        "avg_score": avg_score,
        "results": result_records,
        "n_calls": cost["n_calls"],
        "total_tokens": cost["total_tokens"],
        "elapsed_s": elapsed,
    }


def run_compare(
    dataset_name: str,
    split: str,
    n_samples: int,
    seed: int,
    output_path: str,
    mock: bool,
    verbose: bool,
    k: int,
    baselines: list,
    config: dict,
    rewrite_mode: str = "mock",
) -> dict:
    """Run all baselines and produce comparison."""
    print(f"\n{'='*60}")
    print(f"Comparison: {dataset_name} ({split})")
    print(f"Baselines: {', '.join(baselines)}")
    print(f"Samples: {n_samples if n_samples > 0 else 'all'}, Mock: {mock}")
    print(f"{'='*60}")

    # Build shared components
    llm_client = build_llm_client(config, mock=mock)
    print(f"LLM: {llm_client}")

    # Load dataset
    print(f"\nLoading {dataset_name} ({split})...")
    try:
        dataset = build_dataset(dataset_name, split)
        print(f"  {len(dataset)} samples")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if n_samples > 0 and n_samples < len(dataset):
        samples = dataset.get_subset(n_samples, seed=seed)
    else:
        samples = list(dataset)

    print(f"  Using {len(samples)} samples")

    # Load train data
    train_samples = None
    if any(b in baselines for b in ("random_kshot", "case_bandit", "bpo_rewrite")):
        print(f"Loading training split...")
        train_dataset = build_train_dataset(dataset_name)
        if train_dataset is not None:
            train_samples = list(train_dataset)
            print(f"  {len(train_samples)} training samples")

    # Build shared evaluator
    evaluator = build_evaluator(dataset_name, llm_client=llm_client)

    # Run each baseline
    compare_results = []
    for baseline_name in baselines:
        result = run_single_baseline(
            baseline_name=baseline_name,
            dataset=dataset,
            samples=samples,
            train_samples=train_samples,
            llm_client=llm_client,
            evaluator=evaluator,
            dataset_name=dataset_name,
            k=k,
            seed=seed,
            verbose=verbose,
            rewrite_mode=rewrite_mode,
        )
        compare_results.append(result)

    # Print table
    print(f"\n\n{'='*60}")
    print(f"COMPARISON: {dataset_name.upper()} ({split}), n={len(samples)}")
    print(f"{'='*60}")
    print_comparison_table(compare_results)

    # Build output
    output = {
        "dataset": dataset_name,
        "split": split,
        "n_samples": len(samples),
        "seed": seed,
        "mock": mock,
        "baselines": compare_results,
        "summary": {
            r["baseline"]: {
                "accuracy": r.get("accuracy", 0.0),
                "avg_score": r.get("avg_score", 0.0),
            }
            for r in compare_results
        },
    }

    # Save output
    if output_path:
        out_path = Path(output_path)
        if not out_path.is_absolute():
            results_dir = Path(config.get("results_dir", str(_SCRIPT_DIR / "results")))
            out_path = results_dir / output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nComparison saved to: {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Compare all baselines on a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all baselines on GSM8K (mock mode)
  python eval_pipeline/run_compare.py --dataset gsm8k --split test --n_samples 50 --mock

  # Compare specific baselines
  python eval_pipeline/run_compare.py --dataset aqua_rat --split dev \\
      --baselines zero_shot random_kshot --n_samples 50 --mock
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name from registry",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples (default: 100)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Demos for k-shot baselines (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/compare_<dataset>.json)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM (offline testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress for each baseline",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=ALL_BASELINES,
        choices=ALL_BASELINES,
        help=f"Baselines to compare (default: all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to eval_config.json",
    )
    parser.add_argument(
        "--rewrite_mode",
        type=str,
        default="mock",
        choices=["mock", "hf", "openai", "claude", "auto"],
        help="BPO rewrite mode (default: mock)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["openai", "azure"],
        help="LLM backend: 'openai' (zhizengzeng, default) or 'azure' (ByteDance internal, requires VPN)",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = f"compare_{args.dataset}.json"

    config = load_config(args.config)
    if args.backend:
        config.setdefault("llm", {})["backend"] = args.backend
        import os; os.environ["LLM_BACKEND"] = args.backend

    run_compare(
        dataset_name=args.dataset,
        split=args.split,
        n_samples=args.n_samples,
        seed=args.seed,
        output_path=args.output,
        mock=args.mock,
        verbose=args.verbose,
        k=args.k,
        baselines=args.baselines,
        config=config,
        rewrite_mode=args.rewrite_mode,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
