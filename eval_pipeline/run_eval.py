"""
run_eval.py — Main evaluation script for the SPLICE eval pipeline.

Usage:
    python eval_pipeline/run_eval.py \
        --dataset gsm8k \
        --split test \
        --baseline zero_shot \
        --n_samples 100 \
        --seed 42 \
        --output results/gsm8k_zero_shot.json \
        --mock

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


def load_config(config_path: str = None) -> dict:
    """Load eval configuration."""
    if config_path is None:
        config_path = str(_SCRIPT_DIR / "configs" / "eval_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def build_llm_client(config: dict, mock: bool = False):
    """Build LLM client from config."""
    from eval_pipeline.llm_client import LLMClient
    return LLMClient.from_config(config, mock_mode=mock)


def build_dataset(dataset_name: str, split: str):
    """Load dataset from registry."""
    from eval_pipeline.datasets import load_dataset
    return load_dataset(dataset_name, split=split)


def build_train_dataset(dataset_name: str):
    """Load training split for demo selection."""
    from eval_pipeline.datasets import load_dataset, DATASET_REGISTRY
    try:
        return load_dataset(dataset_name, split="train")
    except (FileNotFoundError, ValueError):
        # Some datasets don't have train split; return None
        return None


def build_baseline(
    baseline_name: str,
    llm_client,
    dataset_name: str,
    k: int,
    seed: int,
    train_samples=None,
    rewrite_mode: str = "mock",
):
    """Build baseline from name."""
    from eval_pipeline.baselines import (
        ZeroShotBaseline,
        RandomKShotBaseline,
        CASEBanditBaseline,
        BPORewriteBaseline,
    )

    if baseline_name == "zero_shot":
        return ZeroShotBaseline(
            llm_client=llm_client,
            dataset_name=dataset_name,
        )
    elif baseline_name == "random_kshot":
        return RandomKShotBaseline(
            llm_client=llm_client,
            k=k,
            seed=seed,
            dataset_name=dataset_name,
            train_samples=list(train_samples) if train_samples else [],
        )
    elif baseline_name == "case_bandit":
        return CASEBanditBaseline(
            llm_client=llm_client,
            k=k,
            dataset_name=dataset_name,
            train_samples=list(train_samples) if train_samples else [],
            seed=seed,
        )
    elif baseline_name == "bpo_rewrite":
        return BPORewriteBaseline(
            llm_client=llm_client,
            rewrite_mode=rewrite_mode,
            dataset_name=dataset_name,
            train_samples=list(train_samples) if train_samples else [],
            k_demos=k,
        )
    else:
        raise ValueError(
            f"Unknown baseline: {baseline_name!r}. "
            f"Available: zero_shot, random_kshot, case_bandit, bpo_rewrite"
        )


def build_evaluator(dataset_name: str, llm_client):
    """Build evaluator for dataset."""
    from eval_pipeline.evaluators import get_evaluator
    return get_evaluator(dataset_name, llm_client=llm_client)


def run_evaluation(
    dataset_name: str,
    split: str,
    baseline_name: str,
    n_samples: int,
    seed: int,
    output_path: str,
    mock: bool,
    verbose: bool,
    k: int,
    config: dict,
    rewrite_mode: str = "mock",
) -> dict:
    """Run evaluation and return results dict."""

    print(f"\n{'='*60}")
    print(f"Dataset:  {dataset_name} ({split})")
    print(f"Baseline: {baseline_name}")
    print(f"Samples:  {n_samples if n_samples > 0 else 'all'}")
    print(f"Mock:     {mock}")
    print(f"{'='*60}\n")

    # Build components
    llm_client = build_llm_client(config, mock=mock)
    print(f"LLM client: {llm_client}")

    # Load dataset
    print(f"Loading dataset: {dataset_name} ({split})...")
    try:
        dataset = build_dataset(dataset_name, split)
        print(f"  Loaded {len(dataset)} samples")
    except FileNotFoundError as e:
        print(f"ERROR: Dataset not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)

    # Get samples
    if n_samples > 0 and n_samples < len(dataset):
        samples = dataset.get_subset(n_samples, seed=seed)
        print(f"  Using subset of {len(samples)} samples (seed={seed})")
    else:
        samples = list(dataset)
        print(f"  Using all {len(samples)} samples")

    # Load train data for k-shot baselines
    train_samples = None
    if baseline_name in ("random_kshot", "case_bandit", "bpo_rewrite"):
        print(f"Loading training split for {baseline_name}...")
        train_dataset = build_train_dataset(dataset_name)
        if train_dataset is not None:
            train_samples = list(train_dataset)
            print(f"  Loaded {len(train_samples)} training samples")
        else:
            print(f"  No training split available; demos will be empty")

    # Build baseline
    print(f"Building baseline: {baseline_name}...")
    baseline = build_baseline(
        baseline_name,
        llm_client=llm_client,
        dataset_name=dataset_name,
        k=k,
        seed=seed,
        train_samples=train_samples,
        rewrite_mode=rewrite_mode,
    )
    print(f"  {baseline}")

    # Build evaluator
    evaluator = build_evaluator(dataset_name, llm_client=llm_client)
    print(f"  Evaluator: {evaluator}")

    # Run predictions
    print(f"\nRunning predictions...")
    start_time = time.time()
    results = []
    n_correct = 0
    total_score = 0.0

    for i, sample in enumerate(samples):
        if verbose or (i % 10 == 0 and i > 0):
            elapsed = time.time() - start_time
            rate = (i / elapsed) if elapsed > 0 else 0
            print(
                f"  [{i+1}/{len(samples)}] {sample.id[:40]:<40} "
                f"({rate:.1f} samples/min)"
            )

        # Generate prediction
        try:
            prediction = baseline.predict(sample)
        except ConnectionError as e:
            print(f"\nERROR: Network unreachable: {e}")
            print("Use --mock flag for offline testing.")
            sys.exit(1)
        except Exception as e:
            print(f"  [warn] Prediction error for {sample.id}: {e}")
            prediction = ""

        # Score prediction
        try:
            score_result = evaluator.score(prediction, sample)
        except ConnectionError as e:
            print(f"\nERROR: Evaluator network error: {e}")
            sys.exit(1)
        except Exception as e:
            score_result = {"correct": False, "score": 0.0, "details": f"eval_error: {e}"}

        correct = score_result.get("correct", False)
        score = score_result.get("score", 0.0)

        if correct:
            n_correct += 1
        total_score += score

        result_record = {
            "id": sample.id,
            "question": sample.question[:200],
            "prediction": prediction[:500] if prediction else "",
            "answer": sample.answer[:200] if sample.answer else "",
            "correct": correct,
            "score": score,
            "details": score_result.get("details", ""),
        }
        results.append(result_record)

        if verbose:
            print(f"    Q: {sample.question[:80]!r}")
            print(f"    A: {sample.answer[:60]!r}")
            print(f"    P: {prediction[:60]!r}")
            print(f"    Correct: {correct} | Score: {score:.2f}")
            print()

    elapsed_total = time.time() - start_time
    n = len(results)
    accuracy = n_correct / n if n > 0 else 0.0
    avg_score = total_score / n if n > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Results: {n_correct}/{n} correct ({accuracy:.1%})")
    print(f"Avg score: {avg_score:.3f}")
    print(f"Time: {elapsed_total:.1f}s ({n/elapsed_total:.1f} samples/s)")
    cost_info = llm_client.cost_tracker.to_dict()
    print(f"LLM calls: {cost_info['n_calls']}, tokens: {cost_info['total_tokens']}")
    print(f"{'='*60}\n")

    # Build output dict
    output = {
        "dataset": dataset_name,
        "split": split,
        "baseline": baseline_name,
        "n_samples": n,
        "accuracy": accuracy,
        "avg_score": avg_score,
        "results": results,
        "cost": cost_info,
        "config": {
            "k": k,
            "seed": seed,
            "mock": mock,
            "elapsed_seconds": elapsed_total,
        },
    }

    # Save output
    if output_path:
        out_path = Path(output_path)
        # If relative path, resolve relative to results dir
        if not out_path.is_absolute():
            results_dir = Path(config.get("results_dir", str(_SCRIPT_DIR / "results")))
            out_path = results_dir / output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run SPLICE evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Zero-shot on GSM8K test (first 100 samples), mock mode
  python eval_pipeline/run_eval.py --dataset gsm8k --split test --baseline zero_shot \\
      --n_samples 100 --mock

  # Random k-shot on AQUA-RAT dev
  python eval_pipeline/run_eval.py --dataset aqua_rat --split dev \\
      --baseline random_kshot --k 3 --n_samples 50

  # CASE bandit on StrategyQA
  python eval_pipeline/run_eval.py --dataset strategyqa --split test \\
      --baseline case_bandit --k 3 --n_samples 50 --mock

  # Full test with verbose output
  python eval_pipeline/run_eval.py --dataset gsm8k --split test --baseline zero_shot \\
      --n_samples 10 --verbose --mock
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name from registry (aqua_rat, gsm8k, tabmwp, finqa, "
             "strategyqa, bpo_test, dolly_eval, self_instruct_eval, skillsbench, demo_bank)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "random_kshot", "case_bandit", "bpo_rewrite"],
        help="Baseline method to use (default: zero_shot)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=0,
        help="Number of samples to evaluate (0 = all, default: 0)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of demonstrations for k-shot baselines (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: results/<dataset>_<baseline>.json)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM client (no API calls, for offline testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each prediction and score",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to eval_config.json (default: eval_pipeline/configs/eval_config.json)",
    )
    parser.add_argument(
        "--rewrite_mode",
        type=str,
        default="mock",
        choices=["mock", "hf", "openai", "claude", "auto"],
        help="BPO rewrite mode for bpo_rewrite baseline (default: mock)",
    )

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        args.output = f"{args.dataset}_{args.baseline}.json"

    # Load config
    config = load_config(args.config)

    # Run evaluation
    results = run_evaluation(
        dataset_name=args.dataset,
        split=args.split,
        baseline_name=args.baseline,
        n_samples=args.n_samples,
        seed=args.seed,
        output_path=args.output,
        mock=args.mock,
        verbose=args.verbose,
        k=args.k,
        config=config,
        rewrite_mode=args.rewrite_mode,
    )

    # Print summary
    print(f"\nFinal accuracy: {results['accuracy']:.4f} ({results['accuracy']:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
