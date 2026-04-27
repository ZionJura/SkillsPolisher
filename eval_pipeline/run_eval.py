"""
run_eval.py — Main evaluation script for the SPLICE eval pipeline.

Usage:
    python eval_pipeline/run_eval.py \
        --dataset gsm8k \
        --split test \
        --baseline zero_shot \
        --n_samples 10 \
        --seed 42 \
        --output results/gsm8k_zero_shot.json \
        --mock

Run from project root.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in path
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _sanitize_model_name(model_name: str) -> str:
    safe = []
    for ch in model_name:
        if ch.isalnum() or ch in {".", "_", "-"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "unknown_model"


_KNOWN_BASELINES = ("zero_shot", "random_kshot", "case_bandit", "bpo_rewrite")


def _results_dataset_dir(model_name: str, dataset_name: str) -> Path:
    return _PROJECT_ROOT / "results" / _sanitize_model_name(model_name) / dataset_name


def _resolve_output_path(output_path: str, model_name: str, dataset_name: str) -> Path:
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = _results_dataset_dir(model_name, dataset_name) / output_path
    return out_path


def _parse_legacy_result_filename(path: Path) -> Optional[tuple[str, str]]:
    if path.suffix != ".json":
        return None

    stem = path.stem
    for baseline_name in _KNOWN_BASELINES:
        suffix = f"_{baseline_name}_result"
        if stem.endswith(suffix):
            dataset_name = stem[:-len(suffix)]
            if dataset_name:
                return dataset_name, f"{baseline_name}_result.json"

    if stem.endswith("_result"):
        dataset_name = stem[:-len("_result")]
        if dataset_name:
            return dataset_name, "compare_result.json"

    return None


def _read_json_safely(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _result_priority(path: Path) -> tuple[int, int, int, float]:
    data = _read_json_safely(path) or {}
    status_rank = {"completed": 2, "running": 1, "pending": 0}
    completed = data.get("completed_samples", data.get("completed_baselines", 0)) or 0
    total = data.get("n_samples", data.get("total_baselines", 0)) or 0
    return (
        status_rank.get(data.get("status", ""), 0),
        int(completed),
        int(total),
        path.stat().st_mtime,
    )


def normalize_results_layout(model_name: str) -> list[tuple[Path, Path]]:
    model_dir = _PROJECT_ROOT / "results" / _sanitize_model_name(model_name)
    if not model_dir.exists():
        return []

    moved = []
    for path in model_dir.iterdir():
        if not path.is_file():
            continue

        parsed = _parse_legacy_result_filename(path)
        if parsed is None:
            continue

        dataset_name, new_name = parsed
        destination = _results_dataset_dir(model_name, dataset_name) / new_name
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            if _result_priority(path) > _result_priority(destination):
                path.replace(destination)
                moved.append((path, destination))
            else:
                path.unlink()
            continue

        path.replace(destination)
        moved.append((path, destination))

    return moved


def load_config(config_path: str = None) -> dict:
    """Load eval configuration."""
    if config_path is None:
        config_path = str(_SCRIPT_DIR / "configs" / "eval_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        # Normalize legacy absolute paths from the original Windows checkout.
        legacy_root = "/mnt/d/Code/AI4R/Skills-Learning2"
        replacements = {
            "data_root": str(_PROJECT_ROOT / "related-works"),
            "splice_root": str(_PROJECT_ROOT / "splice"),
            "results_dir": str(_PROJECT_ROOT / "results"),
        }
        for key, replacement in replacements.items():
            value = config.get(key)
            if not value or str(value).startswith(legacy_root):
                config[key] = replacement
        return config
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
    from eval_pipeline.datasets import load_dataset
    from eval_pipeline.datasets.demo_bank import build_skillsbench_demo_pool
    if dataset_name == "skillsbench":
        return build_skillsbench_demo_pool()
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
    max_tokens: int = 2048,
    temperature: float = 0.0,
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
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif baseline_name == "random_kshot":
        return RandomKShotBaseline(
            llm_client=llm_client,
            k=k,
            seed=seed,
            dataset_name=dataset_name,
            train_samples=list(train_samples) if train_samples else [],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif baseline_name == "case_bandit":
        return CASEBanditBaseline(
            llm_client=llm_client,
            k=k,
            dataset_name=dataset_name,
            train_samples=list(train_samples) if train_samples else [],
            seed=seed,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif baseline_name == "bpo_rewrite":
        return BPORewriteBaseline(
            llm_client=llm_client,
            rewrite_mode=rewrite_mode,
            dataset_name=dataset_name,
            train_samples=list(train_samples) if train_samples else [],
            k_demos=k,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unknown baseline: {baseline_name!r}. "
            f"Available: zero_shot, random_kshot, case_bandit, bpo_rewrite"
        )


def build_generation_config(dataset_name: str, config: dict) -> dict:
    """Build per-dataset generation settings for baseline predictions."""
    llm_cfg = config.get("llm", config)
    max_tokens = int(llm_cfg.get("max_tokens", 2048))
    temperature = float(llm_cfg.get("temperature", 0.0))

    # SkillsBench tasks often require long executable scripts.
    if dataset_name == "skillsbench":
        max_tokens = max(max_tokens, 12048)

    return {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def apply_model_backend_defaults(config: dict, model_name: Optional[str], backend: Optional[str]) -> dict:
    """Apply CLI model/backend routing with explicit defaults."""
    llm_cfg = config.setdefault("llm", {})

    if model_name:
        llm_cfg["model"] = model_name
        os.environ["LLM_MODEL"] = model_name
        if backend is None:
            inferred_backend = "azure" if model_name.strip().lower() == "gpt-4-o" else "openai"
            llm_cfg["backend"] = inferred_backend
            os.environ["LLM_BACKEND"] = inferred_backend

    if backend:
        llm_cfg["backend"] = backend
        os.environ["LLM_BACKEND"] = backend

    return config


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
    migrated = normalize_results_layout(llm_client.model)
    if migrated:
        print(f"Migrated {len(migrated)} legacy result file(s) into dataset directories")
    print(f"LLM client: {llm_client}")
    resolved_output_path = _resolve_output_path(output_path, llm_client.model, dataset_name) if output_path else None
    if resolved_output_path is not None:
        print(f"Output path: {resolved_output_path}")

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
    generation_config = build_generation_config(dataset_name, config)
    print(f"Building baseline: {baseline_name}...")
    baseline = build_baseline(
        baseline_name,
        llm_client=llm_client,
        dataset_name=dataset_name,
        k=k,
        seed=seed,
        train_samples=train_samples,
        rewrite_mode=rewrite_mode,
        max_tokens=generation_config["max_tokens"],
        temperature=generation_config["temperature"],
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
    if resolved_output_path:
        out_path = resolved_output_path
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
  # Zero-shot on GSM8K test (first 10 samples), mock mode
  python eval_pipeline/run_eval.py --dataset gsm8k --split test --baseline zero_shot \\
      --n_samples 10 --mock

  # Random k-shot on AQUA-RAT dev
  python eval_pipeline/run_eval.py --dataset aqua_rat --split dev \\
      --baseline random_kshot --k 3 --n_samples 10

  # CASE bandit on StrategyQA
  python eval_pipeline/run_eval.py --dataset strategyqa --split test \\
      --baseline case_bandit --k 3 --n_samples 10 --mock

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
        default=10,
        help="Number of samples to evaluate (0 = all, default: 10)",
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
        help="Output JSON file path (default: ./results/{model_name}/{dataset_name}/{baseline}_result.json)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM client (no API calls, for offline testing)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["openai", "azure"],
        help="LLM backend: 'openai' (zhizengzeng, default) or 'azure' (ByteDance internal, requires VPN)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name to use. Special rule: 'gpt-4-o' routes to ByteDance API; all others route to zhizengzeng unless --backend overrides.",
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
        args.output = f"{args.baseline}_result.json"

    # Load config
    config = load_config(args.config)
    config = apply_model_backend_defaults(config, args.model_name, args.backend)

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
