"""
run_compare.py — Run all baselines on a dataset and print a comparison table.

Usage:
    python eval_pipeline/run_compare.py --dataset gsm8k --split test --n_samples 10 --mock

Run from project root.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

# Ensure project root is in path
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from eval_pipeline.run_eval import (
    load_config,
    build_llm_client,
    build_dataset,
    build_train_dataset,
    build_baseline,
    build_generation_config,
    build_evaluator,
    normalize_results_layout,
    apply_model_backend_defaults,
)

ALL_BASELINES = ["zero_shot", "random_kshot", "case_bandit", "bpo_rewrite"]


def _sanitize_model_name(model_name: str) -> str:
    safe = []
    for ch in model_name:
        if ch.isalnum() or ch in {".", "_", "-"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "unknown_model"


def _resolve_output_path(output_path: str, config: dict, model_name: str, dataset_name: str) -> Path:
    out_path = Path(output_path)
    if not out_path.is_absolute():
        results_dir = _PROJECT_ROOT / "results" / _sanitize_model_name(model_name) / dataset_name
        out_path = results_dir / output_path
    return out_path


def _build_compare_output(
    dataset_name: str,
    split: str,
    seed: int,
    mock: bool,
    compare_results: list,
    baselines: list,
    status: str = "completed",
) -> dict:
    completed = [r for r in compare_results if r.get("status") == "completed"]
    summary = {}
    for result in compare_results:
        summary[result["baseline"]] = {
            "status": result.get("status", "pending"),
            "accuracy": result.get("accuracy", 0.0),
            "avg_score": result.get("avg_score", 0.0),
            "completed_samples": result.get("completed_samples", result.get("n_samples", 0)),
            "total_samples": result.get("n_samples", 0),
        }

    return {
        "dataset": dataset_name,
        "split": split,
        "n_samples": max((r.get("n_samples", 0) for r in compare_results), default=0),
        "seed": seed,
        "mock": mock,
        "status": status,
        "completed_baselines": len(completed),
        "total_baselines": len(baselines),
        "baselines": compare_results,
        "summary": summary,
    }


def _build_experiment_metadata(
    dataset_name: str,
    split: str,
    requested_n_samples: int,
    seed: int,
    k: int,
    baselines: list,
    mock: bool,
    rewrite_mode: str,
    model_name: str,
    backend: str,
) -> dict:
    params = {
        "dataset": dataset_name,
        "split": split,
        "requested_n_samples": requested_n_samples,
        "seed": seed,
        "k": k,
        "baselines": list(baselines),
        "mock": mock,
        "rewrite_mode": rewrite_mode,
        "model_name": model_name,
        "backend": backend,
    }
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False)
    experiment_id = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return {
        "experiment_id": experiment_id,
        "params": params,
    }


def _cost_delta(before: dict, after: dict) -> dict:
    return {
        "total_tokens": after.get("total_tokens", 0) - before.get("total_tokens", 0),
        "prompt_tokens": after.get("prompt_tokens", 0) - before.get("prompt_tokens", 0),
        "completion_tokens": after.get("completion_tokens", 0) - before.get("completion_tokens", 0),
        "n_calls": after.get("n_calls", 0) - before.get("n_calls", 0),
    }


def _save_output_json(output: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=out_path.parent,
        prefix=f".{out_path.stem}_",
        suffix=".tmp",
        delete=False,
    ) as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        tmp_path = Path(f.name)
    tmp_path.replace(out_path)


def _baseline_output_path(base_dir: Path, baseline_name: str) -> Path:
    return base_dir / f"{baseline_name}_result.json"


def _load_existing_output(out_path: Path) -> Optional[dict]:
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_same_completed_experiment(existing: dict, experiment_meta: dict, baselines: list) -> bool:
    if not existing or existing.get("status") != "completed":
        return False
    existing_meta = existing.get("experiment", {})
    if existing_meta.get("experiment_id") != experiment_meta.get("experiment_id"):
        return False
    completed = existing.get("completed_baselines", 0)
    return completed == len(baselines)


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _update_statistics_csv(
    output: dict,
    out_path: Path,
    experiment_meta: dict,
    llm_client,
    timestamp: str,
) -> Path:
    stats_dir = _PROJECT_ROOT / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    csv_path = stats_dir / "results.csv"

    params = experiment_meta["params"]
    fieldnames = [
        "timestamp",
        "experiment_id",
        "dataset",
        "split",
        "baseline",
        "model_name",
        "backend",
        "mock",
        "requested_n_samples",
        "actual_n_samples",
        "seed",
        "k",
        "rewrite_mode",
        "accuracy",
        "avg_score",
        "avg_used_skills",
        "n_correct",
        "total_tokens",
        "avg_tokens_per_sample",
        "prompt_tokens",
        "completion_tokens",
        "n_calls",
        "avg_calls_per_sample",
        "output_json",
        "baselines_compared",
    ]

    rows = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    key = experiment_meta["experiment_id"]
    remaining = [
        row for row in rows
        if not (row.get("experiment_id") == key and row.get("baseline") in {b["baseline"] for b in output["baselines"]})
    ]

    for baseline_result in output["baselines"]:
        sample_records = baseline_result.get("results", [])
        skill_counts = [len(r.get("used_skills", [])) for r in sample_records]
        avg_used_skills = (sum(skill_counts) / len(skill_counts)) if skill_counts else 0.0
        n_samples = baseline_result.get("n_samples", 0) or 0
        total_tokens = baseline_result.get("total_tokens", 0) or 0
        n_calls = baseline_result.get("n_calls", 0) or 0
        prompt_tokens = sum(r.get("prompt_tokens", 0) for r in sample_records)
        completion_tokens = sum(r.get("completion_tokens", 0) for r in sample_records)
        n_correct = sum(1 for r in sample_records if r.get("correct"))

        remaining.append({
            "timestamp": timestamp,
            "experiment_id": key,
            "dataset": output["dataset"],
            "split": output["split"],
            "baseline": baseline_result["baseline"],
            "model_name": llm_client.model,
            "backend": llm_client.backend,
            "mock": str(output["mock"]),
            "requested_n_samples": str(params["requested_n_samples"]),
            "actual_n_samples": str(n_samples),
            "seed": str(params["seed"]),
            "k": str(params["k"]),
            "rewrite_mode": params["rewrite_mode"],
            "accuracy": _format_float(baseline_result.get("accuracy", 0.0)),
            "avg_score": _format_float(baseline_result.get("avg_score", 0.0)),
            "avg_used_skills": _format_float(avg_used_skills),
            "n_correct": str(n_correct),
            "total_tokens": str(total_tokens),
            "avg_tokens_per_sample": _format_float(total_tokens / n_samples if n_samples else 0.0),
            "prompt_tokens": str(prompt_tokens),
            "completion_tokens": str(completion_tokens),
            "n_calls": str(n_calls),
            "avg_calls_per_sample": _format_float(n_calls / n_samples if n_samples else 0.0),
            "output_json": str(_baseline_output_path(out_path.parent, baseline_result["baseline"])),
            "baselines_compared": ",".join(params["baselines"]),
        })

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(remaining)

    return csv_path


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
    generation_config: Optional[dict] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Run a single baseline and return results summary."""
    print(f"\n  Running {baseline_name}...")

    # Build baseline
    try:
        generation_config = generation_config or {"max_tokens": 2048, "temperature": 0.0}
        baseline = build_baseline(
            baseline_name,
            llm_client=llm_client,
            dataset_name=dataset_name,
            k=k,
            seed=seed,
            train_samples=list(train_samples) if train_samples else [],
            rewrite_mode=rewrite_mode,
            max_tokens=generation_config["max_tokens"],
            temperature=generation_config["temperature"],
        )
    except Exception as e:
        print(f"  [warn] Could not build baseline {baseline_name}: {e}")
        return {
            "baseline": baseline_name,
            "status": "failed",
            "error": str(e),
            "n_samples": 0,
            "completed_samples": 0,
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
    n = len(samples)

    result = {
        "baseline": baseline_name,
        "status": "running",
        "n_samples": n,
        "completed_samples": 0,
        "accuracy": 0.0,
        "avg_score": 0.0,
        "results": result_records,
        "n_calls": 0,
        "total_tokens": 0,
        "elapsed_s": 0.0,
    }

    if progress_callback is not None:
        progress_callback(result)

    iterator = samples
    progress = None
    if tqdm is not None and n > 0:
        progress = tqdm(
            samples,
            total=n,
            desc=f"{baseline_name:>12}",
            unit="sample",
            leave=True,
        )
        iterator = progress

    for i, sample in enumerate(iterator):
        if verbose and (i % 20 == 0):
            print(f"    [{i+1}/{len(samples)}]")

        before_cost = llm_client.cost_tracker.to_dict()

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

        trace = baseline.get_last_trace()

        result_records.append({
            "id": sample.id,
            "question": sample.question,
            "prediction": prediction or "",
            "answer": sample.answer or "",
            "correct": correct,
            "score": score,
            "used_skills": trace.get("used_skills", []),
            "demo_ids": trace.get("demo_ids", []),
            "trace": trace,
            "task_name": sample.metadata.get("task_name", ""),
            "details": score_result.get("details", ""),
        })

        after_cost = llm_client.cost_tracker.to_dict()
        token_stats = _cost_delta(before_cost, after_cost)
        result_records[-1].update(token_stats)

        elapsed = time.time() - start
        completed = i + 1
        accuracy_so_far = n_correct / completed if completed > 0 else 0.0
        avg_score_so_far = total_score / completed if completed > 0 else 0.0
        cost = llm_client.cost_tracker.to_dict()

        result.update({
            "status": "running",
            "completed_samples": completed,
            "accuracy": accuracy_so_far,
            "avg_score": avg_score_so_far,
            "n_calls": cost["n_calls"],
            "total_tokens": cost["total_tokens"],
            "elapsed_s": elapsed,
        })

        if progress is not None:
            progress.set_postfix(
                acc=f"{accuracy_so_far:.1%}",
                calls=cost["n_calls"],
                tokens=cost["total_tokens"],
            )

        if progress_callback is not None:
            progress_callback(result)

    if progress is not None:
        progress.close()

    elapsed = time.time() - start
    accuracy = n_correct / n if n > 0 else 0.0
    avg_score = total_score / n if n > 0 else 0.0
    cost = llm_client.cost_tracker.to_dict()

    print(
        f"    Done: {n_correct}/{n} correct ({accuracy:.1%}), "
        f"time={elapsed:.1f}s, tokens={cost['total_tokens']}"
    )

    result.update({
        "status": "completed",
        "completed_samples": n,
        "accuracy": accuracy,
        "avg_score": avg_score,
        "n_calls": cost["n_calls"],
        "total_tokens": cost["total_tokens"],
        "elapsed_s": elapsed,
    })
    if progress_callback is not None:
        progress_callback(result)
    return result


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
    rerun: bool = False,
) -> dict:
    """Run all baselines and produce comparison."""
    print(f"\n{'='*60}")
    print(f"Comparison: {dataset_name} ({split})")
    print(f"Baselines: {', '.join(baselines)}")
    print(f"Samples: {n_samples if n_samples > 0 else 'all'}, Mock: {mock}")
    print(f"{'='*60}")

    # Build shared components
    llm_client = build_llm_client(config, mock=mock)
    migrated = normalize_results_layout(llm_client.model)
    if migrated:
        print(f"Migrated {len(migrated)} legacy result file(s) into dataset directories")
    print(f"LLM: {llm_client}")
    experiment_meta = _build_experiment_metadata(
        dataset_name=dataset_name,
        split=split,
        requested_n_samples=n_samples,
        seed=seed,
        k=k,
        baselines=baselines,
        mock=mock,
        rewrite_mode=rewrite_mode,
        model_name=llm_client.model,
        backend=llm_client.backend,
    )
    out_path = _resolve_output_path(output_path, config, llm_client.model, dataset_name) if output_path else None
    if out_path is not None:
        print(f"Output: {out_path}")
        existing = _load_existing_output(out_path)
        if not rerun and _is_same_completed_experiment(existing, experiment_meta, baselines):
            print(f"[skip] Experiment already completed: {experiment_meta['experiment_id']}")
            print(f"[skip] Reusing existing results at: {out_path}")
            return existing

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
    generation_config = build_generation_config(dataset_name, config)

    # Run each baseline
    compare_results = []

    def save_partial() -> None:
        if out_path is None:
            return
        output = _build_compare_output(
            dataset_name=dataset_name,
            split=split,
            seed=seed,
            mock=mock,
            compare_results=compare_results,
            baselines=baselines,
            status="running",
        )
        output["experiment"] = experiment_meta
        _save_output_json(output, out_path)
        for baseline_result in compare_results:
            baseline_path = _baseline_output_path(out_path.parent, baseline_result["baseline"])
            baseline_output = {
                "dataset": dataset_name,
                "split": split,
                "baseline": baseline_result["baseline"],
                "status": baseline_result.get("status", "pending"),
                "n_samples": baseline_result.get("n_samples", 0),
                "completed_samples": baseline_result.get("completed_samples", 0),
                "accuracy": baseline_result.get("accuracy", 0.0),
                "avg_score": baseline_result.get("avg_score", 0.0),
                "results": baseline_result.get("results", []),
                "n_calls": baseline_result.get("n_calls", 0),
                "total_tokens": baseline_result.get("total_tokens", 0),
                "elapsed_s": baseline_result.get("elapsed_s", 0.0),
                "mock": mock,
                "experiment": experiment_meta,
            }
            _save_output_json(baseline_output, baseline_path)

    for baseline_name in baselines:
        partial_result = {
            "baseline": baseline_name,
            "status": "pending",
            "n_samples": len(samples),
            "completed_samples": 0,
            "accuracy": 0.0,
            "avg_score": 0.0,
            "results": [],
            "n_calls": 0,
            "total_tokens": 0,
            "elapsed_s": 0.0,
        }
        compare_results.append(partial_result)
        save_partial()

        def _progress_callback(updated_result: dict, baseline_name: str = baseline_name) -> None:
            for idx, existing in enumerate(compare_results):
                if existing["baseline"] == baseline_name:
                    compare_results[idx] = dict(updated_result)
                    break
            save_partial()

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
            generation_config=generation_config,
            progress_callback=_progress_callback,
        )
        for idx, existing in enumerate(compare_results):
            if existing["baseline"] == baseline_name:
                compare_results[idx] = result
                break
        save_partial()

    # Print table
    print(f"\n\n{'='*60}")
    print(f"COMPARISON: {dataset_name.upper()} ({split}), n={len(samples)}")
    print(f"{'='*60}")
    print_comparison_table(compare_results)

    # Build output
    output = _build_compare_output(
        dataset_name=dataset_name,
        split=split,
        seed=seed,
        mock=mock,
        compare_results=compare_results,
        baselines=baselines,
        status="completed",
    )
    output["experiment"] = experiment_meta

    # Save output
    if out_path is not None:
        _save_output_json(output, out_path)
        print(f"\nComparison saved to: {out_path}")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        csv_path = _update_statistics_csv(
            output=output,
            out_path=out_path,
            experiment_meta=experiment_meta,
            llm_client=llm_client,
            timestamp=timestamp,
        )
        print(f"Statistics updated: {csv_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Compare all baselines on a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all baselines on GSM8K (mock mode)
  python eval_pipeline/run_compare.py --dataset gsm8k --split test --n_samples 10 --mock

  # Compare specific baselines
  python eval_pipeline/run_compare.py --dataset aqua_rat --split dev \\
      --baselines zero_shot random_kshot --n_samples 10 --mock
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
        default=10,
        help="Number of samples (default: 10)",
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
        help="Output JSON path (default: ./results/{model_name}/{dataset_name}/compare_result.json)",
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
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name to use. Special rule: 'gpt-4-o' routes to ByteDance API; all others route to zhizengzeng unless --backend overrides.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Force re-run even if the same experiment already completed",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = "compare_result.json"

    config = load_config(args.config)
    config = apply_model_backend_defaults(config, args.model_name, args.backend)

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
        rerun=args.rerun,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
