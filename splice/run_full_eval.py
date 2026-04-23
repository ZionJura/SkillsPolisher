"""
run_full_eval.py — SPLICE Full Evaluation on All 89 SkillsBench Tasks

Runs all methods from Table 1 of the SPLICE paper across all 89 SkillsBench tasks.
Supports parallel evaluation via multiprocessing.

Methods evaluated:
  - baseline
  - random_kshot
  - bpo_only
  - case_only
  - splice_select   (= case_only, included for clarity)
  - splice_rewrite
  - splice_full

Results saved to results/full_eval_results.json.

Usage:
    python run_full_eval.py [--methods baseline case_only splice_full] [--n_workers 4]
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from demo_bank import DemoBank
from eval_runner import EvalRunner
from splice_loop import (
    SPLICEPipeline,
    VALID_METHODS,
    load_config,
    run_task_all_methods,
)
from splice_rewrite import SkillPromptRewriter

# ── Config ─────────────────────────────────────────────────────────────────

SPLICE_ROOT = Path(__file__).parent
SKILLSBENCH_ROOT = SPLICE_ROOT.parent / "related-works" / "skillsbench"
RESULTS_DIR = SPLICE_ROOT / "results"
DEMO_BANK_PATH = SPLICE_ROOT / "data" / "demo_bank.json"

# All methods from Table 1
ALL_TABLE1_METHODS = [
    "baseline",
    "random_kshot",
    "bpo_only",
    "case_only",
    "splice_select",
    "splice_rewrite",
    "splice_full",
]


# ── JSON serialization helper ────────────────────────────────────────────────

def _json_default(obj):
    """Handle non-serializable numpy types."""
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ── Task discovery ────────────────────────────────────────────────────────────

def get_all_task_ids(skillsbench_root: Path) -> List[str]:
    """Discover all task IDs from the SkillsBench tasks directory."""
    tasks_dir = skillsbench_root / "tasks"
    if not tasks_dir.is_dir():
        print(f"[warn] Tasks directory not found: {tasks_dir}")
        return []
    task_ids = sorted(
        d.name for d in tasks_dir.iterdir()
        if d.is_dir() and (d / "instruction.md").exists()
    )
    return task_ids


# ── Results aggregation ───────────────────────────────────────────────────────

def _compute_full_summary(
    results_by_method: Dict[str, List[Dict]],
    task_ids: List[str],
) -> Dict[str, Any]:
    """Compute detailed summary statistics for the full evaluation."""
    summary: Dict[str, Any] = {}

    for method, results in results_by_method.items():
        result_map = {r["task_id"]: r for r in results}
        rewards = []
        times = []
        errors = []

        for tid in task_ids:
            if tid in result_map:
                r = result_map[tid]
                rewards.append(r.get("reward", 0.0))
                times.append(r.get("elapsed_sec", 0.0))
                if r.get("error"):
                    errors.append(r["error"])
            else:
                rewards.append(0.0)
                errors.append("missing")

        n = len(rewards)
        mean = sum(rewards) / n if n > 0 else 0.0
        std = (sum((r - mean) ** 2 for r in rewards) / max(n - 1, 1)) ** 0.5 if n > 1 else 0.0

        summary[method] = {
            "n_tasks": n,
            "n_evaluated": len(results),
            "success_rate": round(mean, 4),
            "std": round(std, 4),
            "n_success": sum(1 for r in rewards if r > 0),
            "n_failure": sum(1 for r in rewards if r <= 0),
            "n_errors": len(errors),
            "mean_time_sec": round(sum(times) / max(len(times), 1), 2),
        }

    return summary


def _print_full_table(
    results_by_method: Dict[str, List[Dict]],
    methods: List[str],
) -> None:
    """Print Table 1 style comparison table."""
    print("\n" + "=" * 80)
    print("Table 1: SPLICE Full Evaluation Results on SkillsBench (89 tasks)")
    print("=" * 80)
    print(f"{'Method':<22} {'Success Rate':>15} {'Std':>8} {'N-eval':>8} {'Mean Time':>12}")
    print("-" * 80)

    for method in methods:
        results = results_by_method.get(method, [])
        if results:
            rewards = [r.get("reward", 0.0) for r in results]
            times = [r.get("elapsed_sec", 0.0) for r in results]
            n = len(rewards)
            mean = sum(rewards) / n if n > 0 else 0.0
            std = (sum((r - mean) ** 2 for r in rewards) / max(n - 1, 1)) ** 0.5 if n > 1 else 0.0
            mean_time = sum(times) / max(len(times), 1)
            print(
                f"{method:<22} {mean:>14.3f} {std:>8.3f} "
                f"{n:>8d} {mean_time:>11.1f}s"
            )
        else:
            print(f"{method:<22} {'N/A':>15}")
    print("=" * 80)


# ── Worker function for multiprocessing ───────────────────────────────────────

def _eval_task_worker(args_tuple) -> Dict[str, Any]:
    """
    Worker function for parallel task evaluation.
    Must be top-level for multiprocessing pickling.
    """
    (
        task_id, methods, config_dict, eval_mode, rewriter_mode,
        skillsbench_root, demo_bank_data, dry_run_seed, verbose
    ) = args_tuple

    try:
        # Reconstruct objects in worker process
        demo_bank = DemoBank(skillsbench_root=skillsbench_root)
        demo_bank.demos = demo_bank_data

        eval_runner = EvalRunner(
            mode=eval_mode,
            skillsbench_root=skillsbench_root,
            dry_run_seed=dry_run_seed,
        )

        needs_rewriter = any(
            m in ("bpo_only", "splice_rewrite", "splice_full") for m in methods
        )
        rewriter = None
        if needs_rewriter:
            try:
                rewriter = SkillPromptRewriter(mode=rewriter_mode)
            except Exception as e:
                if verbose:
                    print(f"  [warn] Rewriter init failed for {task_id}: {e}")
                methods = [m for m in methods if m not in ("bpo_only", "splice_rewrite", "splice_full")]
                if not methods:
                    return {"task_id": task_id, "results": {}, "error": str(e)}

        return run_task_all_methods(
            task_id=task_id,
            config=config_dict,
            methods=methods,
            demo_bank=demo_bank,
            eval_runner=eval_runner,
            rewriter=rewriter,
            verbose=verbose,
        )
    except Exception as e:
        return {
            "task_id": task_id,
            "results": {},
            "error": str(e),
        }


# ── Main evaluation function ──────────────────────────────────────────────────

def run_full_eval(
    eval_mode: str = "dry_run",
    rewriter_mode: str = "auto",
    methods: Optional[List[str]] = None,
    task_ids: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    demo_bank_path: Optional[str] = None,
    output_path: Optional[str] = None,
    n_workers: int = 1,
    verbose: bool = True,
    k: int = 3,
    delta: float = 0.05,
    max_bandit_rounds: int = 50,
    checkpoint_every: int = 10,
) -> Dict[str, Any]:
    """
    Run full evaluation on all SkillsBench tasks.

    Args:
        eval_mode: Evaluation backend.
        rewriter_mode: Prompt rewriter backend.
        methods: Methods to evaluate (default: all Table 1 methods).
        task_ids: Task IDs to evaluate (default: all discovered tasks).
        config_path: Config JSON path.
        demo_bank_path: Demo bank JSON path.
        output_path: Results output path.
        n_workers: Parallel workers (1 = sequential).
        verbose: Verbose output.
        k: Demos per task.
        delta: Bandit delta.
        max_bandit_rounds: Max bandit rounds.
        checkpoint_every: Save intermediate results every N tasks.

    Returns:
        Full results dict.
    """
    methods = methods or ALL_TABLE1_METHODS
    output_path = output_path or str(RESULTS_DIR / "full_eval_results.json")

    # Validate methods
    invalid = [m for m in methods if m not in VALID_METHODS]
    if invalid:
        raise ValueError(f"Invalid methods: {invalid}. Valid: {VALID_METHODS}")

    print("=" * 70)
    print("SPLICE Full Evaluation")
    print("=" * 70)
    print(f"Methods:   {methods}")
    print(f"Eval mode: {eval_mode}")
    print(f"Rewriter:  {rewriter_mode}")
    print(f"Workers:   {n_workers}")
    print("=" * 70)

    t0_total = time.time()

    # ── Load config ────────────────────────────────────────────────────────────
    config = load_config(config_path)
    config["skillsbench_root"] = str(SKILLSBENCH_ROOT)
    config.setdefault("bandit", {}).update({
        "k_selected": k,
        "delta": delta,
        "max_rounds": max_bandit_rounds,
        "n_demo_candidates": 20,
    })

    # ── Discover tasks ──────────────────────────────────────────────────────────
    if task_ids is None:
        task_ids = get_all_task_ids(SKILLSBENCH_ROOT)
        if not task_ids:
            print("[error] No tasks found in SkillsBench. Check skillsbench_root.")
            return {}
    print(f"Tasks:     {len(task_ids)} tasks found")

    # ── Build/load demo bank ───────────────────────────────────────────────────
    bank_path = demo_bank_path or str(DEMO_BANK_PATH)
    if Path(bank_path).exists():
        print(f"\nLoading demo bank from {bank_path}...")
        demo_bank = DemoBank.load(bank_path, skillsbench_root=str(SKILLSBENCH_ROOT))
    else:
        print("\nBuilding demo bank...")
        demo_bank = DemoBank(skillsbench_root=str(SKILLSBENCH_ROOT))
        demo_bank.build_from_tasks(max_tasks=200, verbose=False)
        DEMO_BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
        demo_bank.save(bank_path)
    print(f"Demo bank: {demo_bank}")

    # ── Initialize singletons (for sequential mode) ────────────────────────────
    eval_runner = EvalRunner(
        mode=eval_mode,
        skillsbench_root=str(SKILLSBENCH_ROOT),
        dry_run_seed=42,
    )
    needs_rewriter = any(m in ("bpo_only", "splice_rewrite", "splice_full") for m in methods)
    rewriter = None
    if needs_rewriter and n_workers == 1:
        print(f"\nInitializing rewriter (mode={rewriter_mode})...")
        try:
            rewriter = SkillPromptRewriter(mode=rewriter_mode)
        except Exception as e:
            print(f"[warn] Rewriter unavailable: {e}")
            methods = [m for m in methods if m not in ("bpo_only", "splice_rewrite", "splice_full")]
            print(f"[warn] Removed BPO methods. Running: {methods}")

    # ── Run evaluation ─────────────────────────────────────────────────────────
    all_task_results: List[Dict[str, Any]] = []
    results_by_method: Dict[str, List[Dict]] = {m: [] for m in methods}

    print(f"\nStarting evaluation of {len(task_ids)} tasks...")

    if n_workers > 1:
        # Parallel evaluation using thread pool (avoids pickling issues with models)
        print(f"Running with {n_workers} parallel workers (thread pool)...")
        demo_bank_data = demo_bank.demos  # serializable

        worker_args = [
            (
                task_id, methods, config,
                eval_mode, rewriter_mode,
                str(SKILLSBENCH_ROOT), demo_bank_data,
                42 + idx,  # different seed per worker for dry_run
                False,  # verbose=False in workers
            )
            for idx, task_id in enumerate(task_ids)
        ]

        completed = 0
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {
                executor.submit(_eval_task_worker, args): args[0]
                for args in worker_args
            }
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    task_result = future.result(timeout=120)
                except Exception as e:
                    task_result = {
                        "task_id": task_id,
                        "results": {},
                        "error": str(e),
                    }

                all_task_results.append(task_result)
                for method, result in task_result.get("results", {}).items():
                    if method in results_by_method:
                        results_by_method[method].append(result)

                completed += 1
                if verbose or completed % 5 == 0:
                    rewards_str = " | ".join(
                        f"{m}={task_result['results'].get(m, {}).get('reward', '?'):.1f}"
                        for m in methods[:3]
                    )
                    print(f"  [{completed:3d}/{len(task_ids)}] {task_id}: {rewards_str}")

                # Checkpoint
                if completed % checkpoint_every == 0:
                    _save_checkpoint(
                        all_task_results, results_by_method, methods, task_ids, output_path
                    )
    else:
        # Sequential evaluation
        for task_num, task_id in enumerate(task_ids):
            if verbose:
                print(f"\n[{task_num+1:3d}/{len(task_ids)}] {task_id}")

            try:
                task_result = run_task_all_methods(
                    task_id=task_id,
                    config=config,
                    methods=methods,
                    demo_bank=demo_bank,
                    eval_runner=eval_runner,
                    rewriter=rewriter,
                    verbose=verbose,
                )
            except Exception as e:
                print(f"  [error] {task_id}: {e}")
                task_result = {
                    "task_id": task_id,
                    "results": {},
                    "error": str(e),
                }

            all_task_results.append(task_result)
            for method, result in task_result.get("results", {}).items():
                if method in results_by_method:
                    results_by_method[method].append(result)

            if (task_num + 1) % checkpoint_every == 0:
                _save_checkpoint(
                    all_task_results, results_by_method, methods, task_ids, output_path
                )

    # ── Print results table ───────────────────────────────────────────────────
    _print_full_table(results_by_method, methods)

    # ── Compute summary ────────────────────────────────────────────────────────
    total_time = time.time() - t0_total
    summary = _compute_full_summary(results_by_method, task_ids)
    print(f"\nTotal evaluation time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "experiment": "splice_full_eval",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "eval_mode": eval_mode,
            "rewriter_mode": rewriter_mode,
            "methods": methods,
            "n_tasks": len(task_ids),
            "task_ids": task_ids,
            "k": k,
            "delta": delta,
            "max_bandit_rounds": max_bandit_rounds,
            "n_workers": n_workers,
        },
        "total_time_sec": round(total_time, 2),
        "summary": summary,
        "results_by_method": {
            method: [
                {
                    "task_id": r["task_id"],
                    "reward": r.get("reward", 0.0),
                    "elapsed_sec": r.get("elapsed_sec", 0.0),
                    "error": r.get("error", ""),
                    "n_demos": r.get("n_demos_selected", 0),
                    "was_rewritten": r.get("was_rewritten", False),
                }
                for r in results
            ]
            for method, results in results_by_method.items()
        },
        "full_task_results": all_task_results,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\nFull results saved to {out_path}")

    return output


def _save_checkpoint(
    all_task_results: List[Dict],
    results_by_method: Dict[str, List[Dict]],
    methods: List[str],
    task_ids: List[str],
    output_path: str,
) -> None:
    """Save intermediate checkpoint."""
    checkpoint_path = Path(output_path).with_suffix(".checkpoint.json")
    try:
        checkpoint = {
            "checkpoint": True,
            "n_completed": len(all_task_results),
            "n_total": len(task_ids),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results_by_method": {
                method: [
                    {"task_id": r["task_id"], "reward": r.get("reward", 0.0)}
                    for r in results
                ]
                for method, results in results_by_method.items()
            },
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=_json_default)
        print(f"  [checkpoint] Saved {len(all_task_results)} results to {checkpoint_path}")
    except Exception as e:
        print(f"  [warn] Checkpoint failed: {e}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SPLICE Full Evaluation on all 89 SkillsBench tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval_mode",
        choices=["dry_run", "openai", "bench_cli", "auto"],
        default="dry_run",
    )
    parser.add_argument(
        "--rewriter_mode",
        choices=["auto", "hf", "openai", "claude"],
        default="auto",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=VALID_METHODS,
        default=None,
        help="Methods to run (default: all Table 1 methods)",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task IDs to evaluate (default: all discovered tasks)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--demo_bank",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "full_eval_results.json"),
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of parallel evaluation workers",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--max_bandit_rounds",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
        help="Save checkpoint every N tasks",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
    )
    args = parser.parse_args()

    run_full_eval(
        eval_mode=args.eval_mode,
        rewriter_mode=args.rewriter_mode,
        methods=args.methods,
        task_ids=args.tasks,
        config_path=args.config,
        demo_bank_path=args.demo_bank,
        output_path=args.output,
        n_workers=args.n_workers,
        verbose=args.verbose,
        k=args.k,
        delta=args.delta,
        max_bandit_rounds=args.max_bandit_rounds,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
