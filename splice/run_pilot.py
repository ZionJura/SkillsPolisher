"""
run_pilot.py — SPLICE Pilot Experiment

Evaluates 10 SkillsBench tasks and compares:
  - baseline:    No demos, original skill prompt
  - case_only:   CASE bandit-selected demos, original skill prompt
  - splice_full: CASE demos + BPO-rewritten skill prompt

Designed to run in <1 hour on a single GPU (A100 40GB).
Results saved to results/pilot_results.json.

Usage:
    python run_pilot.py [--eval_mode dry_run|openai|bench_cli] [--rewriter_mode auto|hf|openai]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure splice/ is on path
sys.path.insert(0, str(Path(__file__).parent))

from demo_bank import DemoBank
from eval_runner import EvalRunner
from splice_loop import SPLICEPipeline, run_task_all_methods, load_config
from splice_rewrite import SkillPromptRewriter

# ── Config ────────────────────────────────────────────────────────────────────

SPLICE_ROOT = Path(__file__).parent
SKILLSBENCH_ROOT = SPLICE_ROOT.parent / "related-works" / "skillsbench"
RESULTS_DIR = SPLICE_ROOT / "results"
DEMO_BANK_PATH = SPLICE_ROOT / "data" / "demo_bank.json"

# 10 pilot tasks (mix of single-skill and multi-skill)
PILOT_TASK_IDS = [
    "3d-scan-calc",
    "adaptive-cruise-control",
    "citation-check",
    "civ6-adjacency-optimizer",
    "court-form-filling",
    "data-to-d3",
    "dialogue-parser",
    "earthquake-phase-association",
    "earthquake-plate-calculation",
    "econ-detrending-correlation",
]

PILOT_METHODS = ["baseline", "case_only", "splice_full"]


# ── JSON serialization helper ────────────────────────────────────────────────

def _json_default(obj):
    """Handle non-serializable types (numpy int/float/bool)."""
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


# ── Results formatting ────────────────────────────────────────────────────────

def _format_table(results_by_method: Dict[str, List[Dict]]) -> str:
    """Format results as a markdown-style table."""
    task_ids = sorted(set(
        r["task_id"]
        for method_results in results_by_method.values()
        for r in method_results
    ))
    methods = list(results_by_method.keys())

    # Build header
    col_w = 30
    method_w = 12
    header = f"{'Task':<{col_w}}" + "".join(f"{m:>{method_w}}" for m in methods)
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for tid in task_ids:
        row = f"{tid:<{col_w}}"
        for method in methods:
            method_results = {r["task_id"]: r for r in results_by_method.get(method, [])}
            if tid in method_results:
                reward = method_results[tid]["reward"]
                row += f"{reward:>{method_w}.2f}"
            else:
                row += f"{'N/A':>{method_w}}"
        lines.append(row)

    lines.append(sep)

    # Summary row
    summary = f"{'Mean':.<{col_w}}"
    for method in methods:
        scores = [r["reward"] for r in results_by_method.get(method, []) if "reward" in r]
        if scores:
            mean_score = sum(scores) / len(scores)
            summary += f"{mean_score:>{method_w}.3f}"
        else:
            summary += f"{'N/A':>{method_w}}"
    lines.append(summary)
    lines.append(sep)
    return "\n".join(lines)


def _compute_summary(results_by_method: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Compute summary statistics per method."""
    summary = {}
    for method, results in results_by_method.items():
        rewards = [r["reward"] for r in results if "reward" in r]
        if rewards:
            n = len(rewards)
            mean = sum(rewards) / n
            std = (sum((r - mean) ** 2 for r in rewards) / max(n - 1, 1)) ** 0.5
            summary[method] = {
                "n_tasks": n,
                "success_rate": round(mean, 4),
                "std": round(std, 4),
                "n_success": sum(1 for r in rewards if r > 0),
                "n_failure": sum(1 for r in rewards if r <= 0),
            }
        else:
            summary[method] = {"n_tasks": 0, "success_rate": 0.0}
    return summary


# ── Main pilot function ───────────────────────────────────────────────────────

def run_pilot(
    eval_mode: str = "dry_run",
    rewriter_mode: str = "auto",
    task_ids: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    demo_bank_path: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
    k: int = 3,
    delta: float = 0.05,
    max_bandit_rounds: int = 30,
) -> Dict[str, Any]:
    """
    Run the SPLICE pilot experiment.

    Args:
        eval_mode: Evaluation backend ("dry_run", "openai", "bench_cli").
        rewriter_mode: Prompt rewriter backend ("auto", "hf", "openai", "claude").
        task_ids: List of task IDs to evaluate. Defaults to PILOT_TASK_IDS.
        methods: Methods to compare. Defaults to PILOT_METHODS.
        config_path: Path to splice_default.json.
        demo_bank_path: Path to pre-built demo_bank.json.
        output_path: Where to save results JSON.
        verbose: Print progress.
        k: Number of demonstrations to select per task.
        delta: CASE bandit delta parameter.
        max_bandit_rounds: Maximum bandit rounds per task.

    Returns:
        Full results dict with per-task results and summary statistics.
    """
    task_ids = task_ids or PILOT_TASK_IDS
    methods = methods or PILOT_METHODS
    output_path = output_path or str(RESULTS_DIR / "pilot_results.json")

    print("=" * 60)
    print("SPLICE Pilot Experiment")
    print("=" * 60)
    print(f"Tasks:    {len(task_ids)}")
    print(f"Methods:  {methods}")
    print(f"Eval:     {eval_mode}")
    print(f"Rewriter: {rewriter_mode}")
    print(f"k:        {k}, delta: {delta}, max_rounds: {max_bandit_rounds}")
    print("=" * 60)

    t0_total = time.time()

    # ── Step 1: Load config ───────────────────────────────────────────────────
    config = load_config(config_path)
    config["skillsbench_root"] = str(SKILLSBENCH_ROOT)
    config.setdefault("bandit", {}).update({
        "k_selected": k,
        "delta": delta,
        "max_rounds": max_bandit_rounds,
        "n_demo_candidates": 20,
    })

    # ── Step 2: Build or load demo bank ──────────────────────────────────────
    bank_path = demo_bank_path or str(DEMO_BANK_PATH)
    if Path(bank_path).exists():
        print(f"\nLoading demo bank from {bank_path}...")
        demo_bank = DemoBank.load(bank_path, skillsbench_root=str(SKILLSBENCH_ROOT))
    else:
        print(f"\nBuilding demo bank from {len(task_ids)} tasks...")
        demo_bank = DemoBank(skillsbench_root=str(SKILLSBENCH_ROOT))
        demo_bank.build_from_tasks(max_tasks=89, verbose=verbose)
        if len(demo_bank) > 0:
            DEMO_BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
            demo_bank.save(bank_path)

    print(f"Demo bank: {demo_bank}")

    # ── Step 3: Initialize evaluator ─────────────────────────────────────────
    eval_runner = EvalRunner(
        mode=eval_mode,
        skillsbench_root=str(SKILLSBENCH_ROOT),
        dry_run_seed=42,
        base_success_rate=0.40,
    )

    # ── Step 4: Initialize rewriter (only if needed) ─────────────────────────
    needs_rewriter = any(m in ("bpo_only", "splice_rewrite", "splice_full") for m in methods)
    rewriter = None
    if needs_rewriter:
        print(f"\nInitializing prompt rewriter (mode={rewriter_mode})...")
        try:
            rewriter = SkillPromptRewriter(mode=rewriter_mode)
            print(f"Rewriter ready: {rewriter}")
        except Exception as e:
            print(f"[warn] Could not initialize rewriter: {e}")
            print("[warn] Replacing splice_full with case_only in methods list.")
            methods = [m if m != "splice_full" else "case_only" for m in methods]
            methods = list(dict.fromkeys(methods))  # deduplicate

    # ── Step 5: Run evaluation ────────────────────────────────────────────────
    all_task_results: List[Dict[str, Any]] = []
    results_by_method: Dict[str, List[Dict]] = {m: [] for m in methods}

    for task_num, task_id in enumerate(task_ids):
        print(f"\n[{task_num+1}/{len(task_ids)}] Task: {task_id}")

        task_result = run_task_all_methods(
            task_id=task_id,
            config=config,
            methods=methods,
            demo_bank=demo_bank,
            eval_runner=eval_runner,
            rewriter=rewriter,
            verbose=verbose,
        )
        all_task_results.append(task_result)

        # Aggregate by method
        for method, result in task_result["results"].items():
            results_by_method[method].append(result)

    # ── Step 6: Print results table ───────────────────────────────────────────
    total_time = time.time() - t0_total
    print("\n" + "=" * 60)
    print("PILOT RESULTS")
    print("=" * 60)
    print(_format_table(results_by_method))

    summary = _compute_summary(results_by_method)
    print("\nSummary:")
    for method, stats in summary.items():
        print(
            f"  {method:<20}: "
            f"success_rate={stats['success_rate']:.3f} "
            f"± {stats.get('std', 0):.3f} "
            f"({stats.get('n_success', 0)}/{stats['n_tasks']} tasks)"
        )
    print(f"\nTotal time: {total_time:.1f}s")

    # ── Step 7: Save results ──────────────────────────────────────────────────
    output = {
        "experiment": "splice_pilot",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "eval_mode": eval_mode,
            "rewriter_mode": rewriter_mode,
            "task_ids": task_ids,
            "methods": methods,
            "k": k,
            "delta": delta,
            "max_bandit_rounds": max_bandit_rounds,
        },
        "total_time_sec": round(total_time, 2),
        "summary": summary,
        "results_by_method": {
            method: [
                {
                    "task_id": r["task_id"],
                    "reward": r["reward"],
                    "elapsed_sec": r.get("elapsed_sec", 0),
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
    print(f"\nResults saved to {out_path}")

    return output


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SPLICE Pilot Experiment — compare baseline vs CASE vs SPLICE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval_mode",
        choices=["dry_run", "openai", "bench_cli", "auto"],
        default="dry_run",
        help="Evaluation backend (dry_run for fast testing, bench_cli for real eval)",
    )
    parser.add_argument(
        "--rewriter_mode",
        choices=["auto", "hf", "openai", "claude", "mock"],
        default="mock",
        help="Prompt rewriter backend (mock for dry-run testing, auto detects GPU/API availability)",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task IDs to evaluate (default: 10 pilot tasks)",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=["baseline", "case_only", "splice_full", "bpo_only", "random_kshot"],
        default=None,
        help="Methods to compare (default: baseline case_only splice_full)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config JSON path",
    )
    parser.add_argument(
        "--demo_bank",
        type=str,
        default=None,
        help="Path to pre-built demo_bank.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "pilot_results.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of demos to select per task",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="CASE bandit delta parameter",
    )
    parser.add_argument(
        "--max_bandit_rounds",
        type=int,
        default=30,
        help="Maximum bandit rounds per task (reduce for speed)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
    )
    args = parser.parse_args()

    run_pilot(
        eval_mode=args.eval_mode,
        rewriter_mode=args.rewriter_mode,
        task_ids=args.tasks,
        methods=args.methods,
        config_path=args.config,
        demo_bank_path=args.demo_bank,
        output_path=args.output,
        verbose=args.verbose,
        k=args.k,
        delta=args.delta,
        max_bandit_rounds=args.max_bandit_rounds,
    )


if __name__ == "__main__":
    main()
