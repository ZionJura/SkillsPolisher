"""
splice_loop.py — Main SPLICE Orchestration Loop

Orchestrates the full SPLICE pipeline for a single task:
  1. SPLICE-Select: CASE bandit demo selection
  2. SPLICE-Rewrite: BPO-style skill prompt rewriting
  3. Evaluation: bench CLI / OpenAI / dry_run

Supports --method flag:
  - baseline:       No demos, original skill prompt
  - case_only:      CASE-selected demos, original skill prompt
  - bpo_only:       BPO-rewritten prompt, no bandit selection (random demos)
  - random_kshot:   Random k demos, original skill prompt
  - splice_select:  Bandit demo selection, original skill prompt (= case_only)
  - splice_rewrite: BPO rewriting with CASE demos (= splice_full minus loop)
  - splice_full:    Full SPLICE: bandit selection + BPO rewriting

Usage:
    python splice_loop.py --task 3d-scan-calc --method splice_full --output results/out.json
"""

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from demo_bank import DemoBank
from eval_runner import EvalRunner
from splice_rewrite import SkillPromptRewriter
from splice_select import splice_select, CASEBandit

# ── Defaults ────────────────────────────────────────────────────────────────

SPLICE_ROOT = Path(__file__).parent
SKILLSBENCH_ROOT = SPLICE_ROOT.parent / "related-works" / "skillsbench"
DEFAULT_CONFIG = SPLICE_ROOT / "configs" / "splice_default.json"

VALID_METHODS = [
    "baseline",
    "case_only",
    "bpo_only",
    "random_kshot",
    "splice_select",
    "splice_rewrite",
    "splice_full",
]


# ── Config loading ───────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load SPLICE config from JSON, merging with defaults."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def merge_config_with_args(config: Dict, args_dict: Dict) -> Dict:
    """Override config fields with non-None CLI args."""
    result = dict(config)
    for k, v in args_dict.items():
        if v is not None:
            result[k] = v
    return result


# ── Skill prompt loading ─────────────────────────────────────────────────────

def load_skill_prompt(task_id: str, skillsbench_root: Path) -> str:
    """
    Load the primary skill prompt (SKILL.md) for a task.
    Returns empty string if no skill found.
    """
    task_id_clean = task_id.lstrip("/").removeprefix("tasks/")
    skills_dir = skillsbench_root / "tasks" / task_id_clean / "environment" / "skills"

    if not skills_dir.is_dir():
        return ""

    for skill_md in sorted(skills_dir.rglob("SKILL.md"))[:1]:
        try:
            return skill_md.read_text(encoding="utf-8")
        except Exception:
            pass
    return ""


# ── SPLICE pipeline steps ─────────────────────────────────────────────────────

def _run_bandit_selection(
    task_id: str,
    demo_candidates: List[Dict],
    skill_prompt: str,
    eval_runner: EvalRunner,
    k: int = 3,
    delta: float = 0.05,
    max_rounds: int = 50,
    verbose: bool = False,
) -> Tuple[List[int], List[Dict], CASEBandit]:
    """
    Run CASE bandit to select top-k demonstrations.

    Returns:
        (top_k_indices, selected_demos, bandit)
    """
    if not demo_candidates:
        return [], [], CASEBandit(n_arms=1, k=1)

    def eval_fn(arm_idx: int, candidates: List[Dict]) -> float:
        """Evaluate a single demonstration by running the task with it."""
        demo = candidates[arm_idx]
        result = eval_runner.run_task(
            task_id=task_id,
            skill_prompt_override=None,  # Use original skill prompt
            selected_demos=[demo],
            method="bandit_eval",
        )
        return result["reward"]

    top_k_indices, bandit = splice_select(
        demo_candidates=demo_candidates,
        eval_fn=eval_fn,
        k=k,
        delta=delta,
        max_rounds=max_rounds,
        verbose=verbose,
    )

    selected_demos = [demo_candidates[i] for i in top_k_indices]
    return top_k_indices, selected_demos, bandit


def _run_bpo_rewrite(
    skill_prompt: str,
    selected_demos: List[Dict],
    rewriter: SkillPromptRewriter,
) -> str:
    """Apply BPO-style prompt rewriting. Returns rewritten prompt."""
    if not skill_prompt.strip():
        return skill_prompt
    return rewriter.rewrite(skill_prompt, selected_demos)


# ── Method implementations ───────────────────────────────────────────────────

class SPLICEPipeline:
    """
    Orchestrates the full SPLICE evaluation pipeline for a single task.

    Args:
        task_id: SkillsBench task ID.
        config: Config dict (from splice_default.json or CLI args).
        demo_bank: Pre-loaded DemoBank instance.
        eval_runner: EvalRunner instance.
        rewriter: SkillPromptRewriter instance (may be None for non-BPO methods).
        verbose: Print verbose output.
    """

    def __init__(
        self,
        task_id: str,
        config: Dict[str, Any],
        demo_bank: DemoBank,
        eval_runner: EvalRunner,
        rewriter: Optional[SkillPromptRewriter] = None,
        verbose: bool = True,
    ):
        self.task_id = task_id
        self.config = config
        self.demo_bank = demo_bank
        self.eval_runner = eval_runner
        self.rewriter = rewriter
        self.verbose = verbose

        self.skillsbench_root = Path(
            config.get("skillsbench_root", str(SKILLSBENCH_ROOT))
        )
        self.k = int(config.get("bandit", {}).get("k_selected", 3))
        self.delta = float(config.get("bandit", {}).get("delta", 0.05))
        self.max_bandit_rounds = int(config.get("bandit", {}).get("max_rounds", 100))
        self.n_demo_candidates = int(config.get("bandit", {}).get("n_demo_candidates", 20))

    def _get_demo_candidates(self) -> List[Dict]:
        """Get demo candidates for this task from the bank."""
        return self.demo_bank.get_candidates_for_task(
            task_id=self.task_id,
            max_candidates=self.n_demo_candidates,
            exclude_task=True,
        )

    def _random_demos(self, k: Optional[int] = None) -> List[Dict]:
        """Sample k random demos (used for bpo_only and random_kshot)."""
        candidates = self._get_demo_candidates()
        k = k or self.k
        if not candidates:
            return []
        sample_size = min(k, len(candidates))
        return random.sample(candidates, sample_size)

    def run(self, method: str) -> Dict[str, Any]:
        """
        Run SPLICE pipeline for the given method.

        Args:
            method: One of VALID_METHODS.

        Returns:
            Result dict with reward, timing, method metadata.
        """
        assert method in VALID_METHODS, f"Unknown method: {method}. Valid: {VALID_METHODS}"

        t0 = time.time()
        skill_prompt = load_skill_prompt(self.task_id, self.skillsbench_root)

        if self.verbose:
            print(f"\n[SPLICE] Task: {self.task_id} | Method: {method}")

        # ── baseline: no demos, original prompt ──────────────────────────────
        if method == "baseline":
            result = self.eval_runner.run_task(
                task_id=self.task_id,
                skill_prompt_override=None,
                selected_demos=None,
                method=method,
            )
            return self._finalize(result, method, t0, selected_demos=[])

        # ── random_kshot: random demos, original prompt ───────────────────────
        if method == "random_kshot":
            random_demos = self._random_demos()
            result = self.eval_runner.run_task(
                task_id=self.task_id,
                skill_prompt_override=None,
                selected_demos=random_demos,
                method=method,
            )
            return self._finalize(result, method, t0, selected_demos=random_demos)

        # ── bpo_only: BPO rewriting with random demos, no bandit ─────────────
        if method == "bpo_only":
            if self.rewriter is None:
                raise ValueError("bpo_only requires a SkillPromptRewriter instance.")
            random_demos = self._random_demos()
            rewritten = _run_bpo_rewrite(skill_prompt, random_demos, self.rewriter)
            if self.verbose:
                print(f"  [BPO] Rewritten prompt (first 120 chars): {rewritten[:120]}...")
            result = self.eval_runner.run_task(
                task_id=self.task_id,
                skill_prompt_override=rewritten,
                selected_demos=random_demos,
                method=method,
            )
            return self._finalize(
                result, method, t0,
                selected_demos=random_demos, rewritten_prompt=rewritten
            )

        # ── case_only / splice_select: CASE bandit, original prompt ──────────
        if method in ("case_only", "splice_select"):
            demo_candidates = self._get_demo_candidates()
            if self.verbose:
                print(f"  [CASE] Running bandit on {len(demo_candidates)} candidates...")
            top_k_indices, selected_demos, bandit = _run_bandit_selection(
                task_id=self.task_id,
                demo_candidates=demo_candidates,
                skill_prompt=skill_prompt,
                eval_runner=self.eval_runner,
                k=self.k,
                delta=self.delta,
                max_rounds=self.max_bandit_rounds,
                verbose=self.verbose,
            )
            if self.verbose:
                print(f"  [CASE] Selected top-{self.k}: indices {top_k_indices}")
            result = self.eval_runner.run_task(
                task_id=self.task_id,
                skill_prompt_override=None,
                selected_demos=selected_demos,
                method=method,
            )
            return self._finalize(
                result, method, t0,
                selected_demos=selected_demos,
                bandit_diag=bandit.diagnostics(),
            )

        # ── splice_rewrite: CASE demos + BPO rewriting ────────────────────────
        if method == "splice_rewrite":
            if self.rewriter is None:
                raise ValueError("splice_rewrite requires a SkillPromptRewriter.")
            demo_candidates = self._get_demo_candidates()
            top_k_indices, selected_demos, bandit = _run_bandit_selection(
                task_id=self.task_id,
                demo_candidates=demo_candidates,
                skill_prompt=skill_prompt,
                eval_runner=self.eval_runner,
                k=self.k,
                delta=self.delta,
                max_rounds=self.max_bandit_rounds,
                verbose=self.verbose,
            )
            rewritten = _run_bpo_rewrite(skill_prompt, selected_demos, self.rewriter)
            if self.verbose:
                print(f"  [BPO] Rewritten prompt (first 120 chars): {rewritten[:120]}...")
            result = self.eval_runner.run_task(
                task_id=self.task_id,
                skill_prompt_override=rewritten,
                selected_demos=selected_demos,
                method=method,
            )
            return self._finalize(
                result, method, t0,
                selected_demos=selected_demos, rewritten_prompt=rewritten,
                bandit_diag=bandit.diagnostics(),
            )

        # ── splice_full: Full SPLICE pipeline ────────────────────────────────
        if method == "splice_full":
            if self.rewriter is None:
                raise ValueError("splice_full requires a SkillPromptRewriter.")

            n_rounds = int(self.config.get("loop", {}).get("n_rounds", 1))
            demo_candidates = self._get_demo_candidates()
            selected_demos: List[Dict] = []
            rewritten = skill_prompt
            bandit_diag: Dict = {}

            for round_num in range(n_rounds):
                if self.verbose:
                    print(f"  [SPLICE round {round_num+1}/{n_rounds}]")

                # Step 1: CASE bandit demo selection
                top_k_indices, selected_demos, bandit = _run_bandit_selection(
                    task_id=self.task_id,
                    demo_candidates=demo_candidates,
                    skill_prompt=rewritten,
                    eval_runner=self.eval_runner,
                    k=self.k,
                    delta=self.delta,
                    max_rounds=self.max_bandit_rounds,
                    verbose=self.verbose,
                )
                bandit_diag = bandit.diagnostics()
                if self.verbose:
                    print(f"    Selected demos: {top_k_indices}")

                # Step 2: BPO rewrite using selected demos
                rewritten = _run_bpo_rewrite(skill_prompt, selected_demos, self.rewriter)
                if self.verbose:
                    print(f"    Rewritten (first 100 chars): {rewritten[:100]}...")

            # Final evaluation with selected demos + rewritten prompt
            result = self.eval_runner.run_task(
                task_id=self.task_id,
                skill_prompt_override=rewritten,
                selected_demos=selected_demos,
                method=method,
            )
            return self._finalize(
                result, method, t0,
                selected_demos=selected_demos, rewritten_prompt=rewritten,
                bandit_diag=bandit_diag,
            )

        raise ValueError(f"Unhandled method: {method}")

    def _finalize(
        self,
        result: Dict[str, Any],
        method: str,
        t0: float,
        selected_demos: Optional[List[Dict]] = None,
        rewritten_prompt: Optional[str] = None,
        bandit_diag: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Finalize result with SPLICE-specific metadata."""
        total_elapsed = time.time() - t0
        result.update({
            "method": method,
            "task_id": self.task_id,
            "total_elapsed_sec": round(total_elapsed, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_demos_selected": len(selected_demos) if selected_demos else 0,
            "demo_task_ids": [d["task_id"] for d in (selected_demos or [])],
            "was_rewritten": rewritten_prompt is not None,
            "bandit_diagnostics": bandit_diag,
        })
        if self.verbose:
            print(f"  Reward: {result['reward']:.2f} | Time: {total_elapsed:.1f}s")
        return result


# ── Main runner ──────────────────────────────────────────────────────────────

def run_task_all_methods(
    task_id: str,
    config: Dict[str, Any],
    methods: List[str],
    demo_bank: DemoBank,
    eval_runner: EvalRunner,
    rewriter: Optional[SkillPromptRewriter] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all specified methods on a single task.

    Returns:
        {"task_id": str, "results": {method: result_dict}}
    """
    pipeline = SPLICEPipeline(
        task_id=task_id,
        config=config,
        demo_bank=demo_bank,
        eval_runner=eval_runner,
        rewriter=rewriter,
        verbose=verbose,
    )

    task_results: Dict[str, Any] = {"task_id": task_id, "results": {}}
    for method in methods:
        try:
            result = pipeline.run(method)
        except Exception as e:
            print(f"  [error] {task_id} / {method}: {e}")
            result = {
                "task_id": task_id, "method": method, "reward": 0.0,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        task_results["results"][method] = result

    return task_results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run SPLICE pipeline on a SkillsBench task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", required=True, help="SkillsBench task ID")
    parser.add_argument(
        "--method",
        choices=VALID_METHODS,
        default="splice_full",
        help="SPLICE method to run",
    )
    parser.add_argument(
        "--eval_mode",
        choices=["bench_cli", "openai", "dry_run", "auto"],
        default="dry_run",
        help="Evaluation backend",
    )
    parser.add_argument(
        "--rewriter_mode",
        choices=["hf", "openai", "claude", "auto"],
        default="auto",
        help="Prompt rewriter backend (only used for BPO methods)",
    )
    parser.add_argument("--config", type=str, default=None, help="Config JSON path")
    parser.add_argument("--demo_bank", type=str, default=None, help="Path to demo_bank.json")
    parser.add_argument("--skillsbench_root", type=str, default=str(SKILLSBENCH_ROOT))
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--k", type=int, default=None, help="Number of demos to select")
    parser.add_argument("--delta", type=float, default=None, help="Bandit delta parameter")
    parser.add_argument("--max_rounds", type=int, default=None, help="Max bandit rounds")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.skillsbench_root:
        config["skillsbench_root"] = args.skillsbench_root
    if args.k:
        config.setdefault("bandit", {})["k_selected"] = args.k
    if args.delta:
        config.setdefault("bandit", {})["delta"] = args.delta
    if args.max_rounds:
        config.setdefault("bandit", {})["max_rounds"] = args.max_rounds

    # Load or build demo bank
    if args.demo_bank and Path(args.demo_bank).exists():
        demo_bank = DemoBank.load(args.demo_bank, skillsbench_root=args.skillsbench_root)
    else:
        print("Building demo bank from SkillsBench tasks...")
        demo_bank = DemoBank(skillsbench_root=args.skillsbench_root)
        demo_bank.build_from_tasks(max_tasks=89, verbose=args.verbose)

    # Create evaluator
    eval_runner = EvalRunner(
        mode=args.eval_mode,
        skillsbench_root=args.skillsbench_root,
    )

    # Create rewriter (only if method needs it)
    needs_rewriter = args.method in ("bpo_only", "splice_rewrite", "splice_full")
    rewriter = None
    if needs_rewriter:
        rewriter = SkillPromptRewriter(mode=args.rewriter_mode)

    # Run pipeline
    pipeline = SPLICEPipeline(
        task_id=args.task,
        config=config,
        demo_bank=demo_bank,
        eval_runner=eval_runner,
        rewriter=rewriter,
        verbose=args.verbose,
    )

    result = pipeline.run(args.method)

    print(f"\nResult: reward={result['reward']:.2f} | method={result['method']}")

    # Save output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {out_path}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
