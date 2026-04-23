"""
eval_runner.py — SkillsBench Evaluation Integration

Runs an agent on a SkillsBench task and reads the reward from reward.txt.

Supports three evaluation modes:
  1. bench_cli: Uses `bench eval create` CLI (real SkillsBench evaluation)
  2. openai:    Direct GPT API evaluation (fast approximation)
  3. dry_run:   Mock evaluation for testing without API/bench calls

Usage:
    from eval_runner import EvalRunner
    runner = EvalRunner(mode="bench_cli", skillsbench_root="/path/to/skillsbench")
    result = runner.run_task("3d-scan-calc", skill_prompt="...", selected_demos=[...])
    print(result)  # {"reward": 1.0, "task_id": "3d-scan-calc", ...}
"""

import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Defaults ────────────────────────────────────────────────────────────────

SKILLSBENCH_ROOT = Path(__file__).parent.parent / "related-works" / "skillsbench"
DEFAULT_AGENT = "claude-agent-acp"
DEFAULT_TIMEOUT = 900  # seconds


# ── Result structure ──────────────────────────────────────────────────────────

def make_eval_result(
    task_id: str,
    reward: float,
    method: str = "",
    mode: str = "",
    elapsed_sec: float = 0.0,
    error: str = "",
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Create a standardized eval result dict."""
    return {
        "task_id": task_id,
        "reward": reward,
        "method": method,
        "mode": mode,
        "elapsed_sec": elapsed_sec,
        "error": error,
        "metadata": metadata or {},
    }


# ── reward.txt parsing ────────────────────────────────────────────────────────

def _parse_reward_txt(reward_path: Path) -> float:
    """
    Parse SkillsBench reward.txt.

    SkillsBench test.sh writes a reward.txt file with a float value in [0, 1].
    Format examples:
        "1\n" → 1.0
        "0.75\n" → 0.75
        "PASS\n" → 1.0
        "FAIL\n" → 0.0
    """
    if not reward_path.exists():
        return 0.0

    text = reward_path.read_text(encoding="utf-8").strip()

    # Numeric value
    try:
        val = float(text)
        return float(bool(val > 0.0))  # SkillsBench uses 0/1 binary reward
    except ValueError:
        pass

    # Text keywords
    text_upper = text.upper()
    if any(kw in text_upper for kw in ["PASS", "SUCCESS", "TRUE", "1"]):
        return 1.0
    if any(kw in text_upper for kw in ["FAIL", "FAILURE", "FALSE", "0"]):
        return 0.0

    # Try extracting a number from the text
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match:
        val = float(match.group())
        return float(val > 0.0)

    return 0.0


# ── bench CLI evaluator ───────────────────────────────────────────────────────

class BenchCLIEvaluator:
    """
    Evaluates a SkillsBench task using `bench eval create` CLI.

    Reads the resulting reward.txt to get the binary reward.

    Args:
        skillsbench_root: Path to SkillsBench root directory.
        agent: Agent name for bench eval (e.g., "claude-agent-acp").
        timeout: Max seconds to wait for eval to complete.
        jobs_dir: Path to bench jobs output directory.
    """

    def __init__(
        self,
        skillsbench_root: Optional[str] = None,
        agent: str = DEFAULT_AGENT,
        timeout: int = DEFAULT_TIMEOUT,
        jobs_dir: Optional[str] = None,
    ):
        self.skillsbench_root = Path(skillsbench_root) if skillsbench_root else SKILLSBENCH_ROOT
        self.agent = agent
        self.timeout = timeout
        self.jobs_dir = Path(jobs_dir) if jobs_dir else (self.skillsbench_root / "jobs")

    def _get_task_path(self, task_id: str) -> Path:
        """Resolve task_id to absolute task directory path."""
        # Accept "task-name" or "tasks/task-name"
        task_id_clean = task_id.lstrip("/").removeprefix("tasks/")
        task_path = self.skillsbench_root / "tasks" / task_id_clean
        if not task_path.is_dir():
            raise FileNotFoundError(f"Task directory not found: {task_path}")
        return task_path

    def _get_skills_dir(self, task_path: Path) -> Optional[Path]:
        """Get the skills directory for a task, or None if absent."""
        skills_dir = task_path / "environment" / "skills"
        return skills_dir if skills_dir.is_dir() else None

    def _find_reward_txt(self, run_id: str) -> Optional[Path]:
        """Search for reward.txt in the jobs directory after a bench eval run."""
        # bench writes to jobs/<run-id>/...
        # Try common patterns
        for pattern in [
            self.jobs_dir / run_id / "reward.txt",
            self.jobs_dir / run_id / "tests" / "reward.txt",
            self.jobs_dir / run_id / "output" / "reward.txt",
        ]:
            if pattern.exists():
                return pattern

        # Recursive search under run_id directory
        run_dir = self.jobs_dir / run_id
        if run_dir.is_dir():
            for p in run_dir.rglob("reward.txt"):
                return p

        return None

    def run(
        self,
        task_id: str,
        skill_prompt_override: Optional[str] = None,
        custom_skill_body_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run bench eval create for a task.

        If skill_prompt_override is provided, writes it to a temp SKILL.md
        and passes that as the skills directory.

        Args:
            task_id: SkillsBench task ID.
            skill_prompt_override: Optional rewritten skill prompt text.
            custom_skill_body_path: Optional path to a custom SKILL.md file.

        Returns:
            Eval result dict with reward and metadata.
        """
        t0 = time.time()
        run_id = str(uuid.uuid4())[:8]

        try:
            task_path = self._get_task_path(task_id)
        except FileNotFoundError as e:
            return make_eval_result(
                task_id=task_id, reward=0.0, mode="bench_cli",
                elapsed_sec=time.time() - t0, error=str(e)
            )

        skills_dir = self._get_skills_dir(task_path)

        # Handle skill_prompt_override: write to temp file
        temp_dir = None
        effective_skills_dir = skills_dir

        if skill_prompt_override and skills_dir and skills_dir.is_dir():
            import shutil
            temp_dir = Path(tempfile.mkdtemp(prefix="splice_skills_"))
            try:
                # Copy original skills dir
                shutil.copytree(str(skills_dir), str(temp_dir / "skills"))
                # Overwrite first SKILL.md found
                first_skill_md = next(
                    (p for p in (temp_dir / "skills").rglob("SKILL.md")), None
                )
                if first_skill_md:
                    first_skill_md.write_text(skill_prompt_override, encoding="utf-8")
                effective_skills_dir = temp_dir / "skills"
            except Exception as e:
                print(f"  [warn] Could not override skill prompt: {e}")
                temp_dir = None
                effective_skills_dir = skills_dir

        # Build bench eval command
        cmd = [
            "bench", "eval", "create",
            "-t", str(task_path),
            "-a", self.agent,
        ]
        if effective_skills_dir and effective_skills_dir.is_dir():
            cmd += ["-s", str(effective_skills_dir)]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.skillsbench_root),
            )

            # Parse output to find run directory
            stdout = result.stdout + result.stderr
            reward_txt = self._find_reward_from_output(stdout, task_id)

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            return make_eval_result(
                task_id=task_id, reward=0.0, mode="bench_cli",
                elapsed_sec=elapsed, error=f"Timeout after {self.timeout}s",
                metadata={"cmd": " ".join(cmd)}
            )
        except FileNotFoundError:
            # bench CLI not installed
            return make_eval_result(
                task_id=task_id, reward=0.0, mode="bench_cli",
                elapsed_sec=time.time() - t0,
                error="bench CLI not found. Install benchflow: pip install 'benchflow>=0.3.0a7'",
            )
        except Exception as e:
            return make_eval_result(
                task_id=task_id, reward=0.0, mode="bench_cli",
                elapsed_sec=time.time() - t0, error=str(e),
            )
        finally:
            # Cleanup temp dir
            if temp_dir and temp_dir.exists():
                import shutil
                shutil.rmtree(str(temp_dir), ignore_errors=True)

        elapsed = time.time() - t0
        return make_eval_result(
            task_id=task_id,
            reward=reward_txt,
            mode="bench_cli",
            elapsed_sec=elapsed,
            metadata={"agent": self.agent, "cmd": " ".join(cmd)},
        )

    def _find_reward_from_output(self, output_text: str, task_id: str) -> float:
        """
        Parse bench eval stdout/stderr to find reward.txt and read its value.

        bench eval create outputs something like:
          "Job created: jobs/abc123"
          "Reward: 1" or "reward.txt: 1"
        """
        # Look for explicit reward mention in output
        reward_match = re.search(
            r"[Rr]eward[:=\s]+([0-9]*\.?[0-9]+)", output_text
        )
        if reward_match:
            try:
                return float(reward_match.group(1))
            except ValueError:
                pass

        # Look for job directory mention
        job_match = re.search(r"jobs/([a-zA-Z0-9_\-/]+)", output_text)
        if job_match:
            job_id = job_match.group(1).strip("/")
            reward_path = self._find_reward_txt(job_id)
            if reward_path:
                return _parse_reward_txt(reward_path)

        # Try latest job in jobs directory
        if self.jobs_dir.is_dir():
            job_dirs = sorted(
                self.jobs_dir.iterdir(),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for jd in job_dirs[:3]:
                if jd.is_dir() and task_id in jd.name:
                    reward_path = self._find_reward_txt(jd.name)
                    if reward_path:
                        return _parse_reward_txt(reward_path)

        return 0.0


# ── OpenAI direct evaluator ───────────────────────────────────────────────────

class OpenAIEvaluator:
    """
    Evaluate a task by calling OpenAI GPT directly with the task instruction
    and skill prompt. Fast approximation without running a full agent.

    Args:
        model: OpenAI model.
        skillsbench_root: Path to SkillsBench root.
        max_tokens: Max response tokens.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        skillsbench_root: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model = model
        self.skillsbench_root = Path(skillsbench_root) if skillsbench_root else SKILLSBENCH_ROOT
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "openai package required. Install: pip install openai"
                ) from e
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set.")
            self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def _load_task(self, task_id: str) -> Dict[str, str]:
        """Load task instruction and skill body."""
        task_id_clean = task_id.lstrip("/").removeprefix("tasks/")
        task_path = self.skillsbench_root / "tasks" / task_id_clean

        instruction = ""
        skill_body = ""

        inst_file = task_path / "instruction.md"
        if inst_file.exists():
            instruction = inst_file.read_text(encoding="utf-8")

        skills_dir = task_path / "environment" / "skills"
        if skills_dir.is_dir():
            for skill_md in sorted(skills_dir.rglob("SKILL.md"))[:1]:
                skill_body = skill_md.read_text(encoding="utf-8")

        return {"instruction": instruction, "skill_body": skill_body}

    def run(
        self,
        task_id: str,
        skill_prompt_override: Optional[str] = None,
        selected_demos: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Approximate task evaluation via OpenAI API.

        Builds a prompt from task instruction + skill body (or override),
        optionally with selected demos, and asks GPT to evaluate success.
        """
        t0 = time.time()
        client = self._get_client()

        try:
            task_data = self._load_task(task_id)
        except Exception as e:
            return make_eval_result(
                task_id=task_id, reward=0.0, mode="openai",
                elapsed_sec=time.time() - t0, error=str(e)
            )

        skill_body = skill_prompt_override or task_data["skill_body"]
        instruction = task_data["instruction"]

        # Build prompt
        demo_section = ""
        if selected_demos:
            demo_lines = []
            for i, d in enumerate(selected_demos[:3]):
                demo_lines.append(
                    f"Example {i+1}:\n"
                    f"  Task: {str(d.get('instruction', ''))[:200]}\n"
                    f"  Invocation: {str(d.get('invocation', ''))[:200]}"
                )
            demo_section = "\n\nRelevant examples:\n" + "\n\n".join(demo_lines)

        eval_prompt = (
            f"You are evaluating an AI agent on a task. Based on the task description "
            f"and available skill, determine if the agent would likely succeed.\n\n"
            f"Task:\n{instruction[:500]}\n\n"
            f"Available Skill:\n{skill_body[:500]}"
            f"{demo_section}\n\n"
            f"Would an agent using this skill likely complete the task successfully? "
            f"Answer ONLY with 'YES' or 'NO'."
        )

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": eval_prompt}],
                    max_tokens=10,
                    temperature=0.0,
                )
                answer = response.choices[0].message.content.strip().upper()
                reward = 1.0 if "YES" in answer else 0.0
                elapsed = time.time() - t0
                return make_eval_result(
                    task_id=task_id, reward=reward, mode="openai",
                    elapsed_sec=elapsed,
                    metadata={"model": self.model, "answer": answer},
                )
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"  [warn] OpenAI eval error (attempt {attempt+1}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    return make_eval_result(
                        task_id=task_id, reward=0.0, mode="openai",
                        elapsed_sec=time.time() - t0, error=str(e)
                    )


# ── Dry-run evaluator ─────────────────────────────────────────────────────────

class DryRunEvaluator:
    """
    Mock evaluator for testing SPLICE without real API/bench calls.

    Uses simple heuristics on skill_prompt and instruction length to simulate
    a realistic (but random) reward signal.

    Args:
        seed: Random seed for reproducibility.
        base_success_rate: Base probability of success (default: 0.4).
        demo_bonus: Reward bonus per selected demo (default: 0.05).
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        base_success_rate: float = 0.40,
        demo_bonus: float = 0.05,
    ):
        import random as _random
        self._rng = _random.Random(seed)
        self.base_success_rate = base_success_rate
        self.demo_bonus = demo_bonus

    def run(
        self,
        task_id: str,
        skill_prompt_override: Optional[str] = None,
        selected_demos: Optional[List[Dict]] = None,
        method: str = "",
    ) -> Dict[str, Any]:
        """Simulate task evaluation with controlled random outcomes."""
        t0 = time.time()
        n_demos = len(selected_demos) if selected_demos else 0
        has_rewrite = skill_prompt_override is not None

        # Simulate reward: base + demo bonus + rewrite bonus
        success_prob = self.base_success_rate
        success_prob += n_demos * self.demo_bonus
        if has_rewrite:
            success_prob += 0.08  # BPO rewriting bonus
        success_prob = min(success_prob, 0.95)

        reward = float(self._rng.random() < success_prob)

        elapsed = time.time() - t0 + self._rng.uniform(0.01, 0.05)  # simulate latency
        return make_eval_result(
            task_id=task_id,
            reward=reward,
            mode="dry_run",
            method=method,
            elapsed_sec=elapsed,
            metadata={
                "success_prob": round(success_prob, 3),
                "n_demos": n_demos,
                "has_rewrite": has_rewrite,
            },
        )


# ── Unified EvalRunner ────────────────────────────────────────────────────────

class EvalRunner:
    """
    Unified SPLICE evaluation runner.

    Dispatches to the appropriate backend based on mode.

    Args:
        mode: "bench_cli", "openai", "dry_run", or "auto".
        skillsbench_root: Path to SkillsBench root.
        agent: Agent name for bench CLI.
        dry_run_seed: Seed for dry_run mode.
        base_success_rate: Base success rate for dry_run.
        openai_model: Model for OpenAI evaluator.
        bench_timeout: Timeout for bench CLI.
    """

    def __init__(
        self,
        mode: str = "dry_run",
        skillsbench_root: Optional[str] = None,
        agent: str = DEFAULT_AGENT,
        dry_run_seed: Optional[int] = 42,
        base_success_rate: float = 0.40,
        openai_model: str = "gpt-4o-mini",
        bench_timeout: int = DEFAULT_TIMEOUT,
    ):
        self.mode = mode
        self.skillsbench_root = skillsbench_root or str(SKILLSBENCH_ROOT)

        if mode == "bench_cli":
            self._evaluator = BenchCLIEvaluator(
                skillsbench_root=self.skillsbench_root,
                agent=agent,
                timeout=bench_timeout,
            )
        elif mode == "openai":
            self._evaluator = OpenAIEvaluator(
                model=openai_model,
                skillsbench_root=self.skillsbench_root,
            )
        elif mode == "dry_run":
            self._evaluator = DryRunEvaluator(
                seed=dry_run_seed,
                base_success_rate=base_success_rate,
            )
        elif mode == "auto":
            # Try bench CLI, fall back to OpenAI, then dry_run
            self._evaluator = None  # determined on first call
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from: bench_cli, openai, dry_run, auto")

    def _auto_select(self) -> None:
        """Auto-select evaluator backend."""
        # Check if bench CLI is available
        try:
            result = subprocess.run(
                ["bench", "--help"],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                print("[eval] bench CLI available, using bench_cli mode.")
                self._evaluator = BenchCLIEvaluator(skillsbench_root=self.skillsbench_root)
                return
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Check for OpenAI key
        if os.environ.get("OPENAI_API_KEY"):
            print("[eval] Using OpenAI API evaluator.")
            self._evaluator = OpenAIEvaluator(skillsbench_root=self.skillsbench_root)
            return

        # Fallback to dry_run
        print("[eval] No real evaluator available, using dry_run mode.")
        self._evaluator = DryRunEvaluator()

    def run_task(
        self,
        task_id: str,
        skill_prompt_override: Optional[str] = None,
        selected_demos: Optional[List[Dict]] = None,
        method: str = "",
    ) -> Dict[str, Any]:
        """
        Run evaluation for a single task.

        Args:
            task_id: SkillsBench task ID (directory name under tasks/).
            skill_prompt_override: Optional rewritten skill prompt.
            selected_demos: Optional selected demonstrations.
            method: Method name for result logging.

        Returns:
            Eval result dict with reward, timing, and metadata.
        """
        if self._evaluator is None:
            self._auto_select()

        try:
            if isinstance(self._evaluator, DryRunEvaluator):
                result = self._evaluator.run(
                    task_id=task_id,
                    skill_prompt_override=skill_prompt_override,
                    selected_demos=selected_demos,
                    method=method,
                )
            elif isinstance(self._evaluator, OpenAIEvaluator):
                result = self._evaluator.run(
                    task_id=task_id,
                    skill_prompt_override=skill_prompt_override,
                    selected_demos=selected_demos,
                )
                result["method"] = method
            elif isinstance(self._evaluator, BenchCLIEvaluator):
                result = self._evaluator.run(
                    task_id=task_id,
                    skill_prompt_override=skill_prompt_override,
                )
                result["method"] = method
            else:
                result = make_eval_result(
                    task_id=task_id, reward=0.0,
                    error="Unknown evaluator type", method=method
                )
        except Exception as e:
            result = make_eval_result(
                task_id=task_id, reward=0.0,
                mode=self.mode, method=method,
                error=f"Unexpected error: {e}"
            )

        return result

    def run_tasks_batch(
        self,
        task_ids: List[str],
        skill_prompt_overrides: Optional[Dict[str, str]] = None,
        selected_demos_map: Optional[Dict[str, List[Dict]]] = None,
        method: str = "",
        n_workers: int = 1,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run evaluation for multiple tasks.

        Args:
            task_ids: List of task IDs.
            skill_prompt_overrides: Optional map of task_id -> rewritten prompt.
            selected_demos_map: Optional map of task_id -> demo list.
            method: Method name.
            n_workers: Number of parallel workers (>1 uses multiprocessing).
            verbose: Print progress.

        Returns:
            List of eval result dicts.
        """
        overrides = skill_prompt_overrides or {}
        demos_map = selected_demos_map or {}
        results = []

        if n_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        self.run_task,
                        tid,
                        overrides.get(tid),
                        demos_map.get(tid),
                        method,
                    ): tid
                    for tid in task_ids
                }
                for future in as_completed(futures):
                    tid = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = make_eval_result(
                            task_id=tid, reward=0.0, method=method,
                            error=str(e)
                        )
                    results.append(result)
                    if verbose:
                        print(f"  [{result['reward']:.0f}] {tid}")
        else:
            for i, tid in enumerate(task_ids):
                result = self.run_task(
                    tid,
                    overrides.get(tid),
                    demos_map.get(tid),
                    method,
                )
                results.append(result)
                if verbose:
                    print(
                        f"  [{i+1:3d}/{len(task_ids)}] {tid}: "
                        f"reward={result['reward']:.2f} "
                        f"({result['elapsed_sec']:.1f}s)"
                    )

        return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run SPLICE eval on a SkillsBench task")
    parser.add_argument("task_id", help="SkillsBench task ID")
    parser.add_argument(
        "--mode",
        choices=["bench_cli", "openai", "dry_run", "auto"],
        default="dry_run",
    )
    parser.add_argument("--skillsbench_root", default=str(SKILLSBENCH_ROOT))
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--skill_prompt", type=str, default=None)
    args = parser.parse_args()

    runner = EvalRunner(
        mode=args.mode,
        skillsbench_root=args.skillsbench_root,
        agent=args.agent,
    )

    result = runner.run_task(
        task_id=args.task_id,
        skill_prompt_override=args.skill_prompt,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import json
    main()
