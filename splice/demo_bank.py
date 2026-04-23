"""
demo_bank.py — SPLICE Demonstration Bank

Builds and manages a bank of (task, skill_invocation, outcome) triples
from SkillsBench tasks. Supports save/load from JSON.

Usage:
    from demo_bank import DemoBank
    bank = DemoBank(skillsbench_root="/path/to/skillsbench")
    bank.build_from_tasks()
    bank.save("data/demo_bank.json")
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Defaults ────────────────────────────────────────────────────────────────

SKILLSBENCH_ROOT = Path(__file__).parent.parent / "related-works" / "skillsbench"


# ── Data structures ──────────────────────────────────────────────────────────

def make_demo(
    task_id: str,
    instruction: str,
    skill_name: str,
    skill_body: str,
    invocation: str,
    outcome: int,
    category: str = "",
    difficulty: str = "",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a demonstration record."""
    return {
        "task_id": task_id,
        "instruction": instruction,
        "skill_name": skill_name,
        "skill_body": skill_body,
        "invocation": invocation,
        "outcome": outcome,  # 1 = success, 0 = failure
        "category": category,
        "difficulty": difficulty,
        "tags": tags or [],
    }


# ── Task reading helpers ─────────────────────────────────────────────────────

def _read_text(path: Path, max_chars: int = 8000) -> str:
    """Read a text file, returning empty string if not found."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[:max_chars]
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return ""


def _parse_toml_field(toml_text: str, field: str) -> str:
    """
    Very simple TOML field parser for extracting string values.
    Handles: field = "value" or field = 'value'
    """
    pattern = rf'{re.escape(field)}\s*=\s*["\']([^"\']*)["\']'
    match = re.search(pattern, toml_text)
    return match.group(1) if match else ""


def _parse_toml_list(toml_text: str, field: str) -> List[str]:
    """Parse a TOML array field like: tags = ["a", "b"]"""
    pattern = rf'{re.escape(field)}\s*=\s*\[([^\]]*)\]'
    match = re.search(pattern, toml_text)
    if not match:
        return []
    items = re.findall(r'["\']([^"\']+)["\']', match.group(1))
    return items


def _read_task_metadata(task_dir: Path) -> Dict[str, str]:
    """Read task.toml for metadata fields."""
    toml_text = _read_text(task_dir / "task.toml")
    return {
        "category": _parse_toml_field(toml_text, "category"),
        "difficulty": _parse_toml_field(toml_text, "difficulty"),
        "tags": _parse_toml_list(toml_text, "tags"),
    }


def _read_skills(task_dir: Path) -> List[Dict[str, str]]:
    """
    Read all skills in environment/skills/.
    Returns list of {"name": str, "body": str} dicts.
    """
    skills_dir = task_dir / "environment" / "skills"
    skills = []
    if not skills_dir.is_dir():
        return skills

    for skill_subdir in sorted(skills_dir.iterdir()):
        if not skill_subdir.is_dir():
            continue
        skill_name = skill_subdir.name
        skill_md = skill_subdir / "SKILL.md"
        skill_body = _read_text(skill_md, max_chars=6000)
        if skill_body:
            skills.append({"name": skill_name, "body": skill_body})

    return skills


def _read_oracle_invocation(task_dir: Path) -> str:
    """
    Read the oracle solve.sh as a proxy for the gold invocation.
    Falls back to any .sh file in solution/.
    """
    solve_sh = task_dir / "solution" / "solve.sh"
    if solve_sh.exists():
        return _read_text(solve_sh, max_chars=4000)

    # Fallback: any shell script in solution/
    solution_dir = task_dir / "solution"
    if solution_dir.is_dir():
        for f in solution_dir.glob("*.sh"):
            text = _read_text(f, max_chars=4000)
            if text:
                return text
    return ""


# ── DemoBank class ───────────────────────────────────────────────────────────

class DemoBank:
    """
    Manages a bank of (task, skill_invocation, outcome) demonstrations
    for SPLICE bandit demo selection.

    Attributes:
        skillsbench_root: Path to SkillsBench root directory.
        demos: List of demo dicts (see make_demo()).
    """

    def __init__(self, skillsbench_root: Optional[str] = None):
        self.skillsbench_root = Path(skillsbench_root) if skillsbench_root else SKILLSBENCH_ROOT
        self.demos: List[Dict[str, Any]] = []

    # ── Building ──────────────────────────────────────────────────────────────

    def build_from_tasks(
        self,
        task_ids: Optional[List[str]] = None,
        max_tasks: int = 200,
        verbose: bool = True,
    ) -> "DemoBank":
        """
        Build the demonstration bank by scanning SkillsBench task directories.

        Args:
            task_ids: Optional list of task IDs (directory names under tasks/).
                      If None, scans all tasks.
            max_tasks: Maximum number of tasks to process.
            verbose: Print progress.

        Returns:
            self (for chaining)
        """
        tasks_dir = self.skillsbench_root / "tasks"
        if not tasks_dir.is_dir():
            raise FileNotFoundError(
                f"SkillsBench tasks directory not found: {tasks_dir}. "
                f"Set skillsbench_root correctly."
            )

        # Determine task list
        if task_ids is None:
            all_task_dirs = sorted(
                [d for d in tasks_dir.iterdir() if d.is_dir()]
            )[:max_tasks]
        else:
            all_task_dirs = []
            for tid in task_ids:
                # Accept both "task-name" and "tasks/task-name" formats
                tid_clean = tid.lstrip("/").removeprefix("tasks/")
                td = tasks_dir / tid_clean
                if td.is_dir():
                    all_task_dirs.append(td)
                elif verbose:
                    print(f"  [warn] Task directory not found: {td}")

        added = 0
        skipped = 0
        for task_dir in all_task_dirs:
            task_id = task_dir.name
            try:
                demo = self._build_demo_from_task_dir(task_dir)
            except Exception as e:
                if verbose:
                    print(f"  [skip] {task_id}: {e}")
                skipped += 1
                continue

            if demo:
                self.demos.append(demo)
                added += 1
                if verbose:
                    print(f"  [ok] {task_id} — skill: {demo['skill_name']}")

        if verbose:
            print(f"\nDemoBank built: {added} demos added, {skipped} skipped.")
        return self

    def _build_demo_from_task_dir(self, task_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Build a single demo from a task directory.
        Returns None if insufficient data is found.
        """
        task_id = task_dir.name

        # Required: instruction.md
        instruction = _read_text(task_dir / "instruction.md")
        if not instruction.strip():
            return None  # no instruction, skip

        # Skills
        skills = _read_skills(task_dir)
        if skills:
            primary_skill = skills[0]
            skill_name = primary_skill["name"]
            skill_body = primary_skill["body"]
        else:
            skill_name = "no-skill"
            skill_body = ""

        # Oracle invocation
        invocation = _read_oracle_invocation(task_dir)

        # Metadata
        meta = _read_task_metadata(task_dir)

        # Oracle demos are treated as successes (outcome=1)
        return make_demo(
            task_id=task_id,
            instruction=instruction,
            skill_name=skill_name,
            skill_body=skill_body,
            invocation=invocation,
            outcome=1,
            category=meta.get("category", ""),
            difficulty=meta.get("difficulty", ""),
            tags=meta.get("tags", []),
        )

    def add_demo(
        self,
        task_id: str,
        instruction: str,
        skill_name: str,
        skill_body: str,
        invocation: str,
        outcome: int,
        **kwargs,
    ) -> None:
        """Manually add a demo to the bank."""
        self.demos.append(
            make_demo(
                task_id=task_id,
                instruction=instruction,
                skill_name=skill_name,
                skill_body=skill_body,
                invocation=invocation,
                outcome=outcome,
                **kwargs,
            )
        )

    # ── Querying ──────────────────────────────────────────────────────────────

    def get_candidates_for_task(
        self,
        task_id: str,
        max_candidates: int = 30,
        exclude_task: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Return demo candidates for a given task (excluding itself if requested).

        Args:
            task_id: The task being evaluated.
            max_candidates: Max demos to return.
            exclude_task: If True, exclude demos from the same task.

        Returns:
            List of demo dicts.
        """
        candidates = [
            d for d in self.demos
            if (not exclude_task or d["task_id"] != task_id)
        ]
        return candidates[:max_candidates]

    def get_demos_by_skill(self, skill_name: str) -> List[Dict[str, Any]]:
        """Return all demos for a given skill name."""
        return [d for d in self.demos if d["skill_name"] == skill_name]

    def get_successful_demos(self) -> List[Dict[str, Any]]:
        """Return only success demos (outcome=1)."""
        return [d for d in self.demos if d["outcome"] == 1]

    def __len__(self) -> int:
        return len(self.demos)

    def __repr__(self) -> str:
        skills = set(d["skill_name"] for d in self.demos)
        return (
            f"DemoBank(n_demos={len(self.demos)}, "
            f"n_tasks={len(set(d['task_id'] for d in self.demos))}, "
            f"n_skills={len(skills)})"
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save demo bank to JSON."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "skillsbench_root": str(self.skillsbench_root),
            "n_demos": len(self.demos),
            "demos": self.demos,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"DemoBank saved to {out_path} ({len(self.demos)} demos).")

    @classmethod
    def load(cls, path: str, skillsbench_root: Optional[str] = None) -> "DemoBank":
        """Load demo bank from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        root = skillsbench_root or data.get("skillsbench_root", str(SKILLSBENCH_ROOT))
        bank = cls(skillsbench_root=root)
        bank.demos = data.get("demos", [])
        print(f"DemoBank loaded from {path} ({len(bank.demos)} demos).")
        return bank


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build SPLICE demonstration bank")
    parser.add_argument(
        "--skillsbench_root",
        default=str(SKILLSBENCH_ROOT),
        help="Path to SkillsBench root directory",
    )
    parser.add_argument(
        "--output",
        default="data/demo_bank.json",
        help="Output JSON path (relative to splice/)",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=200,
        help="Maximum number of tasks to process",
    )
    parser.add_argument(
        "--task_ids",
        nargs="*",
        help="Specific task IDs to include (default: all)",
    )
    args = parser.parse_args()

    bank = DemoBank(skillsbench_root=args.skillsbench_root)
    bank.build_from_tasks(
        task_ids=args.task_ids,
        max_tasks=args.max_tasks,
        verbose=True,
    )

    output_path = Path(__file__).parent / args.output
    bank.save(str(output_path))
    print(bank)


if __name__ == "__main__":
    main()
