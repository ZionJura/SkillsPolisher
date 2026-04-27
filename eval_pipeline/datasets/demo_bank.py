"""
datasets/demo_bank.py — Demo Bank dataset loader.

JSON with fields: skillsbench_root, n_demos, demos
Demo fields: task_id, instruction, skill_name, skill_body, invocation,
             outcome, category, difficulty, tags
"""

import json
import os
from typing import List, Optional

from .base import EvalDataset, EvalSample
from .data_utils import resolve_data_path


def _resolve_demo_bank_file(data_path: Optional[str] = None) -> str:
    """Resolve demo_bank.json from an explicit path or the standard data dir."""
    if data_path:
        return str(data_path)
    demo_root = resolve_data_path("demo_bank")
    return str(demo_root / "demo_bank.json") if demo_root.is_dir() else str(demo_root)


def load_demo_bank_records(data_path: Optional[str] = None) -> List[dict]:
    """Load raw demo-bank records from disk."""
    resolved_path = _resolve_demo_bank_file(data_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Demo bank file not found: {resolved_path}")

    try:
        with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise ValueError(f"Error reading demo bank {resolved_path}: {e}") from e

    if isinstance(data, list):
        demos = data
    elif isinstance(data, dict):
        demos = data.get("demos", [])
    else:
        raise ValueError(f"Unexpected demo bank format: {type(data).__name__}")

    return demos


def build_skillsbench_demo_pool(data_path: Optional[str] = None) -> List[EvalSample]:
    """
    Convert demo-bank records into a non-leaking SkillsBench demo pool.

    Each demo keeps the original task id so baselines can exclude the
    corresponding evaluation task by matching `sample.id`.
    """
    demos = load_demo_bank_records(data_path)
    samples: List[EvalSample] = []

    for idx, demo in enumerate(demos):
        task_id = str(demo.get("task_id", f"demo_{idx}")).strip()
        instruction = str(demo.get("instruction", "")).strip()
        invocation = str(demo.get("invocation", "")).strip()
        if not task_id or not instruction or not invocation:
            continue

        skill_name = str(demo.get("skill_name", "")).strip()
        skill_body = str(demo.get("skill_body", "")).strip()
        category = str(demo.get("category", "")).strip()
        difficulty = str(demo.get("difficulty", "")).strip()
        tags = demo.get("tags", [])

        samples.append(
            EvalSample(
                id=f"skillsbench_{task_id}",
                question=instruction,
                answer=invocation,
                context=skill_body,
                choices=[],
                metadata={
                    "task_name": task_id,
                    "skill_name": skill_name,
                    "skill_names": [skill_name] if skill_name else [],
                    "skill_body": skill_body,
                    "invocation": invocation,
                    "outcome": str(demo.get("outcome", "")).strip(),
                    "category": category,
                    "difficulty": difficulty,
                    "tags": tags,
                    "demo_idx": idx,
                    "source_dataset": "demo_bank",
                    "raw_demo": demo,
                },
            )
        )

    return samples


class DemoBankDataset(EvalDataset):
    """
    Demo Bank dataset: skill invocation demonstrations used by SPLICE.
    """

    name = "demo_bank"

    def __init__(self, split: str = "train", data_path: Optional[str] = None):
        self.split = split
        self.data_path = _resolve_demo_bank_file(data_path)
        self._samples: List[EvalSample] = []
        self._raw_demos: List[dict] = []
        self._load()

    def _load(self) -> None:
        """Load and parse the demo bank JSON."""
        self._raw_demos = load_demo_bank_records(self.data_path)
        self._samples = []

        for idx, demo in enumerate(self._raw_demos):
            task_id = str(demo.get("task_id", f"demo_{idx}"))
            instruction = demo.get("instruction", "").strip()
            if not instruction:
                continue

            skill_name = demo.get("skill_name", "").strip()
            skill_body = demo.get("skill_body", "").strip()
            invocation = demo.get("invocation", "").strip()
            outcome = str(demo.get("outcome", "")).strip()
            category = demo.get("category", "").strip()
            difficulty = demo.get("difficulty", "").strip()
            tags = demo.get("tags", [])

            # Context = skill body (the skill documentation)
            context = skill_body

            sample = EvalSample(
                id=task_id,
                question=instruction,
                answer=outcome,
                context=context,
                choices=[],
                metadata={
                    "skill_name": skill_name,
                    "skill_body": skill_body,
                    "invocation": invocation,
                    "category": category,
                    "difficulty": difficulty,
                    "tags": tags,
                    "demo_idx": idx,
                },
            )
            self._samples.append(sample)

    def get_raw_demos(self) -> List[dict]:
        """Return the raw demo dicts (for use with SPLICE components)."""
        return self._raw_demos

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self._samples[idx]
