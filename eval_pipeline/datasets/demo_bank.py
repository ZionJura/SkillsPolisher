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

DEMO_BANK_PATH = "/mnt/d/Code/AI4R/Skills-Learning2/splice/data/demo_bank.json"


class DemoBankDataset(EvalDataset):
    """
    Demo Bank dataset: skill invocation demonstrations used by SPLICE.
    """

    name = "demo_bank"

    def __init__(self, split: str = "train", data_path: Optional[str] = None):
        self.split = split
        self.data_path = data_path or DEMO_BANK_PATH
        self._samples: List[EvalSample] = []
        self._raw_demos: List[dict] = []
        self._load()

    def _load(self) -> None:
        """Load and parse the demo bank JSON."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Demo bank file not found: {self.data_path}"
            )

        try:
            with open(self.data_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise ValueError(f"Error reading demo bank {self.data_path}: {e}") from e

        # Handle both list format and dict format with 'demos' key
        if isinstance(data, list):
            demos = data
        elif isinstance(data, dict):
            demos = data.get("demos", [])
        else:
            raise ValueError(f"Unexpected demo bank format: {type(data).__name__}")

        self._raw_demos = demos
        self._samples = []

        for idx, demo in enumerate(demos):
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
