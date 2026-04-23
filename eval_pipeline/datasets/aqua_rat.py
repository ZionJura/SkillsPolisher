"""
datasets/aqua_rat.py — AQUA-RAT dataset loader.

CSV fields: question, options (Python list repr), rationale, correct (letter)
"""

import ast
import csv
import os
from typing import List, Optional

from .base import EvalDataset, EvalSample
from .data_utils import resolve_data_path

SPLIT_FILES = {
    "train": "aquarat_train.csv",
    "dev": "aquarat_dev.csv",
    "test": "aquarat_dev.csv",  # fallback: use dev for test
}


def _parse_options(options_str: str) -> List[str]:
    """Parse Python list repr like "['A)10', 'B)20']" into a list."""
    if not options_str or not options_str.strip():
        return []
    try:
        result = ast.literal_eval(options_str.strip())
        if isinstance(result, list):
            return [str(o) for o in result]
        return [str(result)]
    except (ValueError, SyntaxError):
        # Fallback: try splitting by comma after stripping brackets
        cleaned = options_str.strip().strip("[]")
        if cleaned:
            return [o.strip().strip("'\"") for o in cleaned.split(",") if o.strip()]
        return []


class AquaRatDataset(EvalDataset):
    """AQUA-RAT dataset: algebraic word problems with multiple choice answers."""

    name = "aqua_rat"

    def __init__(self, split: str = "dev", data_root: Optional[str] = None):
        self.split = split
        self.data_root = str(data_root or resolve_data_path("aqua_rat"))

        split_key = split if split in SPLIT_FILES else "dev"
        filename = SPLIT_FILES[split_key]
        self.file_path = os.path.join(self.data_root, filename)

        self._samples: List[EvalSample] = []
        self._load()

    def _load(self) -> None:
        """Load and parse the CSV file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"AQUA-RAT file not found: {self.file_path}. "
                f"Expected in: {self.data_root}"
            )

        self._samples = []
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader):
                    question = row.get("question", "").strip()
                    options_raw = row.get("options", "")
                    rationale = row.get("rationale", "").strip()
                    correct = row.get("correct", "").strip().upper()

                    if not question:
                        continue

                    choices = _parse_options(options_raw)

                    sample = EvalSample(
                        id=f"aqua_{self.split}_{row_idx}",
                        question=question,
                        answer=correct,
                        context="",
                        choices=choices,
                        metadata={
                            "rationale": rationale,
                            "options_raw": options_raw,
                        },
                    )
                    self._samples.append(sample)
        except (csv.Error, UnicodeDecodeError) as e:
            raise ValueError(f"Error reading AQUA-RAT CSV {self.file_path}: {e}") from e

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self._samples[idx]
