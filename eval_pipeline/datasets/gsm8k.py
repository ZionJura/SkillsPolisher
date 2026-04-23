"""
datasets/gsm8k.py — GSM8K dataset loader.

JSONL fields: question, answer (ends with "#### <number>")
"""

import json
import os
import re
from typing import List, Optional

from .base import EvalDataset, EvalSample
from .data_utils import resolve_data_path

SPLIT_FILES = {
    "train": "gsm8k_train.jsonl",
    "test": "gsm8k_test.jsonl",
    "dev": "gsm8k_test.jsonl",  # fallback
}


def extract_gsm8k_answer(answer_text: str) -> str:
    """
    Extract numeric answer from GSM8K answer field.
    Answers end with '#### <number>' pattern.

    Returns the number as a string, or the full answer if pattern not found.
    """
    match = re.search(r"####\s*([\-\d,\.]+)", answer_text)
    if match:
        # Remove commas from numbers like 1,234
        num_str = match.group(1).replace(",", "").strip()
        return num_str
    # Fallback: return entire answer stripped
    return answer_text.strip()


class GSM8KDataset(EvalDataset):
    """GSM8K grade-school math dataset."""

    name = "gsm8k"

    def __init__(self, split: str = "test", data_root: Optional[str] = None):
        self.split = split
        self.data_root = str(data_root or resolve_data_path("gsm8k"))

        split_key = split if split in SPLIT_FILES else "test"
        filename = SPLIT_FILES[split_key]
        self.file_path = os.path.join(self.data_root, filename)

        self._samples: List[EvalSample] = []
        self._load()

    def _load(self) -> None:
        """Load and parse the JSONL file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"GSM8K file not found: {self.file_path}. "
                f"Expected in: {self.data_root}"
            )

        self._samples = []
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        # Skip malformed lines with a warning
                        continue

                    question = record.get("question", "").strip()
                    answer_full = record.get("answer", "").strip()

                    if not question:
                        continue

                    numeric_answer = extract_gsm8k_answer(answer_full)

                    sample = EvalSample(
                        id=f"gsm8k_{self.split}_{line_idx}",
                        question=question,
                        answer=numeric_answer,
                        context="",
                        choices=[],
                        metadata={
                            "chain_of_thought": answer_full,
                            "answer_full": answer_full,
                        },
                    )
                    self._samples.append(sample)
        except (IOError, OSError) as e:
            raise ValueError(f"Error reading GSM8K file {self.file_path}: {e}") from e

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self._samples[idx]
