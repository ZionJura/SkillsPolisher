"""
datasets/finqa.py — FinQA dataset loader.

CSV fields: question, answer, text, table, id, program
"""

import csv
import os
from typing import List, Optional

from .base import EvalDataset, EvalSample

FINQA_ROOT = (
    "/mnt/d/Code/AI4R/Skills-Learning2/related-works/CASE/Code/"
    "LLM_experiments/data/FinQA"
)

SPLIT_FILES = {
    "train": "finqa_train.csv",
    "test": "finqa_test.csv",
    "dev": "finqa_test.csv",  # fallback
}


class FinQADataset(EvalDataset):
    """FinQA financial question answering dataset."""

    name = "finqa"

    def __init__(self, split: str = "test", data_root: Optional[str] = None):
        self.split = split
        self.data_root = data_root or FINQA_ROOT

        split_key = split if split in SPLIT_FILES else "test"
        filename = SPLIT_FILES[split_key]
        self.file_path = os.path.join(self.data_root, filename)

        self._samples: List[EvalSample] = []
        self._load()

    def _load(self) -> None:
        """Load and parse the CSV file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"FinQA file not found: {self.file_path}. "
                f"Expected in: {self.data_root}"
            )

        self._samples = []
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader):
                    question = row.get("question", "").strip()
                    if not question:
                        continue

                    answer = str(row.get("answer", "")).strip()
                    text = row.get("text", "").strip()
                    table = row.get("table", "").strip()
                    record_id = row.get("id", f"finqa_{self.split}_{row_idx}").strip()
                    program = row.get("program", "").strip()

                    # Combine text and table as context
                    context_parts = []
                    if text:
                        context_parts.append(f"Text:\n{text}")
                    if table:
                        context_parts.append(f"Table:\n{table}")
                    context = "\n\n".join(context_parts)

                    sample = EvalSample(
                        id=record_id or f"finqa_{self.split}_{row_idx}",
                        question=question,
                        answer=answer,
                        context=context,
                        choices=[],
                        metadata={
                            "program": program,
                            "text": text,
                            "table": table,
                        },
                    )
                    self._samples.append(sample)
        except (csv.Error, UnicodeDecodeError) as e:
            raise ValueError(f"Error reading FinQA CSV {self.file_path}: {e}") from e

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self._samples[idx]
