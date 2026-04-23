"""
datasets/strategyqa.py — StrategyQA dataset loader.

CSV fields: id, question, answer (Yes/No), facts (Python list repr), decomposition (Python list repr)
"""

import ast
import csv
import os
from typing import List, Optional

from .base import EvalDataset, EvalSample

STRATEGYQA_ROOT = (
    "/mnt/d/Code/AI4R/Skills-Learning2/related-works/CASE/Code/"
    "LLM_experiments/data/Strategyqa"
)

SPLIT_FILES = {
    "train": "strategyqa_train.csv",
    "test": "strategyqa_test.csv",
    "dev": "strategyqa_test.csv",  # fallback
}


def _parse_list_repr(list_str: str) -> List[str]:
    """Parse Python list repr string like "['fact1', 'fact2']" into a list."""
    if not list_str or not list_str.strip():
        return []
    try:
        result = ast.literal_eval(list_str.strip())
        if isinstance(result, list):
            return [str(item) for item in result]
        return [str(result)]
    except (ValueError, SyntaxError):
        # Fallback: try to split by comma after stripping brackets
        cleaned = list_str.strip().strip("[]")
        if cleaned:
            return [item.strip().strip("'\"") for item in cleaned.split(",") if item.strip()]
        return []


def _normalize_yes_no(answer: str) -> str:
    """Normalize yes/no answer to lowercase."""
    normalized = answer.strip().lower()
    if normalized in ("yes", "true", "1"):
        return "yes"
    if normalized in ("no", "false", "0"):
        return "no"
    return normalized


class StrategyQADataset(EvalDataset):
    """StrategyQA multi-hop boolean question answering dataset."""

    name = "strategyqa"

    def __init__(self, split: str = "test", data_root: Optional[str] = None):
        self.split = split
        self.data_root = data_root or STRATEGYQA_ROOT

        split_key = split if split in SPLIT_FILES else "test"
        filename = SPLIT_FILES[split_key]
        self.file_path = os.path.join(self.data_root, filename)

        self._samples: List[EvalSample] = []
        self._load()

    def _load(self) -> None:
        """Load and parse the CSV file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"StrategyQA file not found: {self.file_path}. "
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

                    record_id = row.get("id", f"strategyqa_{self.split}_{row_idx}").strip()
                    answer_raw = row.get("answer", "").strip()
                    facts_raw = row.get("facts", "")
                    decomposition_raw = row.get("decomposition", "")

                    answer = _normalize_yes_no(answer_raw)
                    facts = _parse_list_repr(facts_raw)
                    decomposition = _parse_list_repr(decomposition_raw)

                    sample = EvalSample(
                        id=record_id or f"strategyqa_{self.split}_{row_idx}",
                        question=question,
                        answer=answer,
                        context="",
                        choices=["Yes", "No"],
                        metadata={
                            "facts": facts,
                            "decomposition": decomposition,
                            "answer_raw": answer_raw,
                        },
                    )
                    self._samples.append(sample)
        except (csv.Error, UnicodeDecodeError) as e:
            raise ValueError(f"Error reading StrategyQA CSV {self.file_path}: {e}") from e

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self._samples[idx]
