"""
datasets/tabmwp.py — TabMWP dataset loader.

JSON dict keyed by string IDs.
Fields: question, choices, answer, table, table_title, solution,
        row_num, column_num, unit, ques_type, ans_type
"""

import json
import os
from typing import List, Optional

from .base import EvalDataset, EvalSample

TABMWP_ROOT = (
    "/mnt/d/Code/AI4R/Skills-Learning2/related-works/CASE/Code/"
    "LLM_experiments/data/tabmwp"
)

SPLIT_FILES = {
    "train": "problems_train.json",
    "test": "problems_test.json",
    "dev": "problems_test.json",  # fallback
}


def _table_to_markdown(table_str: str, title: str = "") -> str:
    """
    Convert a table string to Markdown format for context.

    The table in TabMWP is typically a pipe-separated or CSV-like string.
    We do a best-effort conversion.
    """
    if not table_str or not table_str.strip():
        return ""

    lines = [line.strip() for line in table_str.strip().split("\n") if line.strip()]
    if not lines:
        return ""

    # Build markdown table
    md_lines = []
    if title:
        md_lines.append(f"**{title}**")
        md_lines.append("")

    for i, line in enumerate(lines):
        # Check if it uses | separators
        if "|" in line:
            md_lines.append(line)
        else:
            # Try comma-separated
            parts = [p.strip() for p in line.split(",")]
            md_lines.append("| " + " | ".join(parts) + " |")

        # Add header separator after first row
        if i == 0 and len(lines) > 1:
            if "|" in line:
                cols = line.count("|") - 1
                if cols > 0:
                    md_lines.append("|" + "---|" * cols)
            else:
                parts = [p.strip() for p in line.split(",")]
                md_lines.append("| " + " | ".join(["---"] * len(parts)) + " |")

    return "\n".join(md_lines)


class TabMWPDataset(EvalDataset):
    """TabMWP table-based math word problem dataset."""

    name = "tabmwp"

    def __init__(self, split: str = "test", data_root: Optional[str] = None):
        self.split = split
        self.data_root = data_root or TABMWP_ROOT

        split_key = split if split in SPLIT_FILES else "test"
        filename = SPLIT_FILES[split_key]
        self.file_path = os.path.join(self.data_root, filename)

        self._samples: List[EvalSample] = []
        self._keys: List[str] = []
        self._load()

    def _load(self) -> None:
        """Load and parse the JSON file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"TabMWP file not found: {self.file_path}. "
                f"Expected in: {self.data_root}"
            )

        self._samples = []
        self._keys = []
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise ValueError(f"Error reading TabMWP file {self.file_path}: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(f"TabMWP file expected a JSON dict, got {type(data).__name__}")

        for problem_id, record in data.items():
            question = record.get("question", "").strip()
            if not question:
                continue

            answer = str(record.get("answer", "")).strip()
            table_str = record.get("table", "")
            table_title = record.get("table_title", "")
            solution = record.get("solution", "").strip()
            unit = record.get("unit", "")
            choices_raw = record.get("choices", None)

            # Format table as Markdown for context
            context = _table_to_markdown(table_str, title=table_title)

            # Parse choices (list or null)
            choices: List[str] = []
            if choices_raw and isinstance(choices_raw, list):
                choices = [str(c) for c in choices_raw]

            # Append unit to answer if present
            if unit and unit not in answer:
                answer_with_unit = f"{answer} {unit}".strip()
            else:
                answer_with_unit = answer

            sample = EvalSample(
                id=f"tabmwp_{self.split}_{problem_id}",
                question=question,
                answer=answer_with_unit,
                context=context,
                choices=choices,
                metadata={
                    "solution": solution,
                    "table_title": table_title,
                    "table_raw": table_str,
                    "row_num": record.get("row_num"),
                    "column_num": record.get("column_num"),
                    "unit": unit,
                    "ques_type": record.get("ques_type", ""),
                    "ans_type": record.get("ans_type", ""),
                    "grade": record.get("grade", ""),
                    "problem_id": problem_id,
                },
            )
            self._samples.append(sample)
            self._keys.append(problem_id)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self._samples[idx]
