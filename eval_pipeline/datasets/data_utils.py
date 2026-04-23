"""
data_utils.py — Centralized dataset path resolver.

Resolution order for each dataset:
  1. eval_pipeline/datasets/data/<name>/      (canonical, after running download_datasets.py)
  2. related-works/<legacy-path>/             (backward compat for existing related-works checkout)
  3. Raises DataNotFoundError with download instructions
"""

from pathlib import Path
from typing import Union

# ── Project root detection ────────────────────────────────────────────────────
# This file lives at: <project_root>/eval_pipeline/datasets/data_utils.py
# So: parent = datasets/, parent.parent = eval_pipeline/, parent.parent.parent = project root
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent.parent  # .../Skills-Learning2/

# Canonical data home: eval_pipeline/datasets/data/
DATA_DIR = _THIS_FILE.parent / "data"

# ── Legacy fallback paths (relative to project root) ─────────────────────────
FALLBACK_PATHS = {
    "aqua_rat":    ["related-works/CASE/Code/LLM_experiments/data/AQUA_RAT"],
    "gsm8k":       ["related-works/CASE/Code/LLM_experiments/data/GSM8K"],
    "tabmwp":      ["related-works/CASE/Code/LLM_experiments/data/tabmwp"],
    "finqa":       ["related-works/CASE/Code/LLM_experiments/data/FinQA"],
    "strategyqa":  ["related-works/CASE/Code/LLM_experiments/data/Strategyqa"],
    "bpo":         ["related-works/BPO/data/testset"],
    "skillsbench": ["related-works/skillsbench/tasks"],
    "demo_bank":   ["splice/data/demo_bank.json"],
}


class DataNotFoundError(FileNotFoundError):
    """Raised when a dataset cannot be found in any of the expected locations."""

    def __init__(self, dataset_name: str, tried_paths: list):
        tried = "\n  ".join(str(p) for p in tried_paths)
        msg = (
            f"Dataset '{dataset_name}' not found. Tried:\n  {tried}\n\n"
            f"Run:  python scripts/download_datasets.py --dataset {dataset_name}\n"
            f"Or:   python scripts/download_datasets.py --all"
        )
        super().__init__(msg)
        self.dataset_name = dataset_name
        self.tried_paths = tried_paths


def resolve_data_path(dataset_name: str) -> Path:
    """
    Resolve the data directory (or file) for a named dataset.

    Resolution order:
      1. eval_pipeline/datasets/data/<name>/   (canonical download location)
      2. related-works/<legacy-path>/          (backward compat)
      3. Raises DataNotFoundError

    Returns a Path that exists.
    """
    tried: list = []

    # 1. Canonical location
    canonical = DATA_DIR / dataset_name
    tried.append(canonical)
    if canonical.exists():
        return canonical

    # 2. Legacy fallback paths
    for rel in FALLBACK_PATHS.get(dataset_name, []):
        candidate = PROJECT_ROOT / rel
        tried.append(candidate)
        if candidate.exists():
            return candidate

    raise DataNotFoundError(dataset_name, tried)


def get_data_dir(dataset_name: str) -> Path:
    """
    Return the canonical data directory for a dataset, creating it if needed.

    Used by download scripts to get the target directory before data exists.
    Unlike resolve_data_path(), this always returns the canonical path and
    creates the directory if it does not yet exist.
    """
    canonical = DATA_DIR / dataset_name
    canonical.mkdir(parents=True, exist_ok=True)
    return canonical
