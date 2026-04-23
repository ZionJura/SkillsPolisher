"""
evaluators/__init__.py — Evaluator registry.
"""

from .base import Evaluator
from .exact_match import ExactMatchEvaluator
from .llm_eval import LLMEvaluator

# Datasets that use LLM-based evaluation
LLM_EVAL_DATASETS = {"bpo_test", "dolly_eval", "self_instruct_eval", "skillsbench"}
# Datasets that use exact match evaluation
EXACT_MATCH_DATASETS = {"aqua_rat", "gsm8k", "tabmwp", "finqa", "strategyqa", "demo_bank"}


def get_evaluator(dataset_name: str, llm_client=None):
    """
    Get the appropriate evaluator for a dataset.

    Args:
        dataset_name: Name of the dataset.
        llm_client: Optional LLM client (needed for LLM evaluator).

    Returns:
        Evaluator instance.
    """
    if dataset_name in LLM_EVAL_DATASETS:
        return LLMEvaluator(llm_client=llm_client)
    return ExactMatchEvaluator(dataset_name=dataset_name)


__all__ = [
    "Evaluator",
    "ExactMatchEvaluator",
    "LLMEvaluator",
    "LLM_EVAL_DATASETS",
    "EXACT_MATCH_DATASETS",
    "get_evaluator",
]
