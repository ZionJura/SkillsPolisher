"""
baselines/zero_shot.py — Zero-shot baseline.

Calls LLM with just the question, no demos.
- For math tasks: adds "Let's think step by step."
- For MCQ: formats choices as A) B) C) D)
- For skill tasks: includes skill body in system prompt
"""

from typing import List, Optional

from eval_pipeline.datasets.base import EvalSample
from eval_pipeline.llm_client import LLMClient
from .base import Baseline

# Datasets that are math-oriented (benefit from CoT)
MATH_DATASETS = {"gsm8k", "tabmwp", "finqa", "aqua_rat"}
# Datasets that are skill-oriented
SKILL_DATASETS = {"skillsbench", "demo_bank"}


def _format_choices(choices: List[str]) -> str:
    """Format MCQ choices as A) choice1\nB) choice2\n..."""
    if not choices:
        return ""
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = []
    for i, choice in enumerate(choices):
        label = labels[i] if i < len(labels) else str(i + 1)
        # If choice already starts with a letter like "A)10", just use it
        if len(choice) > 2 and choice[1] in ")." and choice[0].isalpha():
            lines.append(choice)
        else:
            lines.append(f"{label}) {choice}")
    return "\n".join(lines)


class ZeroShotBaseline(Baseline):
    """
    Zero-shot baseline: sends question directly to LLM without few-shot demos.

    Adapts prompt for different task types:
    - Math tasks: adds chain-of-thought prompt suffix
    - MCQ tasks: formats answer choices
    - Skill tasks: includes skill documentation in system prompt
    """

    name = "zero_shot"

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        dataset_name: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        self.llm = llm_client or LLMClient()
        self.dataset_name = dataset_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _build_system_prompt(self, sample: EvalSample) -> str:
        """Build the system prompt based on task type."""
        if self.dataset_name in SKILL_DATASETS or sample.context:
            if sample.context:
                return (
                    "You are an expert at executing skill-based tasks. "
                    "Use the provided skill documentation to answer the question.\n\n"
                    f"Available Skills:\n{sample.context}"
                )
        if self.dataset_name in MATH_DATASETS:
            return (
                "You are an expert mathematician and problem solver. "
                "Show your reasoning step by step and provide a clear final answer."
            )
        return "You are a helpful, accurate assistant. Answer the question concisely."

    def _build_user_prompt(self, sample: EvalSample) -> str:
        """Build the user prompt."""
        parts = []

        # Add context for non-skill datasets (tables, passages)
        if sample.context and self.dataset_name not in SKILL_DATASETS:
            parts.append(f"Context:\n{sample.context}\n")

        # Add question
        parts.append(f"Question: {sample.question}")

        # Format MCQ choices
        if sample.choices:
            choices_str = _format_choices(sample.choices)
            parts.append(f"\nChoices:\n{choices_str}")

        # Add task-specific suffixes
        if self.dataset_name in MATH_DATASETS or (
            sample.choices and self.dataset_name == "aqua_rat"
        ):
            parts.append("\nLet's think step by step.")
            if sample.choices:
                parts.append(
                    "\nAfter reasoning, provide your final answer as just the letter "
                    "(e.g., 'A', 'B', 'C', 'D')."
                )
            else:
                parts.append("\nProvide your final numeric answer after '####'.")
        elif self.dataset_name == "strategyqa":
            parts.append("\nAnswer with 'Yes' or 'No'.")
        elif self.dataset_name in SKILL_DATASETS:
            parts.append(
                "\nProvide a complete solution using the available skills."
            )

        return "\n".join(parts)

    def predict(self, sample: EvalSample, demos: Optional[List[EvalSample]] = None) -> str:
        """Generate a zero-shot prediction."""
        system_prompt = self._build_system_prompt(sample)
        user_prompt = self._build_user_prompt(sample)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.llm.chat(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
