"""
baselines/random_kshot.py — Random k-shot baseline.

Picks k random demos from train split, formats as few-shot examples.
"""

import random
from typing import List, Optional

from eval_pipeline.datasets.base import EvalSample
from eval_pipeline.llm_client import LLMClient
from .base import Baseline
from .zero_shot import _format_choices, MATH_DATASETS, SKILL_DATASETS, SCRIPT_ONLY_DATASETS


def _format_demo(demo: EvalSample, dataset_name: str = "", idx: int = 0) -> str:
    """Format a single demo as a few-shot example."""
    parts = [f"Example {idx + 1}:"]

    if demo.context and dataset_name not in SKILL_DATASETS:
        # Truncate context for demos to keep prompt manageable
        ctx_preview = demo.context[:500].strip()
        if len(demo.context) > 500:
            ctx_preview += "..."
        parts.append(f"Context: {ctx_preview}")

    parts.append(f"Question: {demo.question}")

    if demo.choices:
        parts.append(f"Choices:\n{_format_choices(demo.choices)}")

    parts.append(f"Answer: {demo.answer}")

    return "\n".join(parts)


class RandomKShotBaseline(Baseline):
    """
    Random k-shot baseline.

    Selects k random demonstrations from a pool of training samples
    and prepends them as few-shot examples before the test question.

    Args:
        llm_client: LLM client to use.
        k: Number of few-shot demonstrations (default 3).
        seed: Random seed for reproducibility.
        dataset_name: Name of dataset being evaluated.
        train_samples: Pool of training samples to draw demos from.
        max_tokens: Max tokens for LLM response.
        temperature: LLM temperature.
    """

    name = "random_kshot"

    @staticmethod
    def _extract_demo_skills(samples: List[EvalSample]) -> List[str]:
        skills = []
        for sample in samples:
            names = sample.metadata.get("skill_names", [])
            if isinstance(names, str):
                names = [names]
            if not names:
                skill_name = sample.metadata.get("skill_name")
                if skill_name:
                    names = [skill_name]
            for name in names:
                if name and name not in skills:
                    skills.append(name)
        return skills

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        k: int = 3,
        seed: int = 42,
        dataset_name: str = "",
        train_samples: Optional[List[EvalSample]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        self.llm = llm_client or LLMClient()
        self.k = k
        self.seed = seed
        self.dataset_name = dataset_name
        self.train_samples = train_samples or []
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Pre-select k random demos if train samples are available
        self._rng = random.Random(seed)
        self._selected_demos: List[EvalSample] = []
        if self.train_samples:
            n = min(self.k, len(self.train_samples))
            self._selected_demos = self._rng.sample(self.train_samples, n)

    def _build_system_prompt(self, sample: EvalSample) -> str:
        if self.dataset_name in MATH_DATASETS:
            return (
                "You are an expert mathematician. "
                "Learn from the examples below and solve the new problem step by step."
            )
        if self.dataset_name in SKILL_DATASETS:
            prompt = (
                "You are an expert at executing skill-based tasks. "
                "Learn from the examples and apply the same approach to new tasks."
            )
            if sample.context:
                prompt += f"\n\nAvailable Skills:\n{sample.context}"
            if self.dataset_name in SCRIPT_ONLY_DATASETS:
                prompt += (
                    "\n\nReturn only a complete executable bash script. "
                    "No markdown fences. No explanations."
                )
            return prompt
        return (
            "You are a helpful, accurate assistant. "
            "Learn from the examples below and answer the new question."
        )

    def _build_user_prompt(
        self, sample: EvalSample, demos: List[EvalSample]
    ) -> str:
        """Build few-shot user prompt."""
        parts = []

        # Add few-shot examples
        if demos:
            parts.append("Here are some examples:\n")
            for i, demo in enumerate(demos):
                parts.append(_format_demo(demo, self.dataset_name, i))
                parts.append("")  # blank line between demos

        # Separator
        parts.append("Now solve this new problem:\n")

        # Add context for the test question
        if sample.context and self.dataset_name not in SKILL_DATASETS:
            ctx_str = sample.context[:1000]
            if len(sample.context) > 1000:
                ctx_str += "..."
            parts.append(f"Context: {ctx_str}")

        parts.append(f"Question: {sample.question}")

        if sample.choices:
            parts.append(f"Choices:\n{_format_choices(sample.choices)}")

        # Task-specific instruction
        if self.dataset_name in MATH_DATASETS:
            parts.append("\nLet's think step by step.")
            if sample.choices:
                parts.append(
                    "After reasoning, provide your final answer as just the letter."
                )
            else:
                parts.append("Provide your final numeric answer after '####'.")
        elif self.dataset_name == "strategyqa":
            parts.append("Answer with 'Yes' or 'No'.")
        elif self.dataset_name in SCRIPT_ONLY_DATASETS:
            parts.append(
                "Return only the final executable bash script. "
                "The first line must be '#!/usr/bin/env bash' or '#!/bin/bash'. "
                "Do not include markdown fences or explanatory text."
            )

        parts.append("Answer:")

        return "\n".join(parts)

    def predict(self, sample: EvalSample, demos: Optional[List[EvalSample]] = None) -> str:
        """Generate a k-shot prediction."""
        # Use provided demos, or fall back to pre-selected ones
        effective_demos = demos if demos is not None else self._selected_demos

        # Filter out the test sample itself from demos (avoid data leakage)
        effective_demos = [d for d in effective_demos if d.id != sample.id]

        # Limit to k
        effective_demos = effective_demos[: self.k]
        self.set_last_trace(
            used_skills=self._extract_demo_skills(effective_demos),
            demo_ids=[d.id for d in effective_demos],
            prompt_mode="random_kshot",
        )

        system_prompt = self._build_system_prompt(sample)
        user_prompt = self._build_user_prompt(sample, effective_demos)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prediction = self.llm.chat(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return self.finalize_prediction(sample, system_prompt, user_prompt, prediction)

    def set_train_samples(self, samples: List[EvalSample]) -> None:
        """Update the pool of training samples and re-sample demos."""
        self.train_samples = samples
        self._rng = random.Random(self.seed)
        n = min(self.k, len(samples))
        self._selected_demos = self._rng.sample(samples, n) if samples else []
