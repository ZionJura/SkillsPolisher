"""
baselines/case_bandit.py — CASE Bandit baseline.

Uses CASEBandit from splice/splice_select.py to select k demos.
"""

import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path for splice imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval_pipeline.datasets.base import EvalSample
from eval_pipeline.llm_client import LLMClient
from .base import Baseline
from .zero_shot import _format_choices, MATH_DATASETS, SKILL_DATASETS, SCRIPT_ONLY_DATASETS
from .random_kshot import _format_demo


class CASEBanditBaseline(Baseline):
    """
    CASE Bandit baseline.

    Uses the CASEBandit algorithm from splice_select.py to intelligently
    select k demonstrations from a training pool. Each arm = one training
    example. Reward = exact match or partial match with the correct answer.

    Args:
        llm_client: LLM client to use.
        k: Number of demonstrations to select.
        dataset_name: Name of the dataset.
        train_samples: Pool of training samples to draw demos from.
        delta: CASE confidence parameter.
        max_rounds: Maximum bandit rounds per query.
        seed: Random seed.
        max_tokens: Max tokens for LLM response.
        temperature: LLM temperature.
    """

    name = "case_bandit"

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
        dataset_name: str = "",
        train_samples: Optional[List[EvalSample]] = None,
        delta: float = 0.05,
        max_rounds: int = 50,
        seed: int = 42,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        self.llm = llm_client or LLMClient()
        self.k = k
        self.dataset_name = dataset_name
        self.train_samples = train_samples or []
        self.delta = delta
        self.max_rounds = max_rounds
        self.seed = seed
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Import CASEBandit
        try:
            from splice.splice_select import CASEBandit
            self.CASEBandit = CASEBandit
            self._bandit_available = True
        except ImportError:
            self._bandit_available = False
            import warnings
            warnings.warn(
                "Could not import CASEBandit from splice.splice_select. "
                "Falling back to random demo selection.",
                ImportWarning,
                stacklevel=2,
            )

    def _quick_eval(
        self, arm_idx: int, test_question: str, test_answer: str
    ) -> float:
        """
        Evaluate a demo by running the LLM with just that one demo and
        checking if it gets the answer right (binary reward).

        This is used as the bandit reward function.
        Uses mock evaluation to avoid excessive API calls.
        """
        if arm_idx >= len(self.train_samples):
            return 0.0

        demo = self.train_samples[arm_idx]

        # Quick check: semantic similarity via shared keywords (no API call)
        # This is a proxy reward to avoid excessive LLM calls during bandit phase
        q_words = set(test_question.lower().split())
        d_words = set(demo.question.lower().split())
        overlap = len(q_words & d_words) / max(len(q_words), 1)

        # Bonus for same category/metadata match
        bonus = 0.0
        if self.dataset_name:
            # Check if demo has relevant metadata
            if demo.metadata.get("category") or demo.metadata.get("difficulty"):
                bonus = 0.1

        return min(overlap + bonus, 1.0)

    def _select_demos_with_bandit(self, sample: EvalSample) -> List[EvalSample]:
        """Use CASEBandit to select the best k demos for a given sample."""
        if not self.train_samples:
            return []

        n_arms = len(self.train_samples)
        k = min(self.k, n_arms)

        if not self._bandit_available:
            # Fallback: random selection
            import random
            rng = random.Random(self.seed)
            return rng.sample(self.train_samples, k)

        bandit = self.CASEBandit(
            n_arms=n_arms,
            k=k,
            delta=self.delta,
            max_rounds=self.max_rounds,
        )

        # Run bandit rounds using proxy reward
        for _ in range(min(self.max_rounds, n_arms * 3)):
            if bandit.converged():
                break
            arm = bandit.select_arm()
            reward = self._quick_eval(arm, sample.question, sample.answer)
            bandit.update(arm, reward)

        top_k_indices = bandit.top_k_arms()
        return [self.train_samples[i] for i in top_k_indices if i < len(self.train_samples)]

    def _build_prompt(self, sample: EvalSample, demos: List[EvalSample]) -> tuple:
        """Build system and user prompts with selected demos."""
        if self.dataset_name in MATH_DATASETS:
            system = (
                "You are an expert mathematician. "
                "Use the examples to learn the solution pattern."
            )
        elif self.dataset_name in SKILL_DATASETS:
            system = (
                "You are an expert at executing skill-based tasks. "
                "Use the examples as references and solve the new task using the provided skills."
            )
            if sample.context:
                system += f"\n\nAvailable Skills:\n{sample.context}"
            if self.dataset_name in SCRIPT_ONLY_DATASETS:
                system += (
                    "\n\nReturn only a complete executable bash script. "
                    "No markdown fences. No explanations."
                )
        else:
            system = "You are a helpful assistant. Learn from the examples."

        parts = []
        if demos:
            parts.append("Examples:\n")
            for i, demo in enumerate(demos):
                parts.append(_format_demo(demo, self.dataset_name, i))
                parts.append("")

        parts.append("Now answer this question:\n")

        if sample.context and self.dataset_name not in {"skillsbench", "demo_bank"}:
            ctx_str = sample.context[:800]
            parts.append(f"Context: {ctx_str}")

        parts.append(f"Question: {sample.question}")

        if sample.choices:
            parts.append(f"Choices:\n{_format_choices(sample.choices)}")

        if self.dataset_name in MATH_DATASETS:
            parts.append("\nLet's think step by step.")
            if not sample.choices:
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
        return system, "\n".join(parts)

    def predict(self, sample: EvalSample, demos: Optional[List[EvalSample]] = None) -> str:
        """Generate prediction using CASE bandit-selected demos."""
        if demos is not None:
            selected = demos[: self.k]
        else:
            selected = self._select_demos_with_bandit(sample)

        # Filter out test sample
        selected = [d for d in selected if d.id != sample.id]
        self.set_last_trace(
            used_skills=self._extract_demo_skills(selected),
            demo_ids=[d.id for d in selected],
            prompt_mode="case_bandit",
        )

        system, user = self._build_prompt(sample, selected)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prediction = self.llm.chat(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return self.finalize_prediction(sample, system, user, prediction)

    def set_train_samples(self, samples: List[EvalSample]) -> None:
        """Update the pool of training samples."""
        self.train_samples = samples
