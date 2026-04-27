"""
baselines/bpo_rewrite.py — BPO Rewrite baseline.

Uses SkillPromptRewriter from splice/splice_rewrite.py to rewrite
the question before sending to LLM.
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


class BPORewriteBaseline(Baseline):
    """
    BPO Rewrite baseline.

    Uses SkillPromptRewriter to rewrite/optimize the question prompt
    before sending it to the LLM. This tests whether BPO-style prompt
    optimization improves performance.

    Args:
        llm_client: LLM client to use.
        rewrite_mode: Mode for SkillPromptRewriter ('hf', 'openai', 'claude', 'mock', 'auto').
        dataset_name: Name of the dataset.
        train_samples: Pool for demo selection (used as rewrite context).
        k_demos: Number of demos to use as rewrite context.
        max_tokens: Max tokens for LLM response.
        temperature: LLM temperature.
    """

    name = "bpo_rewrite"

    @staticmethod
    def _merge_skills(sample: EvalSample, demo_dicts: List[dict]) -> List[str]:
        skills = []
        sample_skills = sample.metadata.get("skill_names", [])
        if isinstance(sample_skills, str):
            sample_skills = [sample_skills]
        if not sample_skills:
            skill_name = sample.metadata.get("skill_name")
            if skill_name:
                sample_skills = [skill_name]
        for name in sample_skills:
            if name and name not in skills:
                skills.append(name)
        for demo in demo_dicts:
            name = demo.get("skill_name")
            if name and name not in skills:
                skills.append(name)
        return skills

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rewrite_mode: str = "mock",
        dataset_name: str = "",
        train_samples: Optional[List[EvalSample]] = None,
        k_demos: int = 3,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        self.llm = llm_client or LLMClient()
        self.rewrite_mode = rewrite_mode
        self.dataset_name = dataset_name
        self.train_samples = train_samples or []
        self.k_demos = k_demos
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._rewriter = None

    def _get_rewriter(self):
        """Lazy-initialize the SkillPromptRewriter."""
        if self._rewriter is None:
            try:
                from splice.splice_rewrite import SkillPromptRewriter
                self._rewriter = SkillPromptRewriter(mode=self.rewrite_mode)
            except ImportError as e:
                raise ImportError(
                    f"Could not import SkillPromptRewriter from splice.splice_rewrite: {e}. "
                    f"Ensure the project root is in PYTHONPATH."
                ) from e
        return self._rewriter

    def _sample_to_demo_dict(self, sample: EvalSample) -> dict:
        """Convert EvalSample to demo dict format expected by SkillPromptRewriter."""
        raw_demo = sample.metadata.get("raw_demo")
        if isinstance(raw_demo, dict):
            return {
                "instruction": str(raw_demo.get("instruction", sample.question)),
                "skill_name": str(
                    raw_demo.get(
                        "skill_name",
                        sample.metadata.get("skill_name", self.dataset_name),
                    )
                ),
                "skill_body": str(raw_demo.get("skill_body", sample.context or "")),
                "invocation": str(raw_demo.get("invocation", sample.answer or "")),
                "outcome": str(raw_demo.get("outcome", "")),
            }
        return {
            "instruction": sample.question,
            "skill_name": sample.metadata.get("skill_name", self.dataset_name),
            "skill_body": sample.context[:500] if sample.context else "",
            "invocation": sample.answer if sample.answer else "",
            "outcome": sample.metadata.get("outcome", ""),
        }

    def _get_demo_dicts(self, sample: EvalSample) -> List[dict]:
        """Get demo dicts for rewriting context."""
        if not self.train_samples:
            return []
        # Select demos that are different from the test sample
        pool = [s for s in self.train_samples if s.id != sample.id]
        demos = pool[: self.k_demos]
        return [self._sample_to_demo_dict(d) for d in demos]

    def _build_system_prompt(self, rewritten_question: str, sample: EvalSample) -> str:
        """Build system prompt."""
        if self.dataset_name in SKILL_DATASETS and sample.context:
            prompt = (
                "You are an expert at executing skill-based tasks.\n\n"
                f"Available Skills:\n{sample.context}"
            )
            if self.dataset_name in SCRIPT_ONLY_DATASETS:
                prompt += (
                    "\n\nReturn only a complete executable bash script. "
                    "No markdown fences. No explanations."
                )
            return prompt
        if self.dataset_name in MATH_DATASETS:
            return (
                "You are an expert mathematician. Solve problems step by step."
            )
        return "You are a helpful, accurate assistant."

    def _build_user_prompt(self, rewritten_question: str, sample: EvalSample) -> str:
        """Build user prompt using rewritten question."""
        parts = []

        if sample.context and self.dataset_name not in SKILL_DATASETS:
            ctx_str = sample.context[:800]
            parts.append(f"Context:\n{ctx_str}\n")

        parts.append(rewritten_question)

        if sample.choices:
            parts.append(f"\nChoices:\n{_format_choices(sample.choices)}")

        if self.dataset_name in MATH_DATASETS:
            parts.append("\nLet's think step by step.")
            if sample.choices:
                parts.append(
                    "After reasoning, provide your final answer as just the letter."
                )
            else:
                parts.append("Provide your final numeric answer after '####'.")
        elif self.dataset_name == "strategyqa":
            parts.append("\nAnswer with 'Yes' or 'No'.")
        elif self.dataset_name in SCRIPT_ONLY_DATASETS:
            parts.append(
                "\nReturn only the final executable bash script. "
                "The first line must be '#!/usr/bin/env bash' or '#!/bin/bash'. "
                "Do not include markdown fences or explanatory text."
            )

        return "\n".join(parts)

    def predict(self, sample: EvalSample, demos: Optional[List[EvalSample]] = None) -> str:
        """Generate prediction using BPO-rewritten question."""
        rewriter = self._get_rewriter()

        # Build the skill prompt to rewrite
        skill_prompt = sample.question

        # Get demo dicts for rewrite context
        if demos:
            demo_dicts = [self._sample_to_demo_dict(d) for d in demos[: self.k_demos]]
            demo_ids = [d.id for d in demos[: self.k_demos]]
        else:
            demo_dicts = self._get_demo_dicts(sample)
            pool = [s for s in self.train_samples if s.id != sample.id]
            demo_ids = [d.id for d in pool[: self.k_demos]]

        # Rewrite the prompt
        try:
            rewritten = rewriter.rewrite(
                skill_prompt=skill_prompt,
                selected_demos=demo_dicts,
            )
        except Exception as e:
            # If rewriting fails, fall back to original question
            rewritten = skill_prompt

        self.set_last_trace(
            used_skills=self._merge_skills(sample, demo_dicts),
            demo_ids=demo_ids,
            prompt_mode="bpo_rewrite",
            rewritten_question=rewritten,
        )

        # Send rewritten question to LLM
        system = self._build_system_prompt(rewritten, sample)
        user = self._build_user_prompt(rewritten, sample)

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
