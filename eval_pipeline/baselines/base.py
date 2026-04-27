"""
baselines/base.py — Abstract base class for eval baselines.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional

from eval_pipeline.datasets.base import EvalSample


class Baseline(ABC):
    """Abstract base class for all evaluation baselines."""

    name: str = "base"
    last_trace: dict = {}

    @abstractmethod
    def predict(self, sample: EvalSample, demos: Optional[List[EvalSample]] = None) -> str:
        """
        Generate a prediction for a single sample.

        Args:
            sample: The evaluation sample to predict on.
            demos: Optional list of few-shot demonstration samples.

        Returns:
            Prediction string.
        """
        raise NotImplementedError

    def batch_predict(
        self,
        samples: List[EvalSample],
        demos: Optional[List[EvalSample]] = None,
        verbose: bool = False,
    ) -> List[str]:
        """
        Generate predictions for a batch of samples.

        Args:
            samples: List of evaluation samples.
            demos: Optional shared few-shot demonstrations.
            verbose: If True, print progress.

        Returns:
            List of prediction strings.
        """
        predictions = []
        for i, sample in enumerate(samples):
            if verbose and (i % 10 == 0 or i == len(samples) - 1):
                print(f"  [{self.name}] {i + 1}/{len(samples)}: {sample.id[:40]}...")
            try:
                pred = self.predict(sample, demos=demos)
            except ConnectionError as e:
                raise  # propagate network errors
            except Exception as e:
                print(f"  [warn] Prediction failed for {sample.id}: {e}")
                pred = ""
            predictions.append(pred)
        return predictions

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    def set_last_trace(self, **kwargs) -> None:
        """Store structured metadata about the most recent prediction."""
        self.last_trace = dict(kwargs)

    def get_last_trace(self) -> dict:
        """Return metadata about the most recent prediction."""
        return dict(getattr(self, "last_trace", {}) or {})

    def _requires_script_output(self, sample: EvalSample) -> bool:
        return getattr(self, "dataset_name", "") == "skillsbench"

    def _script_output_instruction(self) -> str:
        return (
            "Return only the final executable bash script. "
            "The first line must be '#!/usr/bin/env bash' or '#!/bin/bash'. "
            "Do not include markdown code fences. "
            "Do not include explanations, planning text, or commentary. "
            "The script must be complete and produce the required output artifacts."
        )

    def _extract_script_candidate(self, prediction: str) -> str:
        text = (prediction or "").strip()
        if not text:
            return ""

        fence_match = re.search(r"```(?:bash|sh|shell|python)?\s*(.*?)```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        shebang_index = text.find("#!")
        if shebang_index >= 0:
            return text[shebang_index:].strip()

        return text

    def _looks_like_final_script(self, prediction: str) -> bool:
        candidate = self._extract_script_candidate(prediction).lstrip()
        return candidate.startswith("#!/usr/bin/env bash") or candidate.startswith("#!/bin/bash")

    def _needs_script_repair(self, prediction: str) -> bool:
        candidate = self._extract_script_candidate(prediction).strip()
        if not candidate:
            return True
        if not self._looks_like_final_script(candidate):
            return True

        head = candidate[:200].lower()
        planning_phrases = (
            "let me ",
            "i will ",
            "we will ",
            "to solve this",
            "here is the solution",
            "the following steps",
        )
        return any(phrase in head for phrase in planning_phrases)

    def _repair_script_prediction(
        self,
        sample: EvalSample,
        system_prompt: str,
        user_prompt: str,
        prediction: str,
    ) -> str:
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "You convert draft task answers into final executable scripts. "
                    f"{self._script_output_instruction()}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original system prompt:\n{system_prompt}\n\n"
                    f"Original task prompt:\n{user_prompt}\n\n"
                    f"Available skills:\n{sample.context or '(none)'}\n\n"
                    f"Draft answer:\n{prediction}\n\n"
                    "Rewrite the draft into one complete final bash script only."
                ),
            },
        ]
        return self.llm.chat(
            messages=repair_messages,
            max_tokens=getattr(self, "max_tokens", 12048),
            temperature=0.0,
        )

    def finalize_prediction(
        self,
        sample: EvalSample,
        system_prompt: str,
        user_prompt: str,
        prediction: str,
    ) -> str:
        if not self._requires_script_output(sample):
            return prediction

        normalized = self._extract_script_candidate(prediction).strip()
        trace = self.get_last_trace()
        trace["normalized_from_code_block"] = normalized != (prediction or "").strip()

        if not self._needs_script_repair(prediction):
            trace["script_repaired"] = False
            trace["final_output_format"] = "script"
            self.set_last_trace(**trace)
            return normalized

        repaired = self._repair_script_prediction(sample, system_prompt, user_prompt, prediction)
        repaired_normalized = self._extract_script_candidate(repaired).strip()
        trace["script_repaired"] = True
        trace["final_output_format"] = (
            "script" if self._looks_like_final_script(repaired_normalized) else "non_script"
        )
        self.set_last_trace(**trace)
        return repaired_normalized or normalized or prediction
