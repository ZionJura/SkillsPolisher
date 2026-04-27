"""
evaluators/skillsbench.py — Specialized evaluator for SkillsBench script tasks.
"""

import re
from typing import Optional

from eval_pipeline.datasets.base import EvalSample
from eval_pipeline.llm_client import LLMClient
from .base import Evaluator
from .llm_eval import _extract_yes_no

SKILLSBENCH_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for script-based benchmark tasks.
Decide whether the submitted script would correctly solve the task.
Be strict about missing required logic, files, and outputs, but do not penalize harmless implementation differences.
For long scripts, focus on whether the script covers the required end-to-end task logic and produces the requested outputs.
Always answer with exactly "Yes" or "No" followed by one concise sentence."""

SKILLSBENCH_JUDGE_TEMPLATE = """Task Instruction:
{question}

Reference Solution Script:
{reference_answer}

Submitted Script:
{model_response}

Does the submitted script correctly solve the task?
Judge functional correctness, required outputs, and task completeness.
Respond with "Yes" or "No" followed by one concise sentence."""


def _normalize_script(text: str) -> str:
    lines = [line.rstrip() for line in (text or "").strip().splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _truncate_keep_ends(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars < 200:
        return text[:max_chars]
    half = (max_chars - 32) // 2
    return text[:half] + "\n\n... [truncated] ...\n\n" + text[-half:]


class SkillsBenchEvaluator(Evaluator):
    """
    LLM-based evaluator specialized for long script outputs in SkillsBench.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_tokens: int = 200,
        temperature: float = 0.0,
        truncate_context: int = 12000,
    ):
        self.llm = llm_client or LLMClient()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.truncate_context = truncate_context

    def score(self, prediction: str, sample: EvalSample) -> dict:
        if not prediction or not prediction.strip():
            return {
                "correct": False,
                "score": 0.0,
                "details": "empty_prediction",
            }

        normalized_prediction = _normalize_script(prediction)
        normalized_reference = _normalize_script(sample.answer)
        if normalized_prediction == normalized_reference:
            return {
                "correct": True,
                "score": 1.0,
                "details": "exact_script_match",
            }

        question = _truncate_keep_ends(sample.question, self.truncate_context)
        reference = _truncate_keep_ends(sample.answer or "(no reference)", self.truncate_context)
        response = _truncate_keep_ends(prediction, self.truncate_context)

        judge_prompt = SKILLSBENCH_JUDGE_TEMPLATE.format(
            question=question,
            reference_answer=reference,
            model_response=response,
        )

        messages = [
            {"role": "system", "content": SKILLSBENCH_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": judge_prompt},
        ]

        try:
            judge_response = self.llm.chat(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except ConnectionError:
            raise
        except Exception as e:
            return {
                "correct": False,
                "score": 0.0,
                "details": f"judge_error: {e}",
            }

        judgment = _extract_yes_no(judge_response)
        if judgment is True:
            return {
                "correct": True,
                "score": 1.0,
                "details": f"judge_yes: {judge_response[:160].strip()}",
            }
        if judgment is False:
            return {
                "correct": False,
                "score": 0.0,
                "details": f"judge_no: {judge_response[:160].strip()}",
            }
        return {
            "correct": False,
            "score": 0.0,
            "details": f"judge_ambiguous: {judge_response[:160].strip()}",
        }
