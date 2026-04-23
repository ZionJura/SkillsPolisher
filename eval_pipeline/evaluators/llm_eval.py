"""
evaluators/llm_eval.py — LLM-based evaluator for open-ended tasks.

For BPO and SkillsBench where exact match doesn't work.
Calls LLM to judge: "Given question X and answer Y, is response Z correct? Yes/No"
"""

import re
from typing import Optional

from eval_pipeline.datasets.base import EvalSample
from eval_pipeline.llm_client import LLMClient
from .base import Evaluator

JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating AI assistant responses.
Your task is to determine whether a given response correctly answers a question.
Be fair but rigorous. Focus on factual correctness and task completion.
Always respond with exactly "Yes" or "No" followed by a brief explanation."""

JUDGE_TEMPLATE = """Question: {question}

Reference Answer: {reference_answer}

Model Response: {model_response}

Does the model response correctly answer the question? Compare with the reference answer.
Consider partial credit: if the response shows correct reasoning but minor differences, answer "Yes".
Respond with "Yes" or "No" followed by a one-sentence explanation."""


def _extract_yes_no(text: str) -> Optional[bool]:
    """Extract yes/no from LLM judge response."""
    if not text:
        return None
    text_lower = text.lower().strip()

    # Check first word or sentence
    first_line = text_lower.split("\n")[0].strip()
    first_word = text_lower.split()[0] if text_lower.split() else ""

    if first_word in ("yes", "yes,", "yes."):
        return True
    if first_word in ("no", "no,", "no."):
        return False

    # Check for "Yes" or "No" anywhere in first line
    if re.search(r"\byes\b", first_line):
        return True
    if re.search(r"\bno\b", first_line):
        return False

    # Check full text
    yes_count = len(re.findall(r"\byes\b", text_lower))
    no_count = len(re.findall(r"\bno\b", text_lower))

    if yes_count > no_count:
        return True
    if no_count > yes_count:
        return False

    return None


class LLMEvaluator(Evaluator):
    """
    LLM-based evaluator for open-ended tasks.

    Uses an LLM judge to determine whether a model response correctly
    answers a question, compared to a reference answer. Suitable for
    BPO datasets and SkillsBench where exact match is not appropriate.

    Args:
        llm_client: LLM client to use as judge.
        max_tokens: Max tokens for judge response.
        temperature: Temperature for judge (low = more deterministic).
        truncate_context: Max chars of context to include in judge prompt.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_tokens: int = 200,
        temperature: float = 0.0,
        truncate_context: int = 1000,
    ):
        self.llm = llm_client or LLMClient()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.truncate_context = truncate_context

    def score(self, prediction: str, sample: EvalSample) -> dict:
        """
        Score prediction using LLM judge.

        Args:
            prediction: Model's predicted response.
            sample: Original eval sample with question and reference answer.

        Returns:
            dict with 'correct', 'score', 'details'.
        """
        if not prediction or not prediction.strip():
            return {
                "correct": False,
                "score": 0.0,
                "details": "empty_prediction",
            }

        # Truncate long inputs
        question = sample.question[:self.truncate_context]
        reference = sample.answer[:self.truncate_context] if sample.answer else "(no reference)"
        response = prediction[:self.truncate_context]

        # Build judge prompt
        judge_prompt = JUDGE_TEMPLATE.format(
            question=question,
            reference_answer=reference,
            model_response=response,
        )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
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
            # If judge fails, fall back to heuristic
            return {
                "correct": False,
                "score": 0.0,
                "details": f"judge_error: {e}",
            }

        # Parse judge response
        judgment = _extract_yes_no(judge_response)

        if judgment is True:
            return {
                "correct": True,
                "score": 1.0,
                "details": f"judge_yes: {judge_response[:100].strip()}",
            }
        elif judgment is False:
            return {
                "correct": False,
                "score": 0.0,
                "details": f"judge_no: {judge_response[:100].strip()}",
            }
        else:
            # Ambiguous — default to incorrect
            return {
                "correct": False,
                "score": 0.0,
                "details": f"judge_ambiguous: {judge_response[:100].strip()}",
            }
