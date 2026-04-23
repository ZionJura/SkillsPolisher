"""
evaluators/exact_match.py — Exact match and normalized match evaluators.

Handles:
- MCQ (AQUA-RAT, TabMWP with choices): extract letter/option from generated text
- Numeric (GSM8K, FinQA): extract final number, compare with tolerance
- Yes/No (StrategyQA): extract yes/no from response
- General: normalized string match
"""

import re
import string
from typing import Optional

from eval_pipeline.datasets.base import EvalSample
from .base import Evaluator

# Datasets that use MCQ-style answers
MCQ_DATASETS = {"aqua_rat"}
# Datasets that use numeric answers
NUMERIC_DATASETS = {"gsm8k", "finqa", "tabmwp"}
# Datasets that use yes/no answers
YESNO_DATASETS = {"strategyqa"}


def normalize_text(text: str) -> str:
    """Normalize text: lowercase, strip whitespace and punctuation."""
    if not text:
        return ""
    text = text.lower().strip()
    # Remove punctuation except for decimal points in numbers
    # Keep alphanumeric, spaces, decimal points, minus signs
    text = re.sub(r"[^\w\s\.\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_letter_answer(text: str, valid_letters: Optional[list] = None) -> str:
    """
    Extract a single letter answer from generated text.
    Looks for patterns like "The answer is A", "A)", "(A)", "A.", etc.

    Args:
        text: Generated text to extract from.
        valid_letters: Valid answer letters (default A-E).

    Returns:
        Extracted letter (uppercase) or empty string.
    """
    if not text:
        return ""

    if valid_letters is None:
        valid_letters = list("ABCDE")

    valid_set = set(v.upper() for v in valid_letters)
    text_upper = text.upper()

    # Pattern 1: "The answer is X" or "answer: X"
    patterns = [
        r"(?:THE\s+)?(?:FINAL\s+)?ANSWER\s+IS\s*:?\s*([A-E])\b",
        r"ANSWER\s*:?\s*([A-E])\b",
        r"(?:OPTION|CHOICE)\s+([A-E])\b",
        r"\bTHEREFORE[,\s]+([A-E])\b",
        r"\bSO[,\s]+(?:THE\s+)?ANSWER\s+IS\s+([A-E])\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text_upper)
        if m and m.group(1) in valid_set:
            return m.group(1)

    # Pattern 2: Standalone letter at end of text
    # Look for the last occurrence of a valid letter followed by punctuation or end
    reversed_text = text_upper.strip()
    last_line = reversed_text.split("\n")[-1].strip()

    for pattern in [
        r"^([A-E])[)\.\s]",
        r"\(([A-E])\)",
        r"\b([A-E])\s*$",
        r"^([A-E])$",
    ]:
        m = re.search(pattern, last_line)
        if m and m.group(1) in valid_set:
            return m.group(1)

    # Pattern 3: Any valid letter surrounded by non-letter chars near end
    matches = re.findall(r"(?<![A-Z])([A-E])(?![A-Z])", text_upper[-200:])
    for match in reversed(matches):
        if match in valid_set:
            return match

    return ""


def extract_numeric_answer(text: str) -> Optional[float]:
    """
    Extract a numeric answer from generated text.

    Looks for:
    1. "#### N" pattern (GSM8K style)
    2. "The answer is N" patterns
    3. The last number in the text
    """
    if not text:
        return None

    # 1. GSM8K-style: #### N
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    # 2. "The answer is N" or "= N" at end
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([\-\$\d,\.]+)",
        r"(?:therefore|so)[,\s]+(?:the\s+)?(?:answer|result)\s+(?:is\s+)?:?\s*([\-\$\d,\.]+)",
        r"=\s*([\-\$\d,\.]+)\s*$",
        r"≈\s*([\-\$\d,\.]+)\s*$",
    ]
    text_lower = text.lower().strip()
    for pattern in patterns:
        m = re.search(pattern, text_lower)
        if m:
            num_str = m.group(1).replace(",", "").replace("$", "").strip()
            try:
                return float(num_str)
            except ValueError:
                continue

    # 3. Last number in the text (last resort)
    numbers = re.findall(r"[\-]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        # Take the last number
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass

    return None


def numbers_match(pred_num: float, true_num: float, tolerance: float = 1e-3) -> bool:
    """Check if two numbers match within tolerance."""
    if true_num == 0:
        return abs(pred_num) < tolerance
    return abs(pred_num - true_num) / (abs(true_num) + 1e-10) < tolerance


def extract_yesno(text: str) -> str:
    """
    Extract yes/no answer from generated text.

    Returns 'yes', 'no', or empty string.
    """
    if not text:
        return ""
    text_lower = text.lower().strip()

    # Check last sentence first
    sentences = re.split(r"[.!?]", text_lower)
    for sent in reversed(sentences):
        sent = sent.strip()
        if not sent:
            continue
        if re.search(r"\byes\b", sent):
            return "yes"
        if re.search(r"\bno\b", sent):
            return "no"

    # Fallback: check full text
    if re.search(r"\byes\b", text_lower):
        return "yes"
    if re.search(r"\bno\b", text_lower):
        return "no"

    return ""


def parse_true_number(answer_str: str) -> Optional[float]:
    """Parse a ground-truth answer string into a float."""
    if not answer_str:
        return None
    # Remove common units and currency symbols
    cleaned = re.sub(r"[^\d\.\-,]", "", answer_str.replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


class ExactMatchEvaluator(Evaluator):
    """
    Exact match evaluator with dataset-aware normalization.

    Handles MCQ, numeric, yes/no, and general text matching.
    """

    def __init__(self, dataset_name: str = "", tolerance: float = 1e-3):
        self.dataset_name = dataset_name
        self.tolerance = tolerance

    def score(self, prediction: str, sample: EvalSample) -> dict:
        """Score prediction against sample ground truth."""
        true_answer = sample.answer
        pred = prediction.strip() if prediction else ""
        true = true_answer.strip() if true_answer else ""

        # Determine scoring mode
        has_choices = bool(sample.choices)
        is_mcq = self.dataset_name in MCQ_DATASETS or has_choices
        is_numeric = self.dataset_name in NUMERIC_DATASETS and not has_choices
        is_yesno = self.dataset_name in YESNO_DATASETS

        # Override based on ground truth format
        if re.match(r"^[A-E]$", true.upper()) and has_choices:
            is_mcq = True
            is_numeric = False
        elif true.lower() in ("yes", "no"):
            is_yesno = True
            is_numeric = False
            is_mcq = False

        # --- MCQ scoring ---
        if is_mcq:
            valid_letters = []
            if sample.choices:
                labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                valid_letters = [labels[i] for i in range(len(sample.choices))]
            else:
                valid_letters = list("ABCDE")

            true_letter = true.upper()[:1] if true else ""
            pred_letter = extract_letter_answer(pred, valid_letters)

            if not pred_letter and pred:
                # Try direct match
                pred_upper = pred.strip().upper()
                if pred_upper in set(valid_letters):
                    pred_letter = pred_upper

            correct = bool(pred_letter and pred_letter == true_letter)
            return {
                "correct": correct,
                "score": 1.0 if correct else 0.0,
                "details": f"pred_letter={pred_letter!r}, true_letter={true_letter!r}",
            }

        # --- Yes/No scoring ---
        if is_yesno:
            pred_yn = extract_yesno(pred)
            true_yn = true.lower().strip()
            correct = pred_yn == true_yn
            return {
                "correct": correct,
                "score": 1.0 if correct else 0.0,
                "details": f"pred_yn={pred_yn!r}, true_yn={true_yn!r}",
            }

        # --- Numeric scoring ---
        if is_numeric:
            pred_num = extract_numeric_answer(pred)
            true_num = parse_true_number(true)

            if pred_num is not None and true_num is not None:
                correct = numbers_match(pred_num, true_num, self.tolerance)
                return {
                    "correct": correct,
                    "score": 1.0 if correct else 0.0,
                    "details": f"pred_num={pred_num}, true_num={true_num}",
                }
            else:
                # Fall through to text match if numeric extraction fails
                pass

        # --- Normalized text match ---
        pred_norm = normalize_text(pred)
        true_norm = normalize_text(true)

        # Exact normalized match
        exact = pred_norm == true_norm
        if exact:
            return {
                "correct": True,
                "score": 1.0,
                "details": "exact_normalized_match",
            }

        # Partial match: check if true answer appears in prediction
        if true_norm and true_norm in pred_norm:
            return {
                "correct": True,
                "score": 0.8,
                "details": "true_in_prediction",
            }

        # Check if prediction appears in true (very short predictions)
        if pred_norm and len(pred_norm) <= 50 and pred_norm in true_norm:
            return {
                "correct": True,
                "score": 0.7,
                "details": "prediction_in_true",
            }

        return {
            "correct": False,
            "score": 0.0,
            "details": f"no_match: pred={pred_norm[:50]!r}, true={true_norm[:50]!r}",
        }
