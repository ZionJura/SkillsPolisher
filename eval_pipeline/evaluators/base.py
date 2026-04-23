"""
evaluators/base.py — Abstract base class for evaluators.
"""

from abc import ABC, abstractmethod

from eval_pipeline.datasets.base import EvalSample


class Evaluator(ABC):
    """Abstract base class for all evaluators."""

    @abstractmethod
    def score(self, prediction: str, sample: EvalSample) -> dict:
        """
        Score a single prediction against the ground truth.

        Args:
            prediction: The model's predicted answer.
            sample: The original evaluation sample with ground truth.

        Returns:
            dict with keys:
                - "correct" (bool): Whether the prediction is correct.
                - "score" (float): Numeric score in [0, 1].
                - "details" (str): Human-readable explanation.
        """
        raise NotImplementedError

    def batch_score(self, predictions: list, samples: list) -> list:
        """
        Score a batch of predictions.

        Args:
            predictions: List of prediction strings.
            samples: List of EvalSample objects.

        Returns:
            List of score dicts.
        """
        if len(predictions) != len(samples):
            raise ValueError(
                f"Mismatched lengths: {len(predictions)} predictions vs "
                f"{len(samples)} samples"
            )
        return [self.score(pred, sample) for pred, sample in zip(predictions, samples)]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
