"""
baselines/base.py — Abstract base class for eval baselines.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from eval_pipeline.datasets.base import EvalSample


class Baseline(ABC):
    """Abstract base class for all evaluation baselines."""

    name: str = "base"

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
