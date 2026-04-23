"""
datasets/base.py — Abstract base classes for eval datasets.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvalSample:
    """A single evaluation sample, normalized across all datasets."""
    id: str
    question: str               # normalized question text
    answer: str                 # ground-truth answer (normalized)
    context: str = ""           # optional supporting context (table, passage)
    choices: List[str] = field(default_factory=list)  # optional MCQ choices
    metadata: dict = field(default_factory=dict)      # dataset-specific extras


class EvalDataset(ABC):
    """Abstract base class for all evaluation datasets."""

    name: str = "base"
    split: str = "test"

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> EvalSample:
        """Return the sample at the given index."""
        raise NotImplementedError

    def get_subset(self, n: int, seed: int = 42) -> List[EvalSample]:
        """
        Return a random subset of n samples.

        Args:
            n: Number of samples to return. If n >= len(dataset), returns all.
            seed: Random seed for reproducibility.

        Returns:
            List of EvalSample objects.
        """
        total = len(self)
        if n >= total:
            return [self[i] for i in range(total)]

        rng = random.Random(seed)
        indices = rng.sample(range(total), n)
        indices.sort()  # maintain original order
        return [self[i] for i in indices]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, split={self.split!r}, n={len(self)})"
