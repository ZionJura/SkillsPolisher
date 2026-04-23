"""
baselines/__init__.py — Baseline registry.
"""

from .base import Baseline
from .zero_shot import ZeroShotBaseline
from .random_kshot import RandomKShotBaseline
from .case_bandit import CASEBanditBaseline
from .bpo_rewrite import BPORewriteBaseline

BASELINE_REGISTRY = {
    "zero_shot": ZeroShotBaseline,
    "random_kshot": RandomKShotBaseline,
    "case_bandit": CASEBanditBaseline,
    "bpo_rewrite": BPORewriteBaseline,
}

__all__ = [
    "Baseline",
    "ZeroShotBaseline",
    "RandomKShotBaseline",
    "CASEBanditBaseline",
    "BPORewriteBaseline",
    "BASELINE_REGISTRY",
]
