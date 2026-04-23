"""
datasets/__init__.py — Dataset registry for the eval pipeline.
"""

from typing import Dict, Type

from .base import EvalDataset, EvalSample
from .aqua_rat import AquaRatDataset
from .gsm8k import GSM8KDataset
from .tabmwp import TabMWPDataset
from .finqa import FinQADataset
from .strategyqa import StrategyQADataset
from .bpo import BPOTestDataset, DollyEvalDataset, SelfInstructEvalDataset
from .skillsbench import SkillsBenchDataset
from .demo_bank import DemoBankDataset

# Registry mapping dataset name -> dataset class
DATASET_REGISTRY: Dict[str, Type[EvalDataset]] = {
    "aqua_rat": AquaRatDataset,
    "gsm8k": GSM8KDataset,
    "tabmwp": TabMWPDataset,
    "finqa": FinQADataset,
    "strategyqa": StrategyQADataset,
    "bpo_test": BPOTestDataset,
    "dolly_eval": DollyEvalDataset,
    "self_instruct_eval": SelfInstructEvalDataset,
    "skillsbench": SkillsBenchDataset,
    "demo_bank": DemoBankDataset,
}


def load_dataset(name: str, split: str = "test", **kwargs) -> EvalDataset:
    """
    Load a dataset by name from the registry.

    Args:
        name: Dataset name (key in DATASET_REGISTRY).
        split: Dataset split ('train', 'dev', 'test').
        **kwargs: Additional arguments passed to the dataset constructor.

    Returns:
        EvalDataset instance.

    Raises:
        ValueError: If dataset name is not in registry.
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Unknown dataset: {name!r}. "
            f"Available datasets: {available}"
        )
    dataset_cls = DATASET_REGISTRY[name]
    return dataset_cls(split=split, **kwargs)


__all__ = [
    "EvalDataset",
    "EvalSample",
    "DATASET_REGISTRY",
    "load_dataset",
    "AquaRatDataset",
    "GSM8KDataset",
    "TabMWPDataset",
    "FinQADataset",
    "StrategyQADataset",
    "BPOTestDataset",
    "DollyEvalDataset",
    "SelfInstructEvalDataset",
    "SkillsBenchDataset",
    "DemoBankDataset",
]
