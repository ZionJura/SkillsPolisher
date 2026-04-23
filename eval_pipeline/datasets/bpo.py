"""
datasets/bpo.py — BPO test set dataset loaders.

Three files:
  - bpo_test.json: fields prompt, optimized_prompt, good_res, bad_res
  - dolly_eval.json: fields instruction, context, response, category, idx
  - self_instruct_eval.json: fields id, category, instruction, context, output
"""

import json
import os
from typing import List, Optional

from .base import EvalDataset, EvalSample
from .data_utils import resolve_data_path


class BPOTestDataset(EvalDataset):
    """
    BPO test set (bpo_test.json).
    question=prompt, context=optimized_prompt, answer=good_res
    """

    name = "bpo_test"

    def __init__(self, split: str = "test", data_root: Optional[str] = None):
        self.split = split
        self.data_root = str(data_root or resolve_data_path("bpo"))
        self.file_path = os.path.join(self.data_root, "bpo_test.json")
        self._samples: List[EvalSample] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"BPO test file not found: {self.file_path}"
            )
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise ValueError(f"Error reading BPO test file {self.file_path}: {e}") from e

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {self.file_path}")

        self._samples = []
        for idx, record in enumerate(data):
            prompt = record.get("prompt", "").strip()
            if not prompt:
                continue

            optimized_prompt = record.get("optimized_prompt", "").strip()
            good_res = record.get("good_res", "").strip()
            bad_res = record.get("bad_res", "").strip()

            sample = EvalSample(
                id=f"bpo_test_{idx}",
                question=prompt,
                answer=good_res,
                context=optimized_prompt,
                choices=[],
                metadata={
                    "optimized_prompt": optimized_prompt,
                    "bad_res": bad_res,
                    "source": "bpo_test",
                },
            )
            self._samples.append(sample)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self._samples[idx]


class DollyEvalDataset(EvalDataset):
    """
    Dolly eval set (dolly_eval.json).
    Standard instruction-following format.
    """

    name = "dolly_eval"

    def __init__(self, split: str = "test", data_root: Optional[str] = None):
        self.split = split
        self.data_root = str(data_root or resolve_data_path("bpo"))
        self.file_path = os.path.join(self.data_root, "dolly_eval.json")
        self._samples: List[EvalSample] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Dolly eval file not found: {self.file_path}"
            )
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise ValueError(f"Error reading Dolly eval file {self.file_path}: {e}") from e

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {self.file_path}")

        self._samples = []
        for idx, record in enumerate(data):
            instruction = record.get("instruction", "").strip()
            if not instruction:
                continue

            context = record.get("context", "").strip()
            response = record.get("response", "").strip()
            category = record.get("category", "").strip()
            record_idx = record.get("idx", idx)

            # Build question: combine instruction + context if present
            if context:
                question = f"{instruction}\n\nContext: {context}"
            else:
                question = instruction

            sample = EvalSample(
                id=f"dolly_{record_idx}",
                question=question,
                answer=response,
                context=context,
                choices=[],
                metadata={
                    "instruction": instruction,
                    "category": category,
                    "idx": record_idx,
                    "source": "dolly_eval",
                },
            )
            self._samples.append(sample)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self._samples[idx]


class SelfInstructEvalDataset(EvalDataset):
    """
    Self-instruct eval set (self_instruct_eval.json).
    Standard instruction-following format.
    """

    name = "self_instruct_eval"

    def __init__(self, split: str = "test", data_root: Optional[str] = None):
        self.split = split
        self.data_root = str(data_root or resolve_data_path("bpo"))
        self.file_path = os.path.join(self.data_root, "self_instruct_eval.json")
        self._samples: List[EvalSample] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Self-instruct eval file not found: {self.file_path}"
            )
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise ValueError(
                f"Error reading self-instruct eval file {self.file_path}: {e}"
            ) from e

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {self.file_path}")

        self._samples = []
        for idx, record in enumerate(data):
            instruction = record.get("instruction", "").strip()
            if not instruction:
                continue

            context = record.get("context", "").strip()
            output = record.get("output", "").strip()
            category = record.get("category", "").strip()
            record_id = str(record.get("id", f"si_{idx}"))

            # Build question: combine instruction + context if present
            if context:
                question = f"{instruction}\n\nContext: {context}"
            else:
                question = instruction

            sample = EvalSample(
                id=f"self_instruct_{record_id}",
                question=question,
                answer=output,
                context=context,
                choices=[],
                metadata={
                    "instruction": instruction,
                    "category": category,
                    "source": "self_instruct_eval",
                },
            )
            self._samples.append(sample)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self._samples[idx]
