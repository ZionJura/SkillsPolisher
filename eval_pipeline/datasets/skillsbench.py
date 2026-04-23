"""
datasets/skillsbench.py — SkillsBench dataset loader.

Per task: task.toml (metadata), instruction.md (problem),
          environment/skills/*/SKILL.md (skill docs),
          solution/solve.sh (reference answer)
"""

import os
from pathlib import Path
from typing import List, Optional

from .base import EvalDataset, EvalSample
from .data_utils import resolve_data_path


def _load_toml(path: str) -> dict:
    """Load TOML file using tomllib (Python 3.11+) or tomli fallback."""
    try:
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        pass
    try:
        import tomli
        with open(path, "rb") as f:
            return tomli.load(f)
    except ImportError:
        pass
    # Final fallback: minimal TOML parser for simple key=value files
    return _minimal_toml_parse(path)


def _minimal_toml_parse(path: str) -> dict:
    """Minimal TOML parser for simple flat TOML files (fallback)."""
    result: dict = {}
    current_section: dict = result
    current_key: str = ""

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    section_name = line[1:-1].strip()
                    # Handle nested sections like [metadata]
                    parts = section_name.split(".")
                    current_section = result
                    for part in parts:
                        if part not in current_section:
                            current_section[part] = {}
                        current_section = current_section[part]
                elif "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip()
                    # Strip quotes
                    if (val.startswith('"') and val.endswith('"')) or \
                       (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    # Parse lists
                    elif val.startswith("[") and val.endswith("]"):
                        inner = val[1:-1]
                        items = [
                            v.strip().strip('"').strip("'")
                            for v in inner.split(",")
                            if v.strip()
                        ]
                        val = items
                    # Parse booleans
                    elif val.lower() == "true":
                        val = True
                    elif val.lower() == "false":
                        val = False
                    else:
                        # Try numeric
                        try:
                            val = float(val) if "." in val else int(val)
                        except (ValueError, TypeError):
                            pass
                    current_section[key] = val
    except (IOError, OSError):
        pass

    return result


def _read_file_safe(path: str, max_chars: int = 50000) -> str:
    """Read a text file safely, returning empty string on error."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars)
        return content.strip()
    except (IOError, OSError):
        return ""


class SkillsBenchDataset(EvalDataset):
    """
    SkillsBench task dataset.
    Loads all tasks from the tasks directory.
    question=instruction, context=skill docs, metadata={difficulty, category, tags, skill_names}
    """

    name = "skillsbench"

    def __init__(self, split: str = "test", data_root: Optional[str] = None):
        self.split = split
        self.data_root = str(data_root or resolve_data_path("skillsbench"))
        self._samples: List[EvalSample] = []
        self._load()

    def _load(self) -> None:
        """Load all tasks from the tasks directory."""
        tasks_dir = Path(self.data_root)
        if not tasks_dir.exists():
            raise FileNotFoundError(
                f"SkillsBench tasks directory not found: {self.data_root}"
            )

        self._samples = []
        task_dirs = sorted([d for d in tasks_dir.iterdir() if d.is_dir()])

        for task_dir in task_dirs:
            try:
                sample = self._load_task(task_dir)
                if sample is not None:
                    self._samples.append(sample)
            except Exception as e:
                # Skip tasks that fail to load, continue with others
                continue

    def _load_task(self, task_dir: Path) -> Optional[EvalSample]:
        """Load a single task from its directory."""
        task_name = task_dir.name

        # Load task.toml
        toml_path = task_dir / "task.toml"
        toml_data = {}
        if toml_path.exists():
            toml_data = _load_toml(str(toml_path))

        # Extract metadata
        metadata_section = toml_data.get("metadata", {})
        difficulty = metadata_section.get("difficulty", "unknown")
        category = metadata_section.get("category", "unknown")
        tags = metadata_section.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        # Load instruction.md
        instruction_path = task_dir / "instruction.md"
        instruction = _read_file_safe(str(instruction_path))
        if not instruction:
            return None  # Skip tasks without instruction

        # Load skill docs from environment/skills/*/SKILL.md
        skills_dir = task_dir / "environment" / "skills"
        skill_bodies: List[str] = []
        skill_names: List[str] = []

        if skills_dir.exists():
            for skill_subdir in sorted(skills_dir.iterdir()):
                if skill_subdir.is_dir():
                    skill_md_path = skill_subdir / "SKILL.md"
                    if skill_md_path.exists():
                        skill_body = _read_file_safe(str(skill_md_path), max_chars=10000)
                        if skill_body:
                            skill_names.append(skill_subdir.name)
                            skill_bodies.append(f"## Skill: {skill_subdir.name}\n\n{skill_body}")

        # Combine skill bodies as context
        context = "\n\n---\n\n".join(skill_bodies) if skill_bodies else ""

        # Load reference solution
        solve_sh_path = task_dir / "solution" / "solve.sh"
        reference_answer = _read_file_safe(str(solve_sh_path), max_chars=5000)

        sample = EvalSample(
            id=f"skillsbench_{task_name}",
            question=instruction,
            answer=reference_answer,
            context=context,
            choices=[],
            metadata={
                "task_name": task_name,
                "difficulty": difficulty,
                "category": category,
                "tags": tags,
                "skill_names": skill_names,
                "task_dir": str(task_dir),
                "toml_data": toml_data,
            },
        )
        return sample

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self._samples[idx]
