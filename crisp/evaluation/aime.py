"""AIME 2024/2025 dataset loaders for evaluation."""
from __future__ import annotations

import re
from typing import List, Optional

from datasets import load_dataset

from crisp.types import Problem

_aime24_cache: Optional[List[Problem]] = None
_aime25_cache: Optional[List[Problem]] = None


def _extract_boxed(s: str) -> str:
    """Extract content from \\boxed{...}, or return string as-is."""
    m = re.search(r"\\boxed\{([^}]+)\}", s)
    return m.group(1).strip() if m else s.strip()


def load_aime24_problems() -> List[Problem]:
    """Load AIME 2024 problems (30 problems: AIME I + II).

    Dataset: math-ai/aime24
    Columns: id, problem, solution (\\boxed{...} wrapped), url
    """
    global _aime24_cache
    if _aime24_cache is not None:
        return _aime24_cache

    ds = load_dataset("math-ai/aime24", split="train")

    problems = []
    for row in ds:
        answer = _extract_boxed(row["solution"])
        problems.append(Problem(
            text=row["problem"],
            ground_truth=answer,
        ))

    _aime24_cache = problems
    return _aime24_cache


def load_aime25_problems() -> List[Problem]:
    """Load AIME 2025 problems (30 problems: AIME I + II).

    Dataset: math-ai/aime25
    Columns: id, problem, answer (plain string)
    """
    global _aime25_cache
    if _aime25_cache is not None:
        return _aime25_cache

    ds = load_dataset("math-ai/aime25", split="test")

    problems = []
    for row in ds:
        problems.append(Problem(
            text=row["problem"],
            ground_truth=str(row["answer"]).strip(),
        ))

    _aime25_cache = problems
    return _aime25_cache
