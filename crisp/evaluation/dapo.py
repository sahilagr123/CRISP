"""DAPO-17k dataset loader for evaluation."""
from __future__ import annotations

from typing import List, Optional

from datasets import load_dataset

from crisp.types import Problem

_dapo_cache: Optional[List[Problem]] = None


def load_dapo_problems(max_problems: Optional[int] = None) -> List[Problem]:
    """Load DAPO-17k math problems from HuggingFace.

    Uses the processed/deduplicated English split from
    open-r1/DAPO-Math-17k-Processed. Results are cached after first load.
    Answers are always integers.

    Args:
        max_problems: Limit number of problems loaded. None = all ~14k.
    """
    global _dapo_cache
    if _dapo_cache is not None:
        return _dapo_cache[:max_problems] if max_problems else _dapo_cache

    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en")["train"]

    problems = []
    for row in ds:
        problems.append(Problem(
            text=row["prompt"],
            ground_truth=row["solution"],
        ))

    _dapo_cache = problems
    return _dapo_cache[:max_problems] if max_problems else _dapo_cache
