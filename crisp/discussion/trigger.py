"""Majority vote computation and discussion trigger logic."""
from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional

from crisp.types import Rollout
from crisp.verifier.sympy_verify import equivalent


def majority_vote(rollouts: List[Rollout]) -> Optional[str]:
    """Compute the majority answer across rollouts.

    Groups symbolically equivalent answers. Ties broken by first occurrence.
    Returns None if all rollouts have answer=None.
    """
    # Group answers, using equivalence checking
    # Each group: (representative_answer, count, first_index)
    groups: list[tuple[str, int, int]] = []

    for i, r in enumerate(rollouts):
        if r.answer is None:
            continue
        matched = False
        for j, (rep, count, first_idx) in enumerate(groups):
            if equivalent(r.answer, rep):
                groups[j] = (rep, count + 1, first_idx)
                matched = True
                break
        if not matched:
            groups.append((r.answer, 1, i))

    if not groups:
        return None

    # Sort by count descending, then by first_index ascending (tie-break)
    groups.sort(key=lambda g: (-g[1], g[2]))
    return groups[0][0]


def should_discuss(
    majority_a: Optional[str],
    majority_b: Optional[str],
) -> bool:
    """Determine if discussion should be triggered between two players.

    Discussion triggered when the two players' majority answers disagree.
    """
    if majority_a is None and majority_b is None:
        return False
    if majority_a is None or majority_b is None:
        return True
    return not equivalent(majority_a, majority_b)
