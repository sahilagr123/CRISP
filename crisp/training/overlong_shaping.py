"""Overlong sequence reward shaping."""
from __future__ import annotations


def compute_overlong_penalty(
    length: int,
    l_max: int = 8192,
    buffer: int = 2048,
) -> float:
    """Compute penalty for sequences exceeding L_max.

    0.0 if length <= l_max.
    Linear ramp from 0.0 to 1.0 in [l_max, l_max + buffer].
    Capped at 1.0 beyond l_hard = l_max + buffer.
    """
    if length <= l_max:
        return 0.0
    l_hard = l_max + buffer
    if length >= l_hard:
        return 1.0
    return (length - l_max) / buffer
