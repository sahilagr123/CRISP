"""Bayesian pass@N estimation (Chen et al., 2021)."""
from __future__ import annotations

from typing import List

from scipy.special import comb


def bayesian_pass_at_n(
    num_correct: List[int],
    num_samples: List[int],
    n: int,
) -> float:
    """Estimate pass@n using the unbiased estimator from Chen et al. (2021).

    For each problem, computes 1 - C(k-c, n) / C(k, n) where k is total
    samples and c is correct samples. Returns the mean across all problems
    with sufficient samples (k >= n).
    """
    total = 0.0
    count = 0
    for c, k in zip(num_correct, num_samples):
        if k < n:
            continue
        if c == 0:
            total += 0.0
        elif c >= k:
            total += 1.0
        else:
            total += 1.0 - comb(k - c, n, exact=True) / comb(k, n, exact=True)
        count += 1
    return total / count if count > 0 else 0.0
