"""Advantage normalization for CRISP players and coach."""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from crisp.rewards.ema_tracker import EMATracker

# Minimum standard deviation for normalization. Prevents advantage explosion
# when all rewards are identical (std=0) while still allowing the std to
# dampen advantages when variance is real. Set to 0.5 so that reward
# differences of ~1.0 produce advantages of ~2.0 (within MAX_ADV=10).
MIN_SIGMA = 0.5


def compute_player_advantages(
    pre_rewards: List[float],
    post_rewards: List[float],
    ema_tracker: EMATracker,
    epsilon: float = 1e-8,
) -> Tuple[List[float], List[float]]:
    """Compute advantages using two normalization pools.

    Pool 1 (pre-discussion): per-batch mean/std normalization with floor.
    Pool 2 (post-discussion): EMA-smoothed mean/std normalization with floor.

    The EMA tracker is updated with post_rewards during this call.

    Args:
        pre_rewards: Rewards for pre-discussion rollouts (already filtered by dynamic sampling).
        post_rewards: Rewards for post-discussion responses.
        ema_tracker: EMA tracker for Pool 2 statistics.
        epsilon: Small constant for numerical stability.

    Returns:
        (pre_advantages, post_advantages) tuple.
    """
    # Pool 1: per-batch normalization with sigma floor.
    if pre_rewards:
        mean_pre = float(np.mean(pre_rewards))
        std_pre = max(float(np.std(pre_rewards)), MIN_SIGMA)
        pre_advantages = [(r - mean_pre) / std_pre for r in pre_rewards]
    else:
        pre_advantages = []

    # Pool 2: EMA-smoothed normalization with sigma floor.
    # Use current EMA stats for normalization, THEN update
    if post_rewards:
        mu = ema_tracker.mu
        sigma = max(math.sqrt(ema_tracker.sigma_sq), MIN_SIGMA)
        post_advantages = [(r - mu) / sigma for r in post_rewards]
        ema_tracker.update(post_rewards)
    else:
        post_advantages = []
        ema_tracker.update([])  # Tracks consecutive empty

    # Defense in depth: clamp advantages to prevent loss explosion
    MAX_ADV = 10.0
    pre_advantages = [max(-MAX_ADV, min(MAX_ADV, a)) for a in pre_advantages]
    post_advantages = [max(-MAX_ADV, min(MAX_ADV, a)) for a in post_advantages]

    return pre_advantages, post_advantages


def compute_coach_advantages(
    rewards: List[float],
    ema_tracker: EMATracker,
    epsilon: float = 1e-8,
) -> List[float]:
    """Compute coach advantages using EMA-smoothed normalization (Step 12).

    Â_i = (r_coach(x_i) - μ_ema) / max(σ_ema, MIN_SIGMA)

    Uses current EMA stats for normalization, then updates the tracker.

    Args:
        rewards: Coach rewards for each problem in the batch.
        ema_tracker: EMA tracker for coach statistics.
        epsilon: Small constant for numerical stability.

    Returns:
        List of coach advantages, one per problem.
    """
    if not rewards:
        ema_tracker.update([])
        return []

    # Use per-batch stats on first update to avoid cold-start bias
    if not ema_tracker._has_been_updated:
        mu = float(np.mean(rewards))
        sigma = max(float(np.std(rewards)), MIN_SIGMA)
    else:
        mu = ema_tracker.mu
        sigma = max(math.sqrt(ema_tracker.sigma_sq), MIN_SIGMA)
    advantages = [(r - mu) / sigma for r in rewards]
    ema_tracker.update(rewards)

    # Clamp to prevent loss explosion
    MAX_ADV = 10.0
    advantages = [max(-MAX_ADV, min(MAX_ADV, a)) for a in advantages]
    return advantages
