"""Tests for two-pool advantage normalization with std floor."""
import math
import numpy as np
import pytest

from crisp.rewards.advantages import compute_player_advantages, compute_coach_advantages, MIN_SIGMA
from crisp.rewards.ema_tracker import EMATracker


class TestComputePlayerAdvantages:
    def test_basic_normalization(self):
        """Mean/std normalization with sigma floor."""
        pre_rewards = [0.0, 0.0, 1.0, 1.0]
        post_rewards = []
        ema = EMATracker(mu=0.5, sigma_sq=0.25)
        pre_adv, post_adv = compute_player_advantages(pre_rewards, post_rewards, ema)
        mean = float(np.mean(pre_rewards))
        std = max(float(np.std(pre_rewards)), MIN_SIGMA)
        expected = [(r - mean) / std for r in pre_rewards]
        for got, exp in zip(pre_adv, expected):
            assert got == pytest.approx(exp)
        assert post_adv == []

    def test_all_same_rewards_zero_advantage(self):
        """All rewards identical -> advantages = 0."""
        pre_rewards = [1.0, 1.0, 1.0, 1.0]
        ema = EMATracker()
        pre_adv, _ = compute_player_advantages(pre_rewards, [], ema)
        for a in pre_adv:
            assert a == pytest.approx(0.0, abs=1e-4)

    def test_post_discussion_uses_ema(self):
        """Post-discussion pool uses EMA-smoothed mean/std."""
        pre_rewards = [0.0, 1.0]
        post_rewards = [1.0, 0.0]
        ema = EMATracker(mu=0.5, sigma_sq=0.25, eta=0.2)
        pre_adv, post_adv = compute_player_advantages(pre_rewards, post_rewards, ema)
        assert len(post_adv) == 2
        # Uses EMA mu=0.5 and sigma=max(sqrt(0.25), MIN_SIGMA) before update
        sigma = max(math.sqrt(0.25), MIN_SIGMA)
        expected_post = [(r - 0.5) / sigma for r in post_rewards]
        for got, exp in zip(post_adv, expected_post):
            assert got == pytest.approx(exp, rel=1e-4)

    def test_ema_updated_after_call(self):
        """The EMA tracker should be updated by the function call."""
        ema = EMATracker(mu=0.0, sigma_sq=0.0, eta=0.2)
        compute_player_advantages([0.0, 1.0], [1.0], ema)
        assert ema.mu != 0.0  # Should have been updated

    def test_negative_rewards_handled(self):
        """Pre-discussion rewards can be -0.5 (no-box penalty)."""
        pre_rewards = [-0.5, 0.0, 1.0, 1.0]
        ema = EMATracker()
        pre_adv, _ = compute_player_advantages(pre_rewards, [], ema)
        assert len(pre_adv) == 4
        # -0.5 should have the most negative advantage
        assert pre_adv[0] < pre_adv[1] < pre_adv[2]

    def test_single_rollout(self):
        """Single rollout -> advantage = 0 (mean = that reward)."""
        pre_adv, _ = compute_player_advantages([1.0], [], EMATracker())
        assert len(pre_adv) == 1
        assert pre_adv[0] == pytest.approx(0.0, abs=1e-4)

    def test_sigma_floor_prevents_explosion(self):
        """When std < MIN_SIGMA, the floor prevents advantage inflation."""
        # 7 zeros and 1 one: std=0.33, below MIN_SIGMA=0.5
        pre_rewards = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ema = EMATracker()
        pre_adv, _ = compute_player_advantages(pre_rewards, [], ema)
        # With sigma floor=0.5: max adv = (1.0 - 0.125) / 0.5 = 1.75
        # Without floor: (1.0 - 0.125) / 0.33 = 2.65
        assert max(abs(a) for a in pre_adv) < 2.0


class TestComputeCoachAdvantages:
    """Coach advantages use single-pool EMA normalization (Step 12)."""

    def test_basic_ema_normalization(self):
        """Coach advantages use EMA mean/std after first update."""
        ema = EMATracker(mu=0.5, sigma_sq=0.25, eta=0.2)
        # First call uses per-batch stats (cold-start fix)
        compute_coach_advantages([0.3, 0.7], ema)
        assert ema._has_been_updated
        # Second call should use EMA mean/std
        rewards = [0.8, 0.2, 0.5]
        advantages = compute_coach_advantages(rewards, ema)
        # After first update: mu=mean(0.3,0.7)=0.5, sigma_sq=var(0.3,0.7)=0.04
        expected_mu = 0.5
        expected_sigma = max(math.sqrt(0.04), MIN_SIGMA)
        expected = [(r - expected_mu) / expected_sigma for r in rewards]
        for got, exp in zip(advantages, expected):
            assert got == pytest.approx(exp, rel=1e-4)

    def test_ema_updated_after_call(self):
        """EMA tracker should be updated with the coach rewards after normalization."""
        ema = EMATracker(mu=0.5, sigma_sq=0.25, eta=0.2)
        rewards = [0.8, 0.2]
        compute_coach_advantages(rewards, ema)
        # First call initializes directly from batch stats (cold-start fix)
        assert ema._has_been_updated
        assert ema.mu == pytest.approx(0.5)  # mean of [0.8, 0.2]
        assert ema.sigma_sq == pytest.approx(0.09)  # var of [0.8, 0.2]

    def test_empty_rewards(self):
        """Empty rewards -> empty advantages, consecutive_empty incremented."""
        ema = EMATracker(mu=0.5, sigma_sq=0.25, eta=0.2)
        advantages = compute_coach_advantages([], ema)
        assert advantages == []
        assert ema.consecutive_empty == 1

    def test_all_finite(self):
        """All advantages must be finite even with zero-variance EMA."""
        ema = EMATracker(mu=0.0, sigma_sq=0.0, eta=0.2)
        # First call: per-batch stats (cold-start)
        compute_coach_advantages([0.5], ema)
        # Now EMA has sigma_sq=0 (single element), test second call
        rewards = [1.0, 0.5, 0.0]
        advantages = compute_coach_advantages(rewards, ema)
        for a in advantages:
            assert math.isfinite(a)

    def test_cold_start_uses_batch_stats(self):
        """First call uses per-batch mean/std, not stale EMA init."""
        ema = EMATracker(mu=0.5, sigma_sq=0.25, eta=0.2)
        rewards = [-0.3, -0.5, -0.1]
        advantages = compute_coach_advantages(rewards, ema)
        # Per-batch: mean=-0.3, std=max(std(rewards), MIN_SIGMA)
        # Should be approximately centered around 0
        assert abs(sum(advantages)) < 1e-6  # mean-centered
