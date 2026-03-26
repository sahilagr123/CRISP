"""Tests for EMA mean/variance tracker."""
import math
import pytest

from crisp.rewards.ema_tracker import EMATracker


class TestEMATracker:
    def test_initial_values(self):
        t = EMATracker()
        assert t.mu == 0.5
        assert t.sigma_sq == 0.25

    def test_custom_init(self):
        t = EMATracker(mu=0.0, sigma_sq=1.0, eta=0.1)
        assert t.mu == 0.0
        assert t.sigma_sq == 1.0
        assert t.eta == 0.1

    def test_single_update(self):
        t = EMATracker(mu=0.5, sigma_sq=0.25, eta=0.2)
        t.update([1.0, 1.0, 1.0])  # batch_mean=1.0, batch_var=0.0
        # First update initializes directly from batch stats (cold-start fix)
        assert t.mu == pytest.approx(1.0)
        assert t.sigma_sq == pytest.approx(0.01)  # clamped to MIN_SIGMA_SQ
        assert t._has_been_updated

    def test_empty_update_is_noop(self):
        t = EMATracker(mu=0.5, sigma_sq=0.25, eta=0.2)
        t.update([])
        assert t.mu == 0.5
        assert t.sigma_sq == 0.25

    def test_convergence_to_constant(self):
        """After many updates with reward=1.0, mu should approach 1.0."""
        t = EMATracker(mu=0.5, sigma_sq=0.25, eta=0.2)
        for _ in range(100):
            t.update([1.0])
        assert t.mu == pytest.approx(1.0, abs=1e-6)
        assert t.sigma_sq == pytest.approx(0.01)  # clamped to MIN_SIGMA_SQ

    def test_multiple_updates_track_mean(self):
        t = EMATracker(mu=0.0, sigma_sq=0.0, eta=0.2)
        t.update([0.0, 1.0])  # mean=0.5, var=0.25
        # First update: direct init from batch stats
        assert t.mu == pytest.approx(0.5)
        assert t.sigma_sq == pytest.approx(0.25)
        # Second update: EMA blending
        t.update([0.0, 1.0])
        assert t.mu == pytest.approx(0.8 * 0.5 + 0.2 * 0.5)  # still 0.5
        assert t.sigma_sq == pytest.approx(0.8 * 0.25 + 0.2 * 0.25)  # still 0.25

    def test_consecutive_empty_batches_counter(self):
        t = EMATracker()
        for _ in range(6):
            t.update([])
        assert t.consecutive_empty >= 6
        t.update([1.0])
        assert t.consecutive_empty == 0

    def test_single_element_update(self):
        """Single-element reward list: var=0, mu updates correctly."""
        t = EMATracker(mu=0.0, sigma_sq=0.0, eta=0.2)
        t.update([1.0])
        # First update: direct init
        assert t.mu == pytest.approx(1.0)
        assert t.sigma_sq == pytest.approx(0.01)  # clamped to MIN_SIGMA_SQ
