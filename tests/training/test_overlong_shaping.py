"""Tests for overlong sequence reward shaping."""
import pytest

from crisp.training.overlong_shaping import compute_overlong_penalty


class TestOverlongPenalty:
    def test_under_l_max_no_penalty(self):
        assert compute_overlong_penalty(length=4000, l_max=8192, buffer=2048) == 0.0

    def test_at_l_max_no_penalty(self):
        assert compute_overlong_penalty(length=8192, l_max=8192, buffer=2048) == 0.0

    def test_in_buffer_zone_linear_ramp(self):
        # Midpoint of buffer: penalty = 0.5
        midpoint = 8192 + 1024
        penalty = compute_overlong_penalty(length=midpoint, l_max=8192, buffer=2048)
        assert penalty == pytest.approx(0.5)

    def test_at_l_hard_full_penalty(self):
        l_hard = 8192 + 2048
        penalty = compute_overlong_penalty(length=l_hard, l_max=8192, buffer=2048)
        assert penalty == pytest.approx(1.0)

    def test_beyond_l_hard_capped(self):
        penalty = compute_overlong_penalty(length=20000, l_max=8192, buffer=2048)
        assert penalty == pytest.approx(1.0)

    def test_post_discussion_limits(self):
        """Post-discussion uses different L_max and buffer."""
        assert compute_overlong_penalty(length=3000, l_max=4096, buffer=1024) == 0.0
        penalty = compute_overlong_penalty(length=4096 + 512, l_max=4096, buffer=1024)
        assert penalty == pytest.approx(0.5)
