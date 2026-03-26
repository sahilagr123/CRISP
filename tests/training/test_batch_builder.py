"""Tests for dynamic sampling and batch assembly."""
import pytest

from crisp.training.batch_builder import filter_dynamic_sampling, build_player_batch
from crisp.types import Rollout, DiscussionResult, TokenSequence
from tests.conftest import make_rollout


class TestFilterDynamicSampling:
    def test_mixed_rewards_retained(self):
        """Problem with mixed correct/wrong rollouts is kept."""
        rollouts = (
            [make_rollout(problem_idx=0, reward=1.0) for _ in range(5)]
            + [make_rollout(problem_idx=0, reward=0.0) for _ in range(3)]
        )
        filtered = filter_dynamic_sampling(rollouts)
        assert len(filtered) == 8

    def test_all_correct_filtered(self):
        """Problem where all 8 rollouts are correct -> filtered out."""
        rollouts = [make_rollout(problem_idx=0, reward=1.0) for _ in range(8)]
        filtered = filter_dynamic_sampling(rollouts)
        assert len(filtered) == 0

    def test_all_wrong_filtered(self):
        """Problem where all rollouts are wrong (reward=0) -> filtered out."""
        rollouts = [make_rollout(problem_idx=0, reward=0.0) for _ in range(8)]
        filtered = filter_dynamic_sampling(rollouts)
        assert len(filtered) == 0

    def test_all_no_box_filtered(self):
        """All -0.5 rewards -> all same -> filtered."""
        rollouts = [make_rollout(problem_idx=0, reward=-0.5) for _ in range(8)]
        filtered = filter_dynamic_sampling(rollouts)
        assert len(filtered) == 0

    def test_mixed_wrong_and_no_box_filtered(self):
        """Mix of 0.0 and -0.5 still has variance -> retained."""
        rollouts = (
            [make_rollout(problem_idx=0, reward=0.0) for _ in range(4)]
            + [make_rollout(problem_idx=0, reward=-0.5) for _ in range(4)]
        )
        filtered = filter_dynamic_sampling(rollouts)
        assert len(filtered) == 8

    def test_multiple_problems_filtered_independently(self):
        """Each problem filtered independently."""
        rollouts = (
            # Problem 0: all correct -> filter
            [make_rollout(problem_idx=0, reward=1.0) for _ in range(8)]
            # Problem 1: mixed -> keep
            + [make_rollout(problem_idx=1, reward=1.0) for _ in range(4)]
            + [make_rollout(problem_idx=1, reward=0.0) for _ in range(4)]
        )
        filtered = filter_dynamic_sampling(rollouts)
        assert len(filtered) == 8
        assert all(r.problem_idx == 1 for r in filtered)

    def test_persuader_bonus_creates_variance(self):
        """Problem with 1.0 and 1.3 rewards has variance -> kept."""
        rollouts = (
            [make_rollout(problem_idx=0, reward=1.3) for _ in range(4)]
            + [make_rollout(problem_idx=0, reward=1.0) for _ in range(4)]
        )
        filtered = filter_dynamic_sampling(rollouts)
        assert len(filtered) == 8
