"""Tests for coach reward computation."""
import numpy as np
import pytest

from crisp.types import Rollout, DiscussionResult, Problem
from crisp.rewards.coach_rewards import (
    compute_coach_reward,
    compute_uncertainty_reward,
    compute_discussion_reward,
    compute_intra_batch_penalty,
)
from crisp.rewards.repetition_buffer import RepetitionBuffer
from tests.conftest import make_rollout, make_problem


class TestUncertaintyReward:
    def test_fifty_percent_is_max(self):
        """p_hat = 0.5 -> r_uncertainty = 1.0 (maximum)."""
        assert compute_uncertainty_reward(0.5) == pytest.approx(1.0)

    def test_zero_percent(self):
        """p_hat = 0.0 -> r_uncertainty = 0.0."""
        assert compute_uncertainty_reward(0.0) == pytest.approx(0.0)

    def test_hundred_percent(self):
        """p_hat = 1.0 -> r_uncertainty = 0.0."""
        assert compute_uncertainty_reward(1.0) == pytest.approx(0.0)

    def test_thirty_percent(self):
        """p_hat = 0.3 -> r_uncertainty = 1 - 2|0.3 - 0.5| = 1 - 0.4 = 0.6."""
        assert compute_uncertainty_reward(0.3) == pytest.approx(0.6)

    def test_seventy_percent(self):
        """Symmetric: p_hat = 0.7 -> same as 0.3."""
        assert compute_uncertainty_reward(0.7) == pytest.approx(0.6)


class TestDiscussionReward:
    def test_resolved_correctly(self):
        assert compute_discussion_reward(
            discussion_occurred=True, resolved_correctly=True, alpha=0.3
        ) == pytest.approx(0.3)

    def test_occurred_but_unresolved(self):
        assert compute_discussion_reward(
            discussion_occurred=True, resolved_correctly=False, alpha=0.3
        ) == pytest.approx(0.15)

    def test_no_discussion(self):
        assert compute_discussion_reward(
            discussion_occurred=False, resolved_correctly=False, alpha=0.3
        ) == pytest.approx(0.0)


class TestIntraBatchPenalty:
    def test_all_unique(self):
        """Orthogonal embeddings -> no intra-batch penalty."""
        embeddings = [
            np.array([1, 0, 0, 0], dtype=np.float32),
            np.array([0, 1, 0, 0], dtype=np.float32),
            np.array([0, 0, 1, 0], dtype=np.float32),
        ]
        penalty = compute_intra_batch_penalty(
            idx=0, embeddings=embeddings, lambda_rep=1.0, tau_sim=0.85
        )
        assert penalty == pytest.approx(0.0)

    def test_all_identical(self):
        """Identical embeddings -> penalty = lambda * (n-1) / (n-1) = lambda."""
        emb = np.ones(4, dtype=np.float32)
        embeddings = [emb.copy() for _ in range(4)]
        penalty = compute_intra_batch_penalty(
            idx=0, embeddings=embeddings, lambda_rep=1.0, tau_sim=0.85
        )
        assert penalty == pytest.approx(1.0)

    def test_single_problem_no_penalty(self):
        """Single problem in batch -> no intra-batch comparison possible."""
        emb = np.ones(4, dtype=np.float32)
        penalty = compute_intra_batch_penalty(
            idx=0, embeddings=[emb], lambda_rep=1.0, tau_sim=0.85
        )
        assert penalty == pytest.approx(0.0)


class TestComputeCoachReward:
    def test_perfect_difficulty_no_discussion(self):
        """p_hat=0.5, no discussion, no repetition -> r = 1.0."""
        problems = [make_problem(embedding=np.ones(4, dtype=np.float32))]
        # 8 correct + 8 wrong = p_hat 0.5
        rollouts = (
            [make_rollout(problem_idx=0, player_id=0, correct=True) for _ in range(4)]
            + [make_rollout(problem_idx=0, player_id=0, correct=False) for _ in range(4)]
            + [make_rollout(problem_idx=0, player_id=1, correct=True) for _ in range(4)]
            + [make_rollout(problem_idx=0, player_id=1, correct=False) for _ in range(4)]
        )
        buf = RepetitionBuffer(max_batches=10, embedding_dim=4)
        reward = compute_coach_reward(
            problem=problems[0],
            problem_idx=0,
            all_embeddings=[problems[0].coach_embedding],
            player_rollouts=rollouts,
            discussion_occurred=False,
            resolved_correctly=False,
            repetition_buffer=buf,
        )
        assert reward == pytest.approx(1.0)

    def test_too_easy_gives_negative_reward(self):
        """p_hat=1.0 (everyone solved) -> too_easy_penalty."""
        problems = [make_problem(embedding=np.ones(4, dtype=np.float32))]
        rollouts = [make_rollout(problem_idx=0, correct=True) for _ in range(16)]
        buf = RepetitionBuffer(max_batches=10, embedding_dim=4)
        reward = compute_coach_reward(
            problem=problems[0],
            problem_idx=0,
            all_embeddings=[problems[0].coach_embedding],
            player_rollouts=rollouts,
            discussion_occurred=False,
            resolved_correctly=False,
            repetition_buffer=buf,
        )
        assert reward == pytest.approx(-0.3)

    def test_too_easy_custom_penalty(self):
        """too_easy_penalty is configurable."""
        problems = [make_problem(embedding=np.ones(4, dtype=np.float32))]
        rollouts = [make_rollout(problem_idx=0, correct=True) for _ in range(16)]
        buf = RepetitionBuffer(max_batches=10, embedding_dim=4)
        reward = compute_coach_reward(
            problem=problems[0],
            problem_idx=0,
            all_embeddings=[problems[0].coach_embedding],
            player_rollouts=rollouts,
            discussion_occurred=False,
            resolved_correctly=False,
            repetition_buffer=buf,
            too_easy_penalty=-0.8,
        )
        assert reward == pytest.approx(-0.8)

    def test_too_hard_gives_negative_reward(self):
        """p_hat=0.0 (nobody solved) -> too_hard_penalty."""
        problems = [make_problem(embedding=np.ones(4, dtype=np.float32))]
        rollouts = [make_rollout(problem_idx=0, correct=False) for _ in range(16)]
        buf = RepetitionBuffer(max_batches=10, embedding_dim=4)
        reward = compute_coach_reward(
            problem=problems[0],
            problem_idx=0,
            all_embeddings=[problems[0].coach_embedding],
            player_rollouts=rollouts,
            discussion_occurred=False,
            resolved_correctly=False,
            repetition_buffer=buf,
            too_hard_penalty=-0.5,
        )
        assert reward == pytest.approx(-0.5)

    def test_too_hard_custom_penalty(self):
        """too_hard_penalty is configurable."""
        problems = [make_problem(embedding=np.ones(4, dtype=np.float32))]
        rollouts = [make_rollout(problem_idx=0, correct=False) for _ in range(8)]
        buf = RepetitionBuffer(max_batches=10, embedding_dim=4)
        reward = compute_coach_reward(
            problem=problems[0],
            problem_idx=0,
            all_embeddings=[problems[0].coach_embedding],
            player_rollouts=rollouts,
            discussion_occurred=False,
            resolved_correctly=False,
            repetition_buffer=buf,
            too_hard_penalty=-1.0,
        )
        assert reward == pytest.approx(-1.0)

    def test_problem_idx_out_of_bounds_raises(self):
        """Using problem_idx beyond all_embeddings length should raise."""
        problems = [make_problem(embedding=np.ones(4, dtype=np.float32))]
        # Mixed rollouts so we don't hit too_easy/too_hard short-circuits
        rollouts = (
            [make_rollout(problem_idx=5, correct=True) for _ in range(2)]
            + [make_rollout(problem_idx=5, correct=False) for _ in range(2)]
        )
        buf = RepetitionBuffer(max_batches=10, embedding_dim=4)
        with pytest.raises(IndexError):
            compute_coach_reward(
                problem=problems[0],
                problem_idx=5,
                all_embeddings=[problems[0].coach_embedding],  # length 1, idx 5
                player_rollouts=rollouts,
                discussion_occurred=False,
                resolved_correctly=False,
                repetition_buffer=buf,
            )

    def test_empty_player_rollouts(self):
        """Empty rollout list should give p_hat=0 (too hard), reward=too_hard_penalty."""
        problems = [make_problem(embedding=np.ones(4, dtype=np.float32))]
        buf = RepetitionBuffer(max_batches=10, embedding_dim=4)
        reward = compute_coach_reward(
            problem=problems[0],
            problem_idx=0,
            all_embeddings=[problems[0].coach_embedding],
            player_rollouts=[],
            discussion_occurred=False,
            resolved_correctly=False,
            repetition_buffer=buf,
        )
        assert reward == pytest.approx(-0.5)

    def test_unsolvable_penalty(self):
        """Problem the coach couldn't solve gets unsolvable_penalty."""
        problem = make_problem(embedding=np.ones(4, dtype=np.float32))
        problem.self_solvable = False
        buf = RepetitionBuffer(max_batches=10, embedding_dim=4)
        reward = compute_coach_reward(
            problem=problem,
            problem_idx=0,
            all_embeddings=[problem.coach_embedding],
            player_rollouts=[],
            discussion_occurred=False,
            resolved_correctly=False,
            repetition_buffer=buf,
            unsolvable_penalty=-0.7,
        )
        assert reward == pytest.approx(-0.7)

    def test_floor_at_zero(self):
        """If repetition penalty exceeds positive components, floor at 0."""
        problems = [make_problem(embedding=np.ones(4, dtype=np.float32))]
        # p_hat=0.5 so we get uncertainty_reward=1.0, not the too_easy short-circuit
        rollouts = (
            [make_rollout(problem_idx=0, correct=True) for _ in range(8)]
            + [make_rollout(problem_idx=0, correct=False) for _ in range(8)]
        )
        # Fill buffer with identical embeddings to get high cross-batch penalty
        buf = RepetitionBuffer(max_batches=2, embedding_dim=4)
        ident = np.ones(4, dtype=np.float32)
        buf.push([ident] * 4)
        buf.push([ident] * 4)
        reward = compute_coach_reward(
            problem=problems[0],
            problem_idx=0,
            all_embeddings=[problems[0].coach_embedding],
            player_rollouts=rollouts,
            discussion_occurred=False,
            resolved_correctly=False,
            repetition_buffer=buf,
        )
        assert reward >= 0.0
