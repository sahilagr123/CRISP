"""End-to-end test: synthetic 8-rollout scenario tracing exact reward values
through compute_solve_reward -> apply_persuader_bonus -> filter_dynamic_sampling
-> compute_player_advantages -> final advantage values.

Asserts specific numerical values at each stage.
"""
import math
import pytest
import numpy as np

from crisp.types import Problem, Rollout, DiscussionResult
from crisp.rewards.player_rewards import compute_solve_reward, apply_persuader_bonus
from crisp.rewards.advantages import compute_player_advantages
from crisp.rewards.ema_tracker import EMATracker
from crisp.training.batch_builder import filter_dynamic_sampling
from tests.conftest import make_rollout, make_problem


class TestFullRewardFlow:
    """Trace exact numbers through the full reward -> advantage pipeline."""

    def test_complete_scenario(self):
        """
        Setup:
        - 2 problems, 8 rollouts per problem per player
        - Problem 0: Alice 6/8 correct, Bob 3/8 correct
          -> Majority: Alice="42", Bob="99" -> DISAGREE -> discussion triggered
          -> Discussion: Alice stays correct, Bob flips to correct
          -> Alice is persuader
        - Problem 1: Alice 8/8 correct, Bob 8/8 correct
          -> No discussion, all-correct -> filtered by dynamic sampling
        """
        problems = [make_problem(ground_truth="42"), make_problem(ground_truth="7")]

        # -- Build rollouts for Player 0 (Alice) --
        alice_rollouts = (
            # Problem 0: 6 correct, 2 wrong
            [make_rollout(problem_idx=0, player_id=0, answer="42", correct=True) for _ in range(6)]
            + [make_rollout(problem_idx=0, player_id=0, answer="99", correct=False) for _ in range(2)]
            # Problem 1: 8 correct
            + [make_rollout(problem_idx=1, player_id=0, answer="7", correct=True) for _ in range(8)]
        )

        # -- Build rollouts for Player 1 (Bob) --
        bob_rollouts = (
            # Problem 0: 3 correct, 5 wrong
            [make_rollout(problem_idx=0, player_id=1, answer="42", correct=True) for _ in range(3)]
            + [make_rollout(problem_idx=0, player_id=1, answer="99", correct=False) for _ in range(5)]
            # Problem 1: 8 correct
            + [make_rollout(problem_idx=1, player_id=1, answer="7", correct=True) for _ in range(8)]
        )

        rollouts = {0: alice_rollouts, 1: bob_rollouts}

        # -- Step 4: Compute solve rewards --
        for pid in rollouts:
            for r in rollouts[pid]:
                r.reward = compute_solve_reward(r)

        # Verify pre-discussion rewards
        assert all(r.reward == 1.0 for r in alice_rollouts[:6])   # correct
        assert all(r.reward == 0.0 for r in alice_rollouts[6:8])  # wrong
        assert all(r.reward == 1.0 for r in alice_rollouts[8:])   # correct

        # -- Step 5: Majority + trigger --
        majority_answers = {(0, 0): "42", (1, 0): "99", (0, 1): "7", (1, 1): "7"}

        # -- Step 6-7: Discussion + persuader bonus --
        discussion_results = {
            0: [DiscussionResult(
                problem_idx=0, player_id=0, tokens=[], text="", log_probs=[],
                final_answer="42", correct=True, reward=1.0,
            )],
            1: [DiscussionResult(
                problem_idx=0, player_id=1, tokens=[], text="", log_probs=[],
                final_answer="42", correct=True, reward=1.0,
            )],
        }

        apply_persuader_bonus(rollouts, discussion_results, majority_answers, problems, gamma=0.3)

        # Alice was persuader on problem 0: her correct rollouts -> 1.3
        assert all(r.reward == pytest.approx(1.3) for r in alice_rollouts[:6])
        assert all(r.reward == pytest.approx(0.0) for r in alice_rollouts[6:8])
        # Problem 1 unaffected
        assert all(r.reward == pytest.approx(1.0) for r in alice_rollouts[8:])
        # Bob NOT persuader: rewards unchanged
        assert all(r.reward == pytest.approx(1.0) for r in bob_rollouts[:3])
        assert all(r.reward == pytest.approx(0.0) for r in bob_rollouts[3:8])

        # -- Step 8: Dynamic sampling filter --
        alice_filtered = filter_dynamic_sampling(alice_rollouts)
        bob_filtered = filter_dynamic_sampling(bob_rollouts)

        # Problem 0: mixed rewards (1.3, 0.0) -> kept (8 rollouts)
        # Problem 1: all 1.0 -> filtered
        assert len(alice_filtered) == 8
        assert all(r.problem_idx == 0 for r in alice_filtered)
        # Bob problem 0: mixed (1.0, 0.0) -> kept, problem 1: all 1.0 -> filtered
        assert len(bob_filtered) == 8
        assert all(r.problem_idx == 0 for r in bob_filtered)

        # -- Step 9: Advantages --
        alice_pre_rewards = [r.reward for r in alice_filtered]
        alice_post_rewards = [dr.reward for dr in discussion_results[0]]
        alice_ema = EMATracker(mu=0.5, sigma_sq=0.25, eta=0.2)

        alice_pre_adv, alice_post_adv = compute_player_advantages(
            alice_pre_rewards, alice_post_rewards, alice_ema
        )

        # Pre-discussion: rewards are [1.3]*6 + [0.0]*2, normalized by mean/std with floor
        mean_pre = (6 * 1.3 + 2 * 0.0) / 8  # = 0.975
        import numpy as _np
        from crisp.rewards.advantages import MIN_SIGMA
        std_pre = max(float(_np.std([1.3]*6 + [0.0]*2)), MIN_SIGMA)
        for i, r in enumerate(alice_pre_rewards):
            expected = (r - mean_pre) / std_pre
            assert alice_pre_adv[i] == pytest.approx(expected, rel=1e-4)

        # Correct rollouts (1.3) have positive advantage
        assert all(a > 0 for a in alice_pre_adv[:6])
        # Wrong rollouts (0.0) have negative advantage
        assert all(a < 0 for a in alice_pre_adv[6:])

        # Post-discussion: uses EMA mean/std (mu=0.5, sigma=max(sqrt(0.25), MIN_SIGMA))
        import math as _math
        ema_sigma = max(_math.sqrt(0.25), MIN_SIGMA)
        for i, r in enumerate(alice_post_rewards):
            expected = (r - 0.5) / ema_sigma
            assert alice_post_adv[i] == pytest.approx(expected, rel=1e-4)
