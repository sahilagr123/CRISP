"""Tests for player reward computation."""
import pytest

from crisp.types import Rollout, DiscussionResult, Problem
from crisp.rewards.player_rewards import compute_solve_reward, apply_persuader_bonus
from tests.conftest import make_rollout, make_problem


class TestComputeSolveReward:
    def test_correct_answer(self):
        r = make_rollout(correct=True)
        assert compute_solve_reward(r) == 1.0

    def test_wrong_answer(self):
        r = make_rollout(answer="99", correct=False)
        assert compute_solve_reward(r) == 0.0

    def test_no_boxed_answer(self):
        r = make_rollout(answer=None, correct=None)
        assert compute_solve_reward(r) == -0.5

    def test_empty_answer_treated_as_no_box(self):
        """Empty string from \\boxed{} is still an extraction — treat as wrong, not missing."""
        r = make_rollout(answer="", correct=False)
        assert compute_solve_reward(r) == 0.0

    def test_correct_none_with_answer_treated_as_wrong(self):
        """If answer extracted but correct=None (unverified), treat as wrong."""
        r = make_rollout(answer="42", correct=None)
        assert compute_solve_reward(r) == 0.0


class TestApplyPersuaderBonus:
    def _make_scenario(self):
        """2 players, 1 problem, 8 rollouts each.
        Player 0: 6 correct, 2 wrong -> majority correct
        Player 1: 3 correct, 5 wrong -> majority wrong
        """
        problems = [make_problem()]
        rollouts = {
            0: [make_rollout(problem_idx=0, player_id=0, answer="42", correct=True, reward=1.0) for _ in range(6)]
              + [make_rollout(problem_idx=0, player_id=0, answer="99", correct=False, reward=0.0) for _ in range(2)],
            1: [make_rollout(problem_idx=0, player_id=1, answer="42", correct=True, reward=1.0) for _ in range(3)]
              + [make_rollout(problem_idx=0, player_id=1, answer="99", correct=False, reward=0.0) for _ in range(5)],
        }
        majority_answers = {(0, 0): "42", (1, 0): "99"}
        # Player 1 flipped to correct after discussion
        discussion_results = {
            0: [DiscussionResult(problem_idx=0, player_id=0, tokens=[], text="", log_probs=[], final_answer="42", correct=True, reward=1.0)],
            1: [DiscussionResult(problem_idx=0, player_id=1, tokens=[], text="", log_probs=[], final_answer="42", correct=True, reward=1.0)],
        }
        return problems, rollouts, majority_answers, discussion_results

    def test_persuader_bonus_on_correct_rollouts(self):
        problems, rollouts, majority, disc = self._make_scenario()
        apply_persuader_bonus(rollouts, disc, majority, problems, gamma=0.3)
        # Player 0 was persuader: correct rollouts get +0.3
        for r in rollouts[0]:
            if r.correct:
                assert r.reward == pytest.approx(1.3)
            else:
                assert r.reward == pytest.approx(0.0)

    def test_no_bonus_on_wrong_player_rollouts(self):
        problems, rollouts, majority, disc = self._make_scenario()
        apply_persuader_bonus(rollouts, disc, majority, problems, gamma=0.3)
        # Player 1 was NOT the persuader — no bonus
        for r in rollouts[1]:
            if r.correct:
                assert r.reward == pytest.approx(1.0)
            else:
                assert r.reward == pytest.approx(0.0)

    def test_no_bonus_when_peer_didnt_flip(self):
        """If peer stayed wrong, no persuader bonus even if one player was correct."""
        problems, rollouts, majority, disc = self._make_scenario()
        # Peer didn't flip — still wrong after discussion
        disc[1][0].correct = False
        disc[1][0].final_answer = "99"
        apply_persuader_bonus(rollouts, disc, majority, problems, gamma=0.3)
        for r in rollouts[0]:
            if r.correct:
                assert r.reward == pytest.approx(1.0)  # No bonus

    def test_idempotency_guard(self):
        """Calling apply_persuader_bonus twice should raise."""
        problems, rollouts, majority, disc = self._make_scenario()
        apply_persuader_bonus(rollouts, disc, majority, problems, gamma=0.3)
        with pytest.raises(RuntimeError, match="already applied"):
            apply_persuader_bonus(rollouts, disc, majority, problems, gamma=0.3)

    def test_no_discussion_no_bonus(self):
        """If no discussion occurred, nothing changes."""
        problems = [make_problem()]
        rollouts = {
            0: [make_rollout(problem_idx=0, player_id=0, correct=True, reward=1.0) for _ in range(8)],
            1: [make_rollout(problem_idx=0, player_id=1, correct=True, reward=1.0) for _ in range(8)],
        }
        majority = {(0, 0): "42", (1, 0): "42"}
        disc = {0: [], 1: []}
        apply_persuader_bonus(rollouts, disc, majority, problems, gamma=0.3)
        for pid in [0, 1]:
            for r in rollouts[pid]:
                assert r.reward == pytest.approx(1.0)

    def test_both_wrong_no_bonus(self):
        """If both players had wrong majority, no persuader bonus."""
        problems = [make_problem()]
        rollouts = {
            0: [make_rollout(problem_idx=0, player_id=0, answer="99", correct=False, reward=0.0) for _ in range(8)],
            1: [make_rollout(problem_idx=0, player_id=1, answer="88", correct=False, reward=0.0) for _ in range(8)],
        }
        majority = {(0, 0): "99", (1, 0): "88"}
        disc = {
            0: [DiscussionResult(problem_idx=0, player_id=0, tokens=[], text="", log_probs=[], final_answer="99", correct=False, reward=0.0)],
            1: [DiscussionResult(problem_idx=0, player_id=1, tokens=[], text="", log_probs=[], final_answer="88", correct=False, reward=0.0)],
        }
        apply_persuader_bonus(rollouts, disc, majority, problems, gamma=0.3)
        for pid in [0, 1]:
            for r in rollouts[pid]:
                assert r.reward == pytest.approx(0.0)
