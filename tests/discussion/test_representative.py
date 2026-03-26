"""Tests for representative rollout selection for discussion."""
import pytest

from crisp.discussion.representative import select_representatives
from tests.conftest import make_rollout


class TestSelectRepresentatives:
    def test_correct_player_gets_highest_logprob(self):
        """Player with correct majority → rollout with highest total log-prob."""
        rollouts = {
            0: [
                make_rollout(player_id=0, answer="42", correct=True, log_probs=[-1.0, -1.0]),
                make_rollout(player_id=0, answer="42", correct=True, log_probs=[-0.1, -0.1]),  # Highest
                make_rollout(player_id=0, answer="42", correct=True, log_probs=[-0.5, -0.5]),
            ],
            1: [
                make_rollout(player_id=1, answer="99", correct=False, log_probs=[-0.5, -0.5]),
                make_rollout(player_id=1, answer="99", correct=False, log_probs=[-0.3, -0.3]),
            ],
        }
        majority = {(0, 0): "42", (1, 0): "99"}
        reps = select_representatives(rollouts, majority, "42", problem_idx=0)
        # Player 0 (correct): highest log-prob correct rollout
        assert sum(reps[0].log_probs) == pytest.approx(-0.2)
        # Player 1 (wrong): rollout matching their majority answer
        assert reps[1].answer == "99"

    def test_both_wrong_gets_longest(self):
        """Both players wrong → longest rollout from each."""
        rollouts = {
            0: [
                make_rollout(player_id=0, answer="99", correct=False, text="short"),
                make_rollout(player_id=0, answer="99", correct=False, text="this is a much longer response"),
            ],
            1: [
                make_rollout(player_id=1, answer="88", correct=False, text="x"),
                make_rollout(player_id=1, answer="88", correct=False, text="longer text here"),
            ],
        }
        majority = {(0, 0): "99", (1, 0): "88"}
        reps = select_representatives(rollouts, majority, "42", problem_idx=0)
        assert reps[0].text == "this is a much longer response"
        assert reps[1].text == "longer text here"

    def test_wrong_player_gets_majority_matching_rollout(self):
        """Wrong player gets a rollout matching their majority answer."""
        rollouts = {
            0: [make_rollout(player_id=0, answer="42", correct=True, log_probs=[-0.1, -0.1])],
            1: [
                make_rollout(player_id=1, answer="99", correct=False),
                make_rollout(player_id=1, answer="88", correct=False),
                make_rollout(player_id=1, answer="99", correct=False),
            ],
        }
        majority = {(0, 0): "42", (1, 0): "99"}
        reps = select_representatives(rollouts, majority, "42", problem_idx=0)
        assert reps[1].answer == "99"
