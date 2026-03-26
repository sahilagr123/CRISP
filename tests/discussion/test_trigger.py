"""Tests for majority vote and discussion trigger."""
import pytest

from crisp.discussion.trigger import majority_vote, should_discuss
from tests.conftest import make_rollout


class TestMajorityVote:
    def test_unanimous(self):
        rollouts = [make_rollout(answer="42") for _ in range(8)]
        assert majority_vote(rollouts) == "42"

    def test_clear_majority(self):
        rollouts = (
            [make_rollout(answer="42") for _ in range(5)]
            + [make_rollout(answer="99") for _ in range(3)]
        )
        assert majority_vote(rollouts) == "42"

    def test_tie_breaks_by_first_occurrence(self):
        rollouts = [
            make_rollout(answer="A"),
            make_rollout(answer="B"),
            make_rollout(answer="A"),
            make_rollout(answer="B"),
        ]
        assert majority_vote(rollouts) == "A"

    def test_all_different(self):
        rollouts = [make_rollout(answer=str(i)) for i in range(4)]
        assert majority_vote(rollouts) == "0"  # First occurrence

    def test_none_answers_ignored(self):
        rollouts = (
            [make_rollout(answer=None) for _ in range(5)]
            + [make_rollout(answer="42") for _ in range(3)]
        )
        assert majority_vote(rollouts) == "42"

    def test_all_none_returns_none(self):
        rollouts = [make_rollout(answer=None) for _ in range(4)]
        assert majority_vote(rollouts) is None

    def test_symbolic_equivalence_grouping(self):
        """Answers that are symbolically equivalent should be grouped."""
        rollouts = [
            make_rollout(answer="1/2"),
            make_rollout(answer="0.5"),
            make_rollout(answer="0.5"),
            make_rollout(answer="99"),
        ]
        # "1/2" and "0.5" are equivalent → group has 3, "99" has 1
        result = majority_vote(rollouts)
        assert result in ("1/2", "0.5")  # Either representative is fine


class TestShouldDiscuss:
    def test_agreeing_players_no_discussion(self):
        assert should_discuss("42", "42") is False

    def test_disagreeing_players_trigger_discussion(self):
        assert should_discuss("42", "99") is True

    def test_equivalent_answers_no_discussion(self):
        assert should_discuss("1/2", "0.5") is False

    def test_none_majority_triggers_discussion(self):
        """If one player has no majority (all None), trigger discussion."""
        assert should_discuss("42", None) is True

    def test_both_none_no_discussion(self):
        """Both None → agree on nothing → no discussion."""
        assert should_discuss(None, None) is False


class TestMajorityVoteEdgeCases:
    def test_empty_rollout_list(self):
        """Empty list should return None."""
        assert majority_vote([]) is None

    def test_single_rollout(self):
        """Single rollout → that answer is the majority."""
        assert majority_vote([make_rollout(answer="42")]) == "42"
