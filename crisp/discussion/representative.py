"""Select representative rollouts for discussion."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from crisp.types import Rollout
from crisp.verifier.sympy_verify import check, equivalent


def select_representatives(
    rollouts: Dict[int, List[Rollout]],
    majority_answers: Dict[Tuple[int, int], str],
    ground_truth: str,
    problem_idx: int,
) -> Dict[int, Rollout]:
    """Select one representative rollout per player for discussion.

    For the player with correct majority: highest log-prob correct rollout.
    For the player with wrong majority: rollout matching their majority answer.
    If both wrong: longest rollout from each.

    Args:
        rollouts: player_id -> list of rollouts for this problem.
        majority_answers: (player_id, problem_idx) -> majority answer string.
        ground_truth: The coach's verified answer.
        problem_idx: Which problem we're selecting for.

    Returns:
        Dict mapping player_id -> selected Rollout.
    """
    player_rollouts = {
        pid: [r for r in rs if r.problem_idx == problem_idx]
        for pid, rs in rollouts.items()
    }

    correct_players = [
        pid for pid in player_rollouts
        if check(majority_answers.get((pid, problem_idx)), ground_truth)
    ]

    reps = {}
    if not correct_players:
        # Both wrong: longest rollout from each
        for pid, rs in player_rollouts.items():
            reps[pid] = max(rs, key=lambda r: len(r.text))
    else:
        for pid, rs in player_rollouts.items():
            if pid in correct_players:
                # Correct player: highest log-prob among correct rollouts
                correct_rollouts = [r for r in rs if r.correct]
                if correct_rollouts:
                    reps[pid] = max(correct_rollouts, key=lambda r: sum(r.log_probs))
                else:
                    reps[pid] = max(rs, key=lambda r: len(r.text))
            else:
                # Wrong player: rollout matching their majority answer
                maj = majority_answers.get((pid, problem_idx))
                matching = [r for r in rs if equivalent(r.answer, maj)] if maj else []
                if matching:
                    reps[pid] = matching[0]
                else:
                    reps[pid] = max(rs, key=lambda r: len(r.text))

    return reps
