"""Player reward computation: solve rewards and persuader bonus."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from crisp.types import DiscussionResult, Problem, Rollout
from crisp.verifier.sympy_verify import check


def compute_solve_reward(rollout: Rollout, no_box_penalty: float = -0.5) -> float:
    """Compute pre-discussion reward for a single rollout.

    Returns:
        1.0 if correct, 0.0 if wrong, no_box_penalty if no \\boxed{} answer extracted.
    """
    if rollout.answer is None:
        return no_box_penalty
    if rollout.correct:
        return 1.0
    return 0.0


def apply_persuader_bonus(
    rollouts: Dict[int, List[Rollout]],
    discussion_results: Dict[int, List[DiscussionResult]],
    majority_answers: Dict[Tuple[int, int], str],
    problems: List[Problem],
    gamma: float = 0.3,
) -> None:
    """Apply persuader bonus to correct pre-discussion rollouts in-place.

    Persuader = player whose majority answer was correct AND whose peer
    flipped from wrong to correct after discussion.

    Raises RuntimeError if called twice on the same rollouts.
    """
    # Idempotency guard: check if any rollout already has the bonus flag
    for pid in rollouts:
        for r in rollouts[pid]:
            if r._persuader_bonus_applied:
                raise RuntimeError(
                    "Persuader bonus already applied. "
                    "apply_persuader_bonus must only be called once per batch."
                )

    # Collect discussed problem indices
    discussed_problems = set()
    for pid in discussion_results:
        for dr in discussion_results[pid]:
            discussed_problems.add(dr.problem_idx)

    for prob_idx in discussed_problems:
        persuader_id = _find_persuader(
            prob_idx, majority_answers, discussion_results, problems
        )
        if persuader_id is not None:
            for r in rollouts[persuader_id]:
                if r.problem_idx == prob_idx and r.correct:
                    r.reward += gamma

    # Mark all rollouts as processed
    for pid in rollouts:
        for r in rollouts[pid]:
            r._persuader_bonus_applied = True


def _find_persuader(
    problem_idx: int,
    majority_answers: Dict[Tuple[int, int], str],
    discussion_results: Dict[int, List[DiscussionResult]],
    problems: List[Problem],
) -> Optional[int]:
    """Find the persuader for a discussed problem, if any.

    Returns player_id of the persuader, or None if no persuasion occurred.
    Persuader = player with correct majority answer whose peer flipped to correct.
    """
    ground_truth = problems[problem_idx].ground_truth

    # Find which player(s) had correct majority
    correct_players = []
    for pid in [0, 1]:
        maj = majority_answers.get((pid, problem_idx))
        if maj is not None and check(maj, ground_truth):
            correct_players.append(pid)

    if len(correct_players) != 1:
        # Both correct (shouldn't trigger discussion) or both wrong (no persuader)
        return None

    persuader_id = correct_players[0]
    peer_id = 1 - persuader_id

    # Check if peer flipped to correct
    peer_results = [
        dr for dr in discussion_results.get(peer_id, [])
        if dr.problem_idx == problem_idx
    ]
    if peer_results and peer_results[0].correct:
        return persuader_id

    return None
