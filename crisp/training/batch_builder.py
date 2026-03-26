"""Dynamic sampling filter and training batch assembly."""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional

from crisp.types import DiscussionResult, Problem, Rollout, TokenSequence, TrainingBatch


def filter_no_box(rollouts: List[Rollout], no_box_penalty: float = -0.5) -> List[Rollout]:
    """Filter out rollouts that failed to produce a \\boxed{} answer.

    These are often garbled/word-salad outputs whose gradients are incoherent
    and can cause cascading policy collapse. Wrong-but-boxed answers (reward=0)
    already provide the 'that's wrong' signal.
    """
    kept = [r for r in rollouts if r.answer is not None]
    n_dropped = len(rollouts) - len(kept)
    if n_dropped > 0:
        import logging
        logging.getLogger(__name__).info(
            "filter_no_box: dropped %d/%d rollouts (no boxed answer)",
            n_dropped, len(rollouts),
        )
    return kept


def filter_dynamic_sampling(rollouts: List[Rollout]) -> List[Rollout]:
    """Filter out problems where all rollouts have identical rewards.

    These produce zero advantage variance and waste gradient computation.
    """
    # Group by problem
    by_problem: dict[int, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        by_problem[r.problem_idx].append(r)

    result = []
    for prob_idx, prob_rollouts in by_problem.items():
        rewards = [r.reward for r in prob_rollouts]
        if len(set(rewards)) > 1:
            result.extend(prob_rollouts)

    return result


def build_player_batch(
    rollouts: List[Rollout],
    pre_advantages: List[float],
    discussion_results: Optional[List[DiscussionResult]] = None,
    post_advantages: Optional[List[float]] = None,
) -> TrainingBatch:
    """Build a training batch from player rollouts and optional discussion results.

    Combines pre-discussion rollouts and post-discussion results into a single
    TrainingBatch. Each sequence's log_probs become the "old" log-probs for
    importance ratio computation.
    """
    sequences: List[TokenSequence] = []
    advantages: List[float] = []
    is_post: List[bool] = []

    for rollout, adv in zip(rollouts, pre_advantages):
        sequences.append(TokenSequence(
            tokens=rollout.tokens,
            log_probs=rollout.log_probs,
            text=rollout.text,
        ))
        advantages.append(adv)
        is_post.append(False)

    if discussion_results and post_advantages:
        for dr, adv in zip(discussion_results, post_advantages):
            sequences.append(TokenSequence(
                tokens=dr.tokens,
                log_probs=dr.log_probs,
                text=dr.text,
            ))
            advantages.append(adv)
            is_post.append(True)

    return TrainingBatch(
        sequences=sequences,
        advantages=advantages,
        ref_log_probs=[],  # Populated later by ref_model.forward()
        is_post_discussion=is_post,
    )


def build_coach_batch(
    problems: List[Problem],
    advantages: List[float],
) -> TrainingBatch:
    """Build a training batch from coach problem sequences.

    Skips problems without a coach_sequence.
    """
    sequences: List[TokenSequence] = []
    filtered_advantages: List[float] = []

    for problem, adv in zip(problems, advantages):
        if problem.coach_sequence is None:
            continue
        sequences.append(problem.coach_sequence)
        filtered_advantages.append(adv)

    return TrainingBatch(
        sequences=sequences,
        advantages=filtered_advantages,
        ref_log_probs=[],
        is_post_discussion=[False] * len(sequences),
    )
