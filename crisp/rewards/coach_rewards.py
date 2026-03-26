"""Coach reward computation: uncertainty + discussion - repetition."""
from __future__ import annotations

from typing import List

import numpy as np

from crisp.rewards.repetition_buffer import RepetitionBuffer
from crisp.types import Problem, Rollout


def compute_uncertainty_reward(p_hat: float) -> float:
    """r_uncertainty = 1 - 2|p_hat - 0.5|. Peaks at p_hat=0.5."""
    return 1.0 - 2.0 * abs(p_hat - 0.5)


def compute_discussion_reward(
    discussion_occurred: bool,
    resolved_correctly: bool,
    alpha: float = 0.3,
) -> float:
    """r_discussion based on whether discussion occurred and resolved correctly."""
    if not discussion_occurred:
        return 0.0
    if resolved_correctly:
        return alpha
    return alpha / 2


def compute_intra_batch_penalty(
    idx: int,
    embeddings: List[np.ndarray],
    lambda_rep: float = 1.0,
    tau_sim: float = 0.85,
) -> float:
    """Compute within-batch repetition penalty for problem at idx."""
    if idx < 0 or idx >= len(embeddings):
        raise IndexError(
            f"problem_idx {idx} out of range for {len(embeddings)} embeddings"
        )
    if len(embeddings) <= 1:
        return 0.0

    query = embeddings[idx]
    others = np.stack([e for j, e in enumerate(embeddings) if j != idx])

    query_norm = query / (np.linalg.norm(query) + 1e-10)
    others_norm = others / (np.linalg.norm(others, axis=1, keepdims=True) + 1e-10)

    similarities = others_norm @ query_norm
    count_similar = int(np.sum(similarities > tau_sim))
    return lambda_rep * count_similar / len(others)


def compute_coach_reward(
    problem: Problem,
    problem_idx: int,
    all_embeddings: List[np.ndarray],
    player_rollouts: List[Rollout],
    discussion_occurred: bool,
    resolved_correctly: bool,
    repetition_buffer: RepetitionBuffer,
    alpha: float = 0.3,
    lambda_rep: float = 1.0,
    tau_sim: float = 0.85,
    too_hard_penalty: float = -0.5,
    too_easy_penalty: float = -0.3,
    too_easy_threshold: float = 1.0,
    unsolvable_penalty: float = -0.5,
) -> float:
    """Compute full coach reward for a single problem.

    Short-circuits:
    - not self_solvable: coach can't solve its own problem, return unsolvable_penalty
    - p_hat == 0: too hard, return too_hard_penalty
    - p_hat >= too_easy_threshold: too easy, return too_easy_penalty
    Otherwise: r_coach = max(0, r_uncertainty + r_discussion - r_repetition).
    """
    # Coach couldn't solve its own problem — penalize heavily so it learns
    # to generate problems within its own capability
    if not problem.self_solvable:
        return unsolvable_penalty

    # Solve rate across both players
    correct_count = sum(1 for r in player_rollouts if r.correct)
    p_hat = correct_count / len(player_rollouts) if player_rollouts else 0.0

    # Short-circuit: nobody solved → penalise coach for being too hard
    if p_hat == 0.0:
        return too_hard_penalty

    # Short-circuit: everyone solved → penalise coach for being too easy
    if p_hat >= too_easy_threshold:
        return too_easy_penalty

    r_uncertainty = compute_uncertainty_reward(p_hat)
    r_discussion = compute_discussion_reward(discussion_occurred, resolved_correctly, alpha)

    # Repetition: intra-batch + cross-batch
    r_intra = compute_intra_batch_penalty(problem_idx, all_embeddings, lambda_rep, tau_sim)
    r_cross = repetition_buffer.compute_penalty(problem.coach_embedding, lambda_rep, tau_sim)
    r_repetition = r_intra + r_cross

    return max(0.0, r_uncertainty + r_discussion - r_repetition)
