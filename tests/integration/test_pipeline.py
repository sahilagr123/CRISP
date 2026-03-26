"""Integration tests: full pipeline with mock data, conservation invariants."""
import math
import numpy as np
import pytest

from crisp.types import Problem, Rollout, DiscussionResult
from crisp.rewards.player_rewards import compute_solve_reward, apply_persuader_bonus
from crisp.rewards.advantages import compute_player_advantages
from crisp.rewards.coach_rewards import compute_coach_reward
from crisp.rewards.ema_tracker import EMATracker
from crisp.rewards.repetition_buffer import RepetitionBuffer
from crisp.discussion.trigger import majority_vote, should_discuss
from crisp.training.batch_builder import filter_dynamic_sampling
from tests.conftest import make_rollout, make_problem


def _build_synthetic_batch(rng, num_problems=8, rollouts_per=8):
    """Build a full synthetic batch with controlled solve rates."""
    problems = []
    all_rollouts = {0: [], 1: []}
    solve_rates = {
        0: rng.uniform(0.2, 0.8, num_problems),
        1: rng.uniform(0.2, 0.8, num_problems),
    }

    for pidx in range(num_problems):
        gt = str(pidx * 10 + 1)
        problems.append(make_problem(
            text=f"Problem {pidx}",
            ground_truth=gt,
            embedding=rng.random(384).astype(np.float32),
        ))
        for player_id in [0, 1]:
            rate = solve_rates[player_id][pidx]
            for _ in range(rollouts_per):
                correct = rng.random() < rate
                answer = gt if correct else "WRONG"
                all_rollouts[player_id].append(make_rollout(
                    problem_idx=pidx,
                    player_id=player_id,
                    answer=answer,
                    correct=correct,
                    text=f"Solution for problem {pidx}",
                ))
    return problems, all_rollouts


@pytest.mark.integration
class TestPipelineConservation:
    def test_batch_size_conservation(self):
        """Total sequences = (non-filtered problems x 8) + discussion count per player."""
        rng = np.random.default_rng(123)
        problems, rollouts = _build_synthetic_batch(rng)

        # Step 4: Compute rewards
        for pid in rollouts:
            for r in rollouts[pid]:
                r.reward = compute_solve_reward(r)

        # Step 5: Majority vote + discussion trigger
        majority_answers = {}
        discuss_problems = []
        for pidx in range(len(problems)):
            for pid in [0, 1]:
                player_rs = [r for r in rollouts[pid] if r.problem_idx == pidx]
                majority_answers[(pid, pidx)] = majority_vote(player_rs)
            if should_discuss(majority_answers[(0, pidx)], majority_answers[(1, pidx)]):
                discuss_problems.append(pidx)

        # Step 6-7: Mock discussion results
        discussion_results = {0: [], 1: []}
        for pidx in discuss_problems:
            for pid in [0, 1]:
                correct = rng.random() > 0.3
                discussion_results[pid].append(DiscussionResult(
                    problem_idx=pidx, player_id=pid, tokens=[], text="",
                    log_probs=[], final_answer=problems[pidx].ground_truth if correct else "WRONG",
                    correct=correct, reward=1.0 if correct else 0.0,
                ))
        apply_persuader_bonus(rollouts, discussion_results, majority_answers, problems)

        # Step 8: Dynamic sampling
        for pid in [0, 1]:
            filtered = filter_dynamic_sampling(rollouts[pid])
            non_filtered_problems = set(r.problem_idx for r in filtered)

            # Conservation: filtered rollouts = non_filtered_problems x 8
            assert len(filtered) == len(non_filtered_problems) * 8

            # Total batch = filtered pre-discussion + discussion
            total = len(filtered) + len(discussion_results[pid])
            assert total > 0, "Batch should not be empty"

    def test_no_nan_advantages(self):
        """Every sequence in the batch should have a finite advantage."""
        rng = np.random.default_rng(456)
        problems, rollouts = _build_synthetic_batch(rng)

        for pid in rollouts:
            for r in rollouts[pid]:
                r.reward = compute_solve_reward(r)

        majority_answers = {}
        discuss_problems = []
        for pidx in range(len(problems)):
            for pid in [0, 1]:
                player_rs = [r for r in rollouts[pid] if r.problem_idx == pidx]
                majority_answers[(pid, pidx)] = majority_vote(player_rs)
            if should_discuss(majority_answers[(0, pidx)], majority_answers[(1, pidx)]):
                discuss_problems.append(pidx)

        discussion_results = {0: [], 1: []}
        for pidx in discuss_problems:
            for pid in [0, 1]:
                discussion_results[pid].append(DiscussionResult(
                    problem_idx=pidx, player_id=pid, tokens=[], text="",
                    log_probs=[], final_answer=problems[pidx].ground_truth,
                    correct=True, reward=1.0,
                ))
        apply_persuader_bonus(rollouts, discussion_results, majority_answers, problems)

        for pid in [0, 1]:
            filtered = filter_dynamic_sampling(rollouts[pid])
            pre_rewards = [r.reward for r in filtered]
            post_rewards = [dr.reward for dr in discussion_results[pid]]
            ema = EMATracker()

            pre_adv, post_adv = compute_player_advantages(pre_rewards, post_rewards, ema)

            for a in pre_adv:
                assert math.isfinite(a), f"Non-finite pre-discussion advantage: {a}"
            for a in post_adv:
                assert math.isfinite(a), f"Non-finite post-discussion advantage: {a}"

    def test_coach_rewards_all_valid(self):
        """All coach rewards should be non-negative and finite."""
        rng = np.random.default_rng(789)
        problems, rollouts = _build_synthetic_batch(rng)

        for pid in rollouts:
            for r in rollouts[pid]:
                r.reward = compute_solve_reward(r)

        buf = RepetitionBuffer(max_batches=10, embedding_dim=384)
        all_embeddings = [p.coach_embedding for p in problems]

        for pidx in range(len(problems)):
            all_player_rollouts = (
                [r for r in rollouts[0] if r.problem_idx == pidx]
                + [r for r in rollouts[1] if r.problem_idx == pidx]
            )
            reward = compute_coach_reward(
                problem=problems[pidx],
                problem_idx=pidx,
                all_embeddings=all_embeddings,
                player_rollouts=all_player_rollouts,
                discussion_occurred=False,
                resolved_correctly=False,
                repetition_buffer=buf,
            )
            assert math.isfinite(reward), f"Non-finite coach reward: {reward}"
            assert reward >= 0.0, f"Negative coach reward: {reward}"
