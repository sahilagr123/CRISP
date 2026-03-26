"""Steps 5-6: Discussion trigger and execution."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from crisp.discussion.post_discussion import parse_discussion_response
from crisp.training.overlong_shaping import compute_overlong_penalty
from crisp.discussion.representative import select_representatives
from crisp.discussion.trigger import majority_vote, should_discuss
from crisp.infra.experience import generate_samples
from crisp.types import DiscussionResult, Problem, Rollout
from crisp.verifier.sympy_verify import check


def run_discussion(
    ctx: Any,
    rollouts: Dict[int, List[Rollout]],
    problems: List[Problem],
    discussion_prompt_ids: Optional[Dict[int, List[List[int]]]] = None,
) -> Tuple[Dict[int, List[DiscussionResult]], Dict[Tuple[int, int], str]]:
    """Run the discussion step for all problems.

    For each problem:
    1. Compute majority votes per player
    2. Check if players disagree (should_discuss)
    3. If disagreement: select representatives, generate discussion, parse results
    4. If agreement: skip discussion

    Returns:
        (discussion_results, majority_answers) where:
        - discussion_results: player_id -> list of DiscussionResult
        - majority_answers: (player_id, problem_idx) -> majority answer string
    """
    majority_answers: Dict[Tuple[int, int], str] = {}
    discussion_results: Dict[int, List[DiscussionResult]] = {0: [], 1: []}
    discussed_problems: List[int] = []

    # Group rollouts by (player_id, problem_idx)
    grouped: Dict[Tuple[int, int], List[Rollout]] = {}
    for pid in rollouts:
        for r in rollouts[pid]:
            key = (pid, r.problem_idx)
            grouped.setdefault(key, []).append(r)

    # Step 5: Compute majority votes and identify disagreements
    for prob_idx, problem in enumerate(problems):
        for pid in [0, 1]:
            player_rollouts = grouped.get((pid, prob_idx), [])
            maj = majority_vote(player_rollouts)
            if maj is not None:
                majority_answers[(pid, prob_idx)] = maj

        maj_a = majority_answers.get((0, prob_idx))
        maj_b = majority_answers.get((1, prob_idx))

        if should_discuss(maj_a, maj_b):
            discussed_problems.append(prob_idx)

    if not discussed_problems:
        return discussion_results, majority_answers

    # Step 6: Select representatives and generate discussion responses
    all_disc_prompts = []
    disc_prompt_metadata = []  # Track (problem_idx, player_id) per prompt

    for prob_idx in discussed_problems:
        prob_rollouts = {
            pid: grouped.get((pid, prob_idx), [])
            for pid in [0, 1]
        }
        reps = select_representatives(
            prob_rollouts, majority_answers,
            problems[prob_idx].ground_truth, prob_idx,
        )

        template = ctx.config.coach.discussion_template

        for pid in [0, 1]:
            if pid not in reps:
                continue
            other_pid = 1 - pid
            own_rep = reps.get(pid)
            other_rep = reps.get(other_pid)
            if own_rep is None or other_rep is None:
                continue

            prompt_text = template.format(
                problem=problems[prob_idx].text,
                own_solution=own_rep.text,
                other_solution=other_rep.text,
            )
            all_disc_prompts.append(prompt_text)
            disc_prompt_metadata.append((prob_idx, pid))

    if not all_disc_prompts:
        return discussion_results, majority_answers

    # Tokenize all discussion prompts upfront
    if discussion_prompt_ids is not None:
        all_disc_token_ids = discussion_prompt_ids
    elif getattr(ctx, 'tokenizer', None) is not None:
        from crisp.workflow.tokenizer import build_discussion_prompts
        all_disc_token_ids = build_discussion_prompts(ctx.tokenizer, ctx.config, all_disc_prompts)
    else:
        all_disc_token_ids = [[] for _ in all_disc_prompts]

    # Group prompts by player_id for per-player weight sync
    alice_indices = []  # indices into all_disc_prompts
    bob_indices = []
    for i, (prob_idx, pid) in enumerate(disc_prompt_metadata):
        if pid == 0:
            alice_indices.append(i)
        else:
            bob_indices.append(i)

    # Generate per-player: sync weights, then generate
    disc_rollouts: List[Optional[Rollout]] = [None] * len(all_disc_prompts)

    ds_alice = getattr(ctx, 'ds_alice', None)
    ds_bob = getattr(ctx, 'ds_bob', None)

    for pid, indices, ds_model in [
        (0, alice_indices, ds_alice),
        (1, bob_indices, ds_bob),
    ]:
        if not indices:
            continue

        # Sync this player's weights to vLLM before generating
        if ds_model is not None:
            ds_model.sync_weights(ctx.player_vllm)

        player_token_ids = [all_disc_token_ids[i] for i in indices]
        player_problem_indices = [disc_prompt_metadata[i][0] for i in indices]

        player_rollouts = generate_samples(
            ctx.player_vllm,
            prompt_token_ids=player_token_ids,
            problem_indices=player_problem_indices,
            player_id=pid,
            max_new_tokens=2048,
        )
        for idx, rollout in zip(indices, player_rollouts):
            disc_rollouts[idx] = rollout

    # Parse discussion responses into DiscussionResults
    for i, (rollout, (prob_idx, pid)) in enumerate(zip(disc_rollouts, disc_prompt_metadata)):
        evaluation_text, final_answer = parse_discussion_response(rollout.text)
        correct = check(final_answer, problems[prob_idx].ground_truth) if final_answer else False

        dr = DiscussionResult(
            problem_idx=prob_idx,
            player_id=pid,
            tokens=rollout.tokens,
            text=rollout.text,
            log_probs=rollout.log_probs,
            evaluation_text=evaluation_text,
            final_answer=final_answer,
            correct=correct,
            reward=1.0 if correct else 0.0,
        )
        # Overlong penalty for post-discussion responses
        grpo_cfg = ctx.config.grpo
        penalty = compute_overlong_penalty(
            len(rollout.tokens),
            l_max=grpo_cfg.post_discussion_l_max,
            buffer=grpo_cfg.post_discussion_buffer,
        )
        dr.reward -= penalty

        discussion_results[pid].append(dr)

    return discussion_results, majority_answers
