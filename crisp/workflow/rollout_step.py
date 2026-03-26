"""Steps 2-4: Player rollout generation, verification, and reward computation."""
from __future__ import annotations

from typing import Dict, List

from crisp.infra.experience import generate_samples
from crisp.rewards.player_rewards import compute_solve_reward
from crisp.training.overlong_shaping import compute_overlong_penalty
from crisp.types import Problem, Rollout
from crisp.verifier.answer_extraction import extract_answer
from crisp.verifier.sympy_verify import check


def _get_player_temperature(ctx, player_id: int) -> float:
    """Get the configured temperature for a player."""
    pcfg = ctx.config.player
    if player_id == 0:
        return pcfg.alice_temperature
    else:
        return pcfg.bob_temperature


def generate_rollouts(
    ctx,
    problems: List[Problem],
    player_id: int,
    prompt_token_ids: List[List[int]] = None,
) -> List[Rollout]:
    """Generate player rollouts, verify answers, and compute rewards.

    For each rollout:
    1. Extract answer via extract_boxed()
    2. Check correctness against ground_truth
    3. Compute solve reward (1.0 / 0.0 / -0.5)
    4. Subtract overlong penalty if applicable
    """
    if prompt_token_ids is None and getattr(ctx, 'tokenizer', None) is not None:
        from crisp.workflow.tokenizer import build_player_prompts
        prompt_token_ids = build_player_prompts(
            ctx.tokenizer, ctx.config, problems, player_id,
        )
    elif prompt_token_ids is None:
        prompt_token_ids = []

    # Prompts are grouped: rpg copies per problem [P0,P0,..,P1,P1,..]
    rpg = ctx.config.player.rollouts_per_problem
    problem_indices = []
    for i, _prompt in enumerate(prompt_token_ids):
        problem_indices.append(i // rpg if problems else 0)

    temperature = _get_player_temperature(ctx, player_id)

    rollouts = generate_samples(
        ctx.player_vllm,
        prompt_token_ids=prompt_token_ids,
        problem_indices=problem_indices,
        player_id=player_id,
        temperature=temperature,
        max_new_tokens=ctx.config.player.max_new_tokens,
    )

    grpo_cfg = ctx.config.grpo

    max_tok = ctx.config.player.max_new_tokens

    for rollout in rollouts:
        prob = problems[rollout.problem_idx]

        # Step 3: Extract and verify answer
        # If the generation hit the token limit, the model was cut off
        # mid-reasoning — don't trust fallback patterns in that case.
        # Compare response tokens only (total - prompt), not total tokens.
        response_len = len(rollout.tokens) - rollout.prompt_len
        truncated = response_len >= max_tok
        rollout.answer = extract_answer(rollout.text, truncated=truncated)
        if rollout.answer is not None:
            rollout.correct = check(rollout.answer, prob.ground_truth)
        else:
            rollout.correct = None

        # Step 4: Compute reward
        rollout.reward = compute_solve_reward(
            rollout, no_box_penalty=ctx.config.player.no_box_penalty,
        )

        # Overlong penalty
        penalty = compute_overlong_penalty(
            len(rollout.tokens),
            l_max=grpo_cfg.pre_discussion_l_max,
            buffer=grpo_cfg.pre_discussion_buffer,
        )
        rollout.reward -= penalty

    return rollouts


def generate_all_rollouts(
    ctx,
    problems: List[Problem],
    prompt_token_ids: Dict[int, List[List[int]]] = None,
) -> Dict[int, List[Rollout]]:
    """Generate rollouts for both players (Alice=0, Bob=1)."""
    if prompt_token_ids is None and getattr(ctx, 'tokenizer', None) is not None:
        from crisp.workflow.tokenizer import build_player_prompts
        prompt_token_ids = {
            pid: build_player_prompts(ctx.tokenizer, ctx.config, problems, pid)
            for pid in [0, 1]
        }
    elif prompt_token_ids is None:
        prompt_token_ids = {0: None, 1: None}
    return {
        pid: generate_rollouts(ctx, problems, player_id=pid,
                               prompt_token_ids=prompt_token_ids.get(pid))
        for pid in [0, 1]
    }
