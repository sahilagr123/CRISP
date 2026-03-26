"""experience.py — Maps vLLM outputs to CRISP Rollout objects.

This is the sole boundary between vLLM inference output and CRISP domain types.
All downstream code works exclusively with Rollout from crisp.types.
"""
from __future__ import annotations

from typing import Any, List, Optional

from crisp.types import Rollout


def map_vllm_output_to_rollout(
    vllm_output: Any,
    problem_idx: int,
    player_id: int,
) -> Rollout:
    """Convert a single vLLM RequestOutput into a CRISP Rollout.

    Prompt tokens get log_prob=0.0 (no model score for the prompt).
    Response tokens use the logprob values from vLLM when available,
    otherwise default to 0.0.

    Args:
        vllm_output: A vLLM RequestOutput (or mock with matching interface).
        problem_idx: Index of the problem this rollout answers.
        player_id: 0 = Alice, 1 = Bob.

    Returns:
        A Rollout with text from vLLM's decoded completion.
    """
    prompt_token_ids: List[int] = list(vllm_output.prompt_token_ids)
    completion = vllm_output.outputs[0]
    text = getattr(completion, 'text', '') or ''
    output_token_ids: List[int] = list(completion.token_ids)

    # Prompt tokens have no model-assigned log-prob.
    prompt_log_probs = [0.0] * len(prompt_token_ids)

    # Extract response log-probs from vLLM's per-token logprob dicts.
    response_log_probs: List[float] = []
    if completion.logprobs is not None:
        for tid, lp_dict in zip(output_token_ids, completion.logprobs):
            entry = lp_dict.get(tid)
            response_log_probs.append(entry.logprob if entry is not None else 0.0)
    else:
        response_log_probs = [0.0] * len(output_token_ids)

    return Rollout(
        problem_idx=problem_idx,
        player_id=player_id,
        tokens=prompt_token_ids + output_token_ids,
        log_probs=prompt_log_probs + response_log_probs,
        text=text,
        reward=0.0,
        prompt_len=len(prompt_token_ids),
    )


def generate_samples(
    vllm_engines: List[Any],
    prompt_token_ids: List[List[int]],
    problem_indices: List[int],
    player_id: int,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
) -> List[Rollout]:
    """Distribute prompts across vLLM engines and collect Rollouts.

    Prompts are split round-robin across the available engines. Each engine
    is called via Ray remote methods (add_requests / get_responses).

    Args:
        vllm_engines: List of Ray actor handles wrapping vLLM engines.
        prompt_token_ids: Token-id sequences for each prompt.
        problem_indices: Problem index corresponding to each prompt.
        player_id: 0 = Alice, 1 = Bob.
        max_new_tokens: Maximum new tokens to generate per sample.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling (-1 disables).

    Returns:
        List of Rollout objects, one per prompt, in the original prompt order.
    """
    import ray
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        logprobs=1,
    )

    num_engines = len(vllm_engines)

    # Round-robin partition: assign each prompt to an engine.
    engine_prompts: List[List[List[int]]] = [[] for _ in range(num_engines)]
    engine_indices: List[List[int]] = [[] for _ in range(num_engines)]
    for i, (tokens, pidx) in enumerate(zip(prompt_token_ids, problem_indices)):
        eidx = i % num_engines
        engine_prompts[eidx].append(tokens)
        engine_indices[eidx].append(pidx)

    # Generate from all engines in parallel.
    gen_refs = []
    for eidx, engine in enumerate(vllm_engines):
        ref = engine.generate.remote(sampling_params, engine_prompts[eidx])
        gen_refs.append(ref)
    all_responses = ray.get(gen_refs)

    # Map vLLM outputs to Rollouts, preserving original order.
    # Build a mapping from original prompt index to rollout.
    rollouts: List[Optional[Rollout]] = [None] * len(prompt_token_ids)
    global_cursor = [0] * num_engines  # track position within each engine's batch

    for i in range(len(prompt_token_ids)):
        eidx = i % num_engines
        local_idx = global_cursor[eidx]
        global_cursor[eidx] += 1
        vllm_output = all_responses[eidx][local_idx]
        pidx = engine_indices[eidx][local_idx]
        rollouts[i] = map_vllm_output_to_rollout(vllm_output, pidx, player_id)

    return rollouts  # type: ignore[return-value]
