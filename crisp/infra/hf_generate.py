"""HF generate for DS models (coach generation without vLLM).

Used when coach and player are different models on a single GPU,
so a separate vLLM engine can't be created.
"""
from __future__ import annotations

import logging
from typing import List

import torch

from crisp.types import Rollout

logger = logging.getLogger(__name__)


def generate_from_ds_model(
    ds_model,
    tokenizer,
    prompt_token_ids: List[List[int]],
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
) -> List[Rollout]:
    """Generate from a DeepSpeed-wrapped Actor using HF generate.

    Generates one sequence at a time (coach only makes ~4 problems,
    so throughput isn't critical). Returns Rollout objects with tokens
    and log_probs suitable for GRPO training.
    """
    actor = ds_model.module  # Actor wrapping HF model
    hf_model = actor.model   # The actual HF CausalLM
    device = next(hf_model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # Switch to generation mode
    hf_model.config.use_cache = True
    was_training = hf_model.training
    hf_model.eval()

    rollouts = []
    try:
        for i, prompt in enumerate(prompt_token_ids):
            input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
            attn_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                gen_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_id,
                )
                if temperature > 0:
                    gen_kwargs.update(do_sample=True, temperature=temperature)
                else:
                    gen_kwargs.update(do_sample=False)

                output_ids = hf_model.generate(**gen_kwargs)
                # output_ids: [1, prompt_len + response_len]

                # Forward pass for old-policy log_probs (needed by GRPO)
                full_attn = torch.ones_like(output_ids)
                lp = actor.forward(output_ids, attention_mask=full_attn)
                # lp: [1, T-1] (next-token log probs)

            prompt_len = len(prompt)
            all_tokens = output_ids[0].tolist()
            response_tokens = all_tokens[prompt_len:]
            text = tokenizer.decode(response_tokens, skip_special_tokens=True)

            # Log probs: 0.0 for prompt, model log_probs for response.
            # lp[0, j] is the log-prob of token at position j+1.
            # Response starts at position prompt_len, so its log-prob
            # is at lp[0, prompt_len - 1].
            prompt_lps = [0.0] * prompt_len
            resp_lps = [
                lp[0, prompt_len - 1 + j].item()
                for j in range(len(response_tokens))
            ]

            rollouts.append(Rollout(
                problem_idx=i,
                player_id=-1,
                tokens=all_tokens,
                log_probs=prompt_lps + resp_lps,
                text=text,
                reward=0.0,
                prompt_len=prompt_len,
            ))
            logger.debug("Coach HF gen %d: %d response tokens", i, len(response_tokens))

            # Free KV cache between sequences to prevent OOM from cumulative
            # fragmentation (14B model × 4096 tokens = ~25GB KV cache per seq)
            del input_ids, attn_mask, output_ids, full_attn, lp
            torch.cuda.empty_cache()
    finally:
        hf_model.config.use_cache = False
        if was_training:
            hf_model.train()
        # Free KV cache memory from generate() — critical for 14B model
        # with long sequences (up to 16K tokens)
        torch.cuda.empty_cache()

    avg_resp = (
        sum(len(r.tokens) - len(p) for r, p in zip(rollouts, prompt_token_ids))
        / max(len(rollouts), 1)
    )
    logger.info("HF generate: %d sequences, avg %.0f response tokens", len(rollouts), avg_resp)
    return rollouts
