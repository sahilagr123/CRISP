"""Steps 7-12.5: Player and coach training."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)

from crisp.rewards.advantages import compute_coach_advantages, compute_player_advantages
from crisp.rewards.coach_rewards import compute_coach_reward
from crisp.rewards.player_rewards import apply_persuader_bonus
from crisp.training.batch_builder import (
    build_coach_batch,
    build_player_batch,
    filter_dynamic_sampling,
    filter_no_box,
)
from crisp.training.grpo_loss import compute_grpo_loss
from crisp.training.tensor_utils import pad_sequences
from crisp.types import DiscussionResult, Problem, Rollout


def _get_device(strategy: Any) -> torch.device:
    """Get the device a strategy's model lives on."""
    if hasattr(strategy, '_engine') and strategy._engine is not None:
        return next(strategy._engine.module.parameters()).device
    return torch.device('cpu')


def _gpu_alloc_gb(device: torch.device) -> float:
    """Return GPU memory allocated in GB, or 0.0 if device is CPU."""
    if device.type == 'cuda':
        return torch.cuda.memory_allocated(device) / 1e9
    return 0.0


TRAIN_CHUNK_SIZE = 4  # sequences per micro-batch during training
MAX_GPU_SEQUENCES = 64  # max sequences loaded to GPU at once (H100 80GB limit)


def _chunked_ref_forward(model: Any, input_ids: torch.Tensor,
                         attention_mask: torch.Tensor) -> torch.Tensor:
    """Ref model forward in chunks (no grad). Returns detached log-probs.

    If the ref model is parked on CPU (to save GPU memory), it is
    temporarily moved to GPU for the forward pass and then moved back.
    """
    B = input_ids.shape[0]
    device = input_ids.device

    import logging as _logging
    _rl = _logging.getLogger("crisp.workflow.train_step")

    # Move ref to GPU if parked on CPU (stage=0, no optimizer — safe to move)
    engine = getattr(model, '_engine', model)
    module = getattr(engine, 'module', engine)
    was_cpu = not next(module.parameters()).is_cuda
    if was_cpu:
        module.to(device)
        _rl.info("ref_forward: ref model moved to GPU: %.1fGB", _gpu_alloc_gb(device))

    try:
        if B <= TRAIN_CHUNK_SIZE:
            with torch.no_grad():
                return model.forward(input_ids, attention_mask=attention_mask).detach()

        chunks = []
        with torch.no_grad():
            for i in range(0, B, TRAIN_CHUNK_SIZE):
                end = min(i + TRAIN_CHUNK_SIZE, B)
                chunk_lp = model.forward(
                    input_ids[i:end], attention_mask=attention_mask[i:end],
                )
                chunks.append(chunk_lp.detach())
        _rl.info("ref_forward: after all chunks (before move back): %.1fGB", _gpu_alloc_gb(device))
        return torch.cat(chunks, dim=0)
    finally:
        if was_cpu:
            module.to('cpu')
            torch.cuda.empty_cache()
            _rl.info("ref_forward: after move back + empty_cache: %.1fGB", _gpu_alloc_gb(device))


def _chunked_grpo_backward(
    ds_model: Any, ref_log_probs: torch.Tensor,
    input_ids: torch.Tensor, attention_mask: torch.Tensor,
    old_lp: torch.Tensor, advantages: torch.Tensor, mask: torch.Tensor,
    dcpo_alpha: float, clip_low: float, clip_high: float, js_beta: float,
    grad_scale: float = 1.0,
) -> float:
    """Forward + loss + backward per chunk. No optimizer step.

    grad_scale should be 1/n_splits when using gradient accumulation
    across multiple sub-batches, so total gradient magnitude is correct.
    """
    import logging as _logging
    _bl = _logging.getLogger("crisp.workflow.train_step")
    B = input_ids.shape[0]
    n_chunks = max(1, (B + TRAIN_CHUNK_SIZE - 1) // TRAIN_CHUNK_SIZE)
    total_loss = 0.0
    _dev = input_ids.device

    for i in range(0, B, TRAIN_CHUNK_SIZE):
        end = min(i + TRAIN_CHUNK_SIZE, B)

        if i == 0:
            _bl.info("backward chunk 0: before forward: %.1fGB", _gpu_alloc_gb(_dev))

        chunk_lp = ds_model.forward(
            input_ids[i:end], attention_mask=attention_mask[i:end],
        )

        if i == 0:
            _bl.info("backward chunk 0: after forward: %.1fGB", _gpu_alloc_gb(_dev))

        chunk_loss = compute_grpo_loss(
            chunk_lp, old_lp[i:end], ref_log_probs[i:end],
            advantages[i:end], mask[i:end],
            dcpo_alpha=dcpo_alpha, clip_low=clip_low, clip_high=clip_high,
            js_beta=js_beta,
        )
        # Scale for both within-batch chunking and cross-batch accumulation
        ds_model.backward(chunk_loss * (grad_scale / n_chunks))

        if i == 0:
            _bl.info("backward chunk 0: after backward: %.1fGB", _gpu_alloc_gb(_dev))

        total_loss += chunk_loss.item()

    return total_loss / n_chunks


def _chunked_grpo_train(
    ds_model: Any, ref_log_probs: torch.Tensor,
    input_ids: torch.Tensor, attention_mask: torch.Tensor,
    old_lp: torch.Tensor, advantages: torch.Tensor, mask: torch.Tensor,
    dcpo_alpha: float, clip_low: float, clip_high: float, js_beta: float,
) -> float:
    """Forward + loss + backward + optimizer step. Convenience wrapper."""
    loss = _chunked_grpo_backward(
        ds_model, ref_log_probs, input_ids, attention_mask,
        old_lp, advantages, mask, dcpo_alpha, clip_low, clip_high, js_beta,
    )
    ds_model.optimizer_step()
    return loss


def train_player(
    ctx: Any,
    player_id: int,
    rollouts: List[Rollout],
    discussion_results: List[DiscussionResult],
    problems: List[Problem],
    ds_model: Any,
    ema_tracker: Any,
    sync_weights: bool = True,
) -> float:
    """Execute player training: Steps 7-10.5.

    Trains a single player (Alice or Bob) on their own rollouts.
    Persuader bonus must be applied BEFORE calling this function.

    1. Filter dynamic sampling (zero-variance problems removed)
    2. Compute two-pool advantages
    3. Build training batch
    4. Get reference log-probs
    5. Compute GRPO loss + backward + optimizer step
    6. Sync weights to vLLM (unless sync_weights=False)
    """
    cfg = ctx.config

    player_name = "Alice" if player_id == 0 else "Bob"
    n_total = len(rollouts)

    # Step 7.5: Drop garbled outputs (no \boxed{} answer = incoherent gradients)
    clean = filter_no_box(rollouts)
    n_no_box = n_total - len(clean)

    # Step 8: Dynamic sampling filter
    filtered = filter_dynamic_sampling(clean)
    n_zero_var = len(clean) - len(filtered)

    logger.info(
        "%s filter: %d total → %d no_box_dropped → %d zero_var_dropped → %d training",
        player_name, n_total, n_no_box, n_zero_var, len(filtered),
    )

    # Split rewards into pre/post discussion pools
    pre_rewards = [r.reward for r in filtered]
    post_rewards = [dr.reward for dr in discussion_results]

    # Step 9: Advantages
    pre_advantages, post_advantages = compute_player_advantages(
        pre_rewards, post_rewards, ema_tracker,
        epsilon=cfg.advantage.epsilon,
    )

    # Step 9.5-10: Build batch, get ref log-probs, compute loss
    batch = build_player_batch(
        filtered, pre_advantages,
        discussion_results=discussion_results if discussion_results else None,
        post_advantages=post_advantages if post_advantages else None,
    )

    if not batch.sequences:
        return 0.0

    # Sort by sequence length so each sub-batch has similar lengths,
    # minimizing padding waste and peak GPU memory.
    indices = sorted(range(len(batch.sequences)),
                     key=lambda i: len(batch.sequences[i].tokens))
    sorted_seqs = [batch.sequences[i] for i in indices]
    sorted_advs = [batch.advantages[i] for i in indices]

    B = len(sorted_seqs)
    n_splits = max(1, (B + MAX_GPU_SEQUENCES - 1) // MAX_GPU_SEQUENCES)
    device = _get_device(ds_model)
    total_loss = 0.0

    import logging as _logging
    _mem_log = _logging.getLogger("crisp.workflow.train_step")

    for split_idx in range(n_splits):
        start = split_idx * MAX_GPU_SEQUENCES
        end = min(start + MAX_GPU_SEQUENCES, B)
        sub_seqs = sorted_seqs[start:end]
        sub_advs = sorted_advs[start:end]

        input_ids, attention_mask, old_lp = pad_sequences(
            sub_seqs, pad_token_id=ctx.pad_token_id,
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        old_lp = old_lp.to(device)

        _alloc = _gpu_alloc_gb(device)
        _mem_log.info(
            "split %d/%d: %d seqs, max_len=%d, after pad_to_gpu: %.1fGB",
            split_idx + 1, n_splits, end - start,
            input_ids.shape[1], _alloc,
        )

        ref_log_probs = _chunked_ref_forward(ctx.ref_model, input_ids, attention_mask)
        torch.cuda.empty_cache()

        _alloc = _gpu_alloc_gb(device)
        _mem_log.info("split %d/%d: after ref_forward+empty_cache: %.1fGB", split_idx + 1, n_splits, _alloc)

        old_lp = old_lp[:, 1:]
        mask = attention_mask[:, 1:].float()
        advantages_t = torch.tensor(sub_advs, device=device)

        split_loss = _chunked_grpo_backward(
            ds_model, ref_log_probs,
            input_ids, attention_mask, old_lp, advantages_t, mask,
            dcpo_alpha=cfg.grpo.dcpo_alpha,
            clip_low=cfg.grpo.clip_low,
            clip_high=cfg.grpo.clip_high,
            js_beta=cfg.grpo.js_beta,
            grad_scale=1.0 / n_splits,
        )
        total_loss += split_loss

        # Free GPU memory before next split
        del input_ids, attention_mask, old_lp, ref_log_probs, mask, advantages_t
        torch.cuda.empty_cache()

    ds_model.optimizer_step()
    player_loss = total_loss / n_splits

    # Step 10.5: Sync weights to vLLM
    if sync_weights:
        ds_model.sync_weights(ctx.player_vllm)

    return player_loss


def train_coach(
    ctx: Any,
    problems: List[Problem],
    rollouts: Dict[int, List[Rollout]],
    discussion_results: Dict[int, List[DiscussionResult]],
    sync_weights: bool = True,
) -> Tuple[float, List[float]]:
    """Execute coach training: Steps 11-12.5.

    1. Compute coach reward per problem
    2. Compute coach advantages (EMA-smoothed)
    3. Build coach batch
    4. Compute GRPO loss with js_beta=0
    5. Backward + optimizer step
    6. Sync weights to vLLM (unless sync_weights=False)
    """
    cfg = ctx.config

    # Collect all rollouts per problem for p_hat computation
    rollouts_by_problem: Dict[int, List[Rollout]] = {}
    for pid in rollouts:
        for r in rollouts[pid]:
            rollouts_by_problem.setdefault(r.problem_idx, []).append(r)

    # Check which problems had discussion
    discussed_problems = set()
    resolved_correctly = {}
    for pid in discussion_results:
        for dr in discussion_results[pid]:
            discussed_problems.add(dr.problem_idx)
            if dr.correct:
                resolved_correctly[dr.problem_idx] = True

    # Step 11: Coach rewards
    # During warmup, suppress too_easy penalty — coach is instructed to
    # generate easy problems, so penalizing 100% accuracy is contradictory
    in_warmup = ctx.iteration < cfg.coach.warmup_iters
    effective_too_easy = 0.0 if in_warmup else cfg.coach.too_easy_penalty

    all_embeddings = [p.coach_embedding for p in problems]
    coach_rewards = []
    for i, problem in enumerate(problems):
        player_rolls = rollouts_by_problem.get(i, [])
        disc_occurred = i in discussed_problems
        disc_resolved = resolved_correctly.get(i, False)

        reward = compute_coach_reward(
            problem, i, all_embeddings, player_rolls,
            disc_occurred, disc_resolved, ctx.rep_buffer,
            alpha=cfg.coach.discussion_alpha,
            lambda_rep=cfg.coach.repetition_lambda,
            tau_sim=cfg.coach.repetition_tau,
            too_hard_penalty=cfg.coach.too_hard_penalty,
            too_easy_penalty=effective_too_easy,
            too_easy_threshold=cfg.coach.too_easy_threshold,
            unsolvable_penalty=cfg.coach.unsolvable_penalty,
        )
        coach_rewards.append(reward)

    # Skip coach training when all rewards are identical — no learning signal.
    # Training on zero-variance advantages only adds JS-div noise that causes drift.
    if len(set(coach_rewards)) <= 1:
        logger.info("Coach rewards all identical (%.2f), skipping training step",
                     coach_rewards[0] if coach_rewards else 0.0)
        return 0.0, coach_rewards

    # Step 11 cont: Coach advantages
    coach_advantages = compute_coach_advantages(
        coach_rewards, ctx.coach_ema,
        epsilon=cfg.advantage.epsilon,
    )

    # Step 12: Build batch and compute loss
    batch = build_coach_batch(problems, coach_advantages)

    if not batch.sequences:
        return 0.0, coach_rewards

    input_ids, attention_mask, old_lp = pad_sequences(
        batch.sequences, pad_token_id=ctx.pad_token_id,
    )

    # Move tensors to model device
    device = _get_device(ctx.ds_coach)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    old_lp = old_lp.to(device)

    # Coach self-referencing: use generation-time log probs as reference.
    # The player ref model (4B) is wrong for the 14B coach — DCPO bounds
    # and JSD become meaningless. Using old_lp gives correct KL anchor.
    old_lp = old_lp[:, 1:]  # [B, T] -> [B, T-1] to match model output shape
    ref_log_probs = old_lp   # Coach's own log probs as reference

    mask = attention_mask[:, 1:].float()
    advantages_t = torch.tensor(batch.advantages, device=device)

    # Step 12: Chunked forward + loss + backward + optimizer step
    coach_loss = _chunked_grpo_train(
        ctx.ds_coach, ref_log_probs,
        input_ids, attention_mask, old_lp, advantages_t, mask,
        dcpo_alpha=cfg.grpo.dcpo_alpha,
        clip_low=cfg.grpo.clip_low,
        clip_high=cfg.grpo.clip_high,
        js_beta=cfg.grpo.coach_js_beta,  # stronger anchoring for self-referencing coach
    )

    # Step 12.5: Sync coach weights (skip if no vLLM engine for coach)
    if sync_weights and ctx.coach_vllm is not None:
        ctx.ds_coach.sync_weights(ctx.coach_vllm)

    return coach_loss, coach_rewards
