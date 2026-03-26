"""Steps 1-13 orchestration: the CRISP training main loop."""
from __future__ import annotations

import logging
from typing import Any, List, Optional

import ray

from crisp.rewards.player_rewards import apply_persuader_bonus
from crisp.workflow import coach_step, discussion_step, rollout_step, train_step
from crisp.workflow.collector import IterationData, StepCollector
from crisp.workflow.context import StepResult

logger = logging.getLogger(__name__)


def _log_gpu_memory(label: str) -> None:
    """Log current GPU memory usage from this process's perspective."""
    import torch
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info("GPU mem [%s]: allocated=%.1fGB reserved=%.1fGB", label, alloc, reserved)


def _sleep_vllm(engines: List[Any]) -> None:
    """Put vLLM engines to sleep to free GPU memory (KV cache)."""
    _log_gpu_memory("before sleep")
    refs = [e.sleep.remote() for e in engines]
    ray.get(refs)
    _log_gpu_memory("after sleep")
    logger.info("vLLM engines asleep (KV cache freed)")


def _wake_vllm(engines: List[Any]) -> None:
    """Wake vLLM engines (reallocate KV cache + reload weights)."""
    refs = [e.wake_up.remote() for e in engines]
    ray.get(refs)
    _log_gpu_memory("after wake")
    logger.info("vLLM engines awake")


def step(ctx: Any, collector: Optional[StepCollector] = None) -> StepResult:
    """Execute one full training iteration (Steps 1-13).

    Two-player orchestration: Alice (player 0) and Bob (player 1) are trained
    independently with separate DeepSpeed strategies and EMA trackers.

    Sequential rollout generation: sync Alice weights -> roll Alice ->
    sync Bob weights -> roll Bob. This ensures each player's vLLM engine
    reflects its latest trained weights before generating rollouts.

    Coach training gated by ctx.config.coach.update_freq.

    Args:
        ctx: WorkflowContext with all infra handles.
        collector: Optional StepCollector to capture intermediate data.
    """
    # === GENERATION PHASE ===
    # Determine if we're using shared-GPU sleep/wake (coach vLLM on GPU 0)
    # Only active when vllm_enable_sleep is True (2-GPU time-sharing mode).
    # With dedicated GPUs (4-GPU mode), engines run independently — no sleep/wake.
    _shared_gpu = (ctx.config.infra.vllm_enable_sleep
                   and ctx.coach_vllm is not None
                   and ctx.coach_vllm is not ctx.player_vllm)

    # Step 1: Coach generates problems (with recent accuracy context)
    # Wake coach vLLM if using shared GPU (player already sleeping from prev iter)
    if _shared_gpu:
        _wake_vllm(ctx.coach_vllm)

    problems = coach_step.generate_problems(
        ctx, accuracy_history=ctx.accuracy_history,
    )

    # Sleep coach, wake player for rollouts
    if _shared_gpu:
        _sleep_vllm(ctx.coach_vllm)
        _wake_vllm(ctx.player_vllm)

    if not problems:
        logger.warning("iter=%d: Coach produced 0 valid problems, skipping iteration",
                       ctx.iteration)
        # Sleep player before returning — otherwise next iteration's coach
        # wake will OOM because player KV cache still holds GPU 0 memory.
        if _shared_gpu:
            _sleep_vllm(ctx.player_vllm)
        ctx.iteration += 1
        return StepResult(
            alice_loss=0.0, bob_loss=0.0, coach_loss=None, num_problems=0,
            num_discussions=0, player_accuracy=0.0, coach_iteration=False,
        )

    # Separate solvable problems (for players) from all problems (for coach training).
    # Unsolvable problems still get coach training signal (negative reward).
    solvable_problems = [p for p in problems if p.self_solvable]
    n_unsolvable = len(problems) - len(solvable_problems)
    if n_unsolvable:
        logger.info("iter=%d: %d/%d problems unsolvable by coach (will penalize)",
                     ctx.iteration, n_unsolvable, len(problems))

    for i, p in enumerate(solvable_problems):
        logger.info("iter=%d problem[%d] (answer=%s): %s", ctx.iteration, i,
                     p.ground_truth, p.text[:200].replace("\n", " "))

    # Step 2-4: Sequential per-player rollouts with weight sync
    # Sync Alice -> roll Alice -> sync Bob -> roll Bob
    # This ensures each player's vLLM engine reflects its latest weights.
    if solvable_problems:
        # Sync Alice weights to vLLM, then generate Alice rollouts
        ctx.ds_alice.sync_weights(ctx.player_vllm)
        alice_rollouts = rollout_step.generate_rollouts(
            ctx, solvable_problems, player_id=0,
        )

        # Sync Bob weights to vLLM, then generate Bob rollouts
        ctx.ds_bob.sync_weights(ctx.player_vllm)
        bob_rollouts = rollout_step.generate_rollouts(
            ctx, solvable_problems, player_id=1,
        )

        rollouts = {0: alice_rollouts, 1: bob_rollouts}

        for pid in sorted(rollouts):
            for r in rollouts[pid]:
                player_name = "Alice" if pid == 0 else "Bob"
                logger.info(
                    "iter=%d %s p%d: correct=%s answer=%s reward=%.2f last200=%.200s",
                    ctx.iteration, player_name, r.problem_idx, r.correct,
                    r.answer, r.reward, r.text[-200:].replace("\n", " "),
                )
    else:
        rollouts = {0: [], 1: []}

    # Steps 5-6: Discussion trigger + execution
    discussion_results, majority_answers = discussion_step.run_discussion(
        ctx, rollouts, solvable_problems,
    )

    # Compute metrics before training
    all_rollouts = []
    for pid in rollouts:
        all_rollouts.extend(rollouts[pid])

    num_correct = sum(1 for r in all_rollouts if r.correct)
    player_accuracy = num_correct / len(all_rollouts) if all_rollouts else 0.0

    num_discussions = 0
    for pid in discussion_results:
        num_discussions += len(discussion_results[pid])
    num_discussions = num_discussions // 2  # 2 results per discussed problem

    # === TRAINING PHASE ===
    # Free residual GPU memory before training
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    # Sleep player vLLM for training (shared-GPU: coach already sleeping)
    if _shared_gpu:
        _sleep_vllm(ctx.player_vllm)
    elif ctx.config.infra.vllm_enable_sleep:
        all_engines = list(ctx.player_vllm)
        if ctx.coach_vllm is not None and ctx.coach_vllm is not ctx.player_vllm:
            all_engines.extend(ctx.coach_vllm)
        _sleep_vllm(all_engines)

    _log_gpu_memory("before player training")

    # Apply persuader bonus ONCE before per-player training split
    apply_persuader_bonus(
        rollouts, discussion_results, majority_answers, solvable_problems,
        gamma=ctx.config.player.persuader_bonus,
    )

    # Steps 7-10.5: Per-player training (defer weight sync — engines sleeping)
    # Train Alice (player 0)
    alice_loss = train_step.train_player(
        ctx, player_id=0,
        rollouts=rollouts[0],
        discussion_results=discussion_results[0],
        problems=solvable_problems,
        ds_model=ctx.ds_alice,
        ema_tracker=ctx.alice_ema,
        sync_weights=False,
    )

    # Free cached GPU memory between player trainings
    torch.cuda.empty_cache()

    # Train Bob (player 1)
    bob_loss = train_step.train_player(
        ctx, player_id=1,
        rollouts=rollouts[1],
        discussion_results=discussion_results[1],
        problems=solvable_problems,
        ds_model=ctx.ds_bob,
        ema_tracker=ctx.bob_ema,
        sync_weights=False,
    )

    # Free cached GPU memory between player and coach training
    torch.cuda.empty_cache()

    # Steps 11-12.5: Coach training (gated by update_freq)
    coach_loss = None
    coach_rewards = None
    is_coach_iter = ctx.iteration % ctx.config.coach.update_freq == 0
    if is_coach_iter:
        coach_loss, coach_rewards = train_step.train_coach(
            ctx, problems, rollouts, discussion_results,
            sync_weights=False,
        )

    # Post-training: sync weights to vLLM engines
    # Free training buffers first so vLLM can reclaim memory
    torch.cuda.empty_cache()

    if _shared_gpu:
        # Shared-GPU: wake each engine, sync, sleep. They stay sleeping
        # until next iteration's generation phase wakes them.
        # Player weight sync deferred to next iteration's rollout phase
        # (sequential sync Alice -> roll Alice -> sync Bob -> roll Bob)
        if is_coach_iter:
            _wake_vllm(ctx.coach_vllm)
            ctx.ds_coach.sync_weights(ctx.coach_vllm)
            _sleep_vllm(ctx.coach_vllm)
    elif ctx.config.infra.vllm_enable_sleep:
        _log_gpu_memory("after training")
        torch.cuda.empty_cache()
        _log_gpu_memory("after empty_cache")
        _wake_vllm(all_engines)
        # Player weight sync deferred to next iteration's rollout phase
        if ctx.coach_vllm is not None:
            ctx.ds_coach.sync_weights(ctx.coach_vllm)
    else:
        # No sleep mode (dedicated GPUs) — sync coach weights directly.
        # Player weight sync deferred to next iteration's rollout phase
        # (sequential sync Alice -> roll Alice -> sync Bob -> roll Bob).
        if is_coach_iter and ctx.coach_vllm is not None:
            ctx.ds_coach.sync_weights(ctx.coach_vllm)

    # INVARIANT: push AFTER train_coach() so current batch's embeddings
    # are NOT included in their own cross-batch repetition penalty.
    ctx.rep_buffer.push([p.coach_embedding for p in problems])

    # Track accuracy for coach calibration prompt
    ctx.accuracy_history.append(player_accuracy)

    # Step 13: Increment iteration
    ctx.iteration += 1

    # Backward-compat: average loss for collector
    player_loss = (alice_loss + bob_loss) / 2

    result = StepResult(
        alice_loss=alice_loss,
        bob_loss=bob_loss,
        coach_loss=coach_loss,
        num_problems=len(solvable_problems),
        num_discussions=num_discussions,
        player_accuracy=player_accuracy,
        coach_iteration=is_coach_iter,
    )

    if collector is not None:
        collector.record(IterationData(
            iteration=ctx.iteration - 1,
            problems=problems,
            rollouts=rollouts,
            majority_answers=majority_answers,
            discussion_results=discussion_results,
            player_loss=player_loss,
            coach_loss=coach_loss,
            coach_rewards=coach_rewards,
            result=result,
        ))

    return result
