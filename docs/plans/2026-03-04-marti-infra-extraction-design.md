# MARTI Infra Extraction Design

**Date**: 2026-03-04
**Status**: Approved
**Phase**: 9

## Overview

Extract MARTI's infrastructure layer (Ray/vLLM/DeepSpeed) into `crisp/infra/` by vendoring files with PPO-specific code stripped. CRISP's domain logic (rewards, discussion, verification) remains untouched. The infra layer provides: distributed actor management, vLLM inference engines, DeepSpeed training strategy, and weight synchronization between training and inference.

**Source**: [MARTI](https://github.com/TsinghuaC3I/MARTI) (OpenRLHF-derived)
**Approach**: Full vendor + adapt (copy files, rename imports, strip PPO code)
**Hardware targets**: Local single-GPU (4080 Super 16GB) AND Modal multi-GPU

## Module Structure

```
crisp/infra/
├── __init__.py
├── ray_launcher.py        # BaseDistributedActor, BaseModelActor, RayActorGroup
│                          # Placement group creation, rank/GPU management
│                          # async_run_method_batch() for work distribution
│
├── vllm_engine.py         # LLMRayActor wrapper around vllm.LLM
│                          # create_vllm_engines() factory with placement groups
│                          # Synchronous add_requests/get_responses API
│                          # Structured to allow async extension later
│
├── vllm_worker_wrap.py    # WorkerWrap extension injected into vLLM workers
│                          # init_process_group() for DS↔vLLM coordination
│                          # update_weight() via NCCL broadcast
│                          # update_weight_cuda_ipc() for same-node transfer
│
├── deepspeed_strategy.py  # DeepspeedStrategy class
│                          # ZeRO-2/3 config builders (train + eval)
│                          # FusedAdam / CPUAdam optimizer creation
│                          # Sleep mode: offload/reload optimizer states
│                          # Gradient checkpointing setup
│
├── weight_sync.py         # broadcast_weights_to_vllm() — extracted from PPO actor
│                          # Handles ZeRO-3 GatheredParameters context manager
│                          # NCCL broadcast to vLLM workers
│                          # CUDA IPC path for colocated models
│                          # Used by both player and coach training steps
│
├── actor_model.py         # Actor class: HF CausalLM wrapper
│                          # LoRA/PEFT support (load, merge, save)
│                          # forward() → log-probs over token sequences
│                          # Gradient checkpointing toggle
│
├── experience.py          # generate_samples(): vLLM output → CRISP Rollout objects
│                          # This is the ONLY place MARTI output types map to CRISP types
│                          # No MARTI Experience dataclass exposed — all consumers
│                          # work with Rollout, DiscussionResult, TrainingBatch
│
└── utils.py               # init_process_group(), stateless variant
                           # GPU ID detection, bundle index helpers
                           # Environment variable management
```

### What's stripped from MARTI

- CriticModelActor, CriticPPOTrainer (GRPO is critic-free)
- RewardModelActor, RewardModel (CRISP uses rule-based rewards)
- PolicyLoss / PPO clip loss (replaced by crisp/training/grpo_loss.py)
- KL controller (CRISP uses fixed JS-divergence via js_beta)
- GAE computation (GRPO uses group normalization, already in crisp/rewards/advantages.py)
- NaiveReplayBuffer (not needed for GRPO)

### What's kept from MARTI

- LoRA/PEFT support in actor_model.py
- Sleep mode (offload/reload optimizer states) in deepspeed_strategy.py
- CUDA IPC weight transfer path in weight_sync.py
- Both NCCL and Ray Collective backends for process groups

## Data Flow

The infra layer integrates with CRISP's existing main loop (Steps 1-13):

```
grpo_trainer.__init__():
    player_vllm = create_vllm_engines(player_model, ...)   # created ONCE
    coach_vllm  = create_vllm_engines(coach_model, ...)     # created ONCE
    ref_model   = BaseModelActor(ref_weights, eval_config)  # frozen, NEVER updated
    # WARNING: ref_model must NOT be included in any optimizer group.
    # Accidental updates corrupt JS-divergence signal silently.
    ds_player   = DeepspeedStrategy(train_config)
    ds_coach    = DeepspeedStrategy(train_config)

grpo_trainer.step():
    # --- Generation phase (vLLM active, DS offloaded) ---
    ds_strategy.offload_deepspeed_states()

    # Step 1: Coach generates problems
    problems: List[Problem] = generate_samples(coach_vllm, prompts)

    # Step 2: Players generate rollouts
    rollouts: List[Rollout] = generate_samples(player_vllm, problem_prompts)

    # --- Domain logic (CPU, existing CRISP modules) ---
    # Steps 3-9: verify, rewards, discussion, advantages
    # No infra dependency — pure functions, already tested

    # --- Training phase (DS active, vLLM idle) ---
    ds_strategy.reload_deepspeed_states()

    # Step 9.5: Reference log-probs
    ref_log_probs = ref_model.forward(batch.sequences)

    # Step 10: Player GRPO update
    loss = compute_grpo_loss(current_lp, old_lp, ref_lp, advantages, ...)
    ds_player.backward(loss)
    ds_player.optimizer_step()

    # Step 10.5: Sync player weights to vLLM
    broadcast_weights_to_vllm(player_model, player_vllm)

    # Steps 11-12: Coach rewards + GRPO update (same pattern)
    # Step 12.5: Sync coach weights
    broadcast_weights_to_vllm(coach_model, coach_vllm)
```

### Boundary principle

The infra layer never imports from `crisp/rewards/`, `crisp/discussion/`, `crisp/verifier/`, or `crisp/training/`. It only knows about: models, tokens, log-probs, gradients, and weight sync. All domain logic stays in existing modules.

### Local single-GPU mode

Same code path with `num_gpus=1`, `tensor_parallel_size=1`, ZeRO-stage-2. Ray runs in local mode (no cluster). Sleep mode is critical: shares the single GPU between vLLM inference and DeepSpeed training by offloading optimizer states during generation and reloading before training.

## Testing Strategy

### Tier 1 (no GPU, mocked — CI)

| Test file | What it covers |
|-----------|---------------|
| `test_weight_sync.py` | Mock model + mock vLLM engines. Verify `broadcast_weights_to_vllm()` calls `update_weight()` for each named parameter. Verify ZeRO-3 path uses `GatheredParameters` context manager. |
| `test_experience.py` | Mock vLLM output → verify `generate_samples()` returns correct `Rollout` objects with proper field mapping. No MARTI types leak out. |
| `test_deepspeed_config.py` | Verify config builders produce valid ZeRO-2/3 JSON for train and eval modes. Verify sleep mode methods exist and are callable. |

### Tier 3 (real GPU, smoke tests)

| Test file | What it covers |
|-----------|---------------|
| `test_single_gpu_loop.py` | 1 iteration, batch_size=2, rollouts=2 on Qwen3-0.6B. Verify: gradients flow, no NaN/Inf, weight sync updates vLLM, generation produces valid tokens after sync. |
| `test_sleep_mode.py` | Verify GPU memory drops after offload, restores after reload. |
| `test_weight_sync_correctness.py` | After `broadcast_weights_to_vllm()`, generate same prompt through vLLM engine and `actor_model.forward()`. Compare log-probs — must match within bf16 tolerance (~1e-3). Catches silent stale/partial weight sync bugs. |

### Not tested in isolation (covered by Tier 3 end-to-end)

- Ray actor placement groups (requires multi-GPU)
- CUDA IPC weight transfer (requires colocated GPUs)
- DeepSpeed training convergence (that's the full experiment)

## Key Design Decisions

1. **experience.py maps to CRISP types** — No MARTI `Experience` dataclass exposed. `generate_samples()` returns `Rollout` directly. Type boundary exists in one place only.
2. **weight_sync.py is standalone** — Extracted from MARTI's PPO actor. Clean API: `broadcast_weights_to_vllm(model, engines)`. Preserves ZeRO-3 `GatheredParameters` context manager.
3. **Engines created once in __init__** — `create_vllm_engines()` and `ref_model` instantiated once, reused across all `step()` calls. Only weights change (via broadcast).
4. **ref_model is frozen** — Never included in optimizer groups. Explicit comment in code to prevent accidental updates that would silently corrupt JS-divergence.
5. **LoRA support preserved** — Kept in actor_model.py for future experimentation, even though initial runs use full fine-tuning.
6. **Sync-first, async-ready** — vLLM engine uses synchronous API. Code structured so async extension can be added later without refactoring.
7. **Sleep mode for single-GPU** — Offload/reload optimizer states to share GPU between vLLM inference and DeepSpeed training phases.
