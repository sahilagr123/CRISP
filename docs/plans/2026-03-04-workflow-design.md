# Workflow Orchestration Layer Design

**Date**: 2026-03-04
**Status**: Approved
**Phase**: 10

## Overview

Build `crisp/workflow/` — the orchestration layer that wires `crisp/infra/` (Ray/vLLM/DeepSpeed) to domain logic (rewards, discussion, verification, training). Implements the Steps 1-13 main loop from the CRISP design doc.

**Approach**: WorkflowContext dataclass for dependency injection. Each step module is a thin orchestrator calling existing domain functions. main_loop.step() is a linear sequence.

## Module Structure

```
crisp/workflow/
├── __init__.py
├── context.py          # WorkflowContext dataclass + StepResult
├── coach_step.py       # Step 1: Problem generation + parsing
├── rollout_step.py     # Step 2: Player rollout generation + verification + rewards
├── discussion_step.py  # Steps 5-6: Trigger + discussion execution
├── train_step.py       # Steps 9.5-12.5: Training + weight sync
└── main_loop.py        # step() orchestrating Steps 1-13
```

Plus modification to existing code:
- `crisp/training/batch_builder.py` — implement `build_player_batch()` + add `build_coach_batch()`
- `crisp/config.py` — add `update_freq: int = 5` to `CoachConfig`

## context.py — WorkflowContext

```python
@dataclass
class WorkflowContext:
    player_vllm: List[Any]       # vLLM engine actors
    coach_vllm: List[Any]        # vLLM engine actors
    ref_model: Any               # Frozen reference policy (NEVER updated)
    ds_player: Any               # DeepSpeed strategy for player
    ds_coach: Any                # DeepSpeed strategy for coach
    config: CRISPConfig
    # Stateful — persist across iterations
    player_ema: EMATracker       # Pool 2 advantages for players
    coach_ema: EMATracker        # Coach advantages
    rep_buffer: RepetitionBuffer # Cross-batch repetition penalty
    iteration: int = 0

@dataclass
class StepResult:
    player_loss: float
    coach_loss: Optional[float]  # None on non-coach-update iterations
    num_problems: int
    num_discussions: int
    player_accuracy: float       # fraction correct pre-discussion
    coach_iteration: bool        # whether coach was updated
```

## Step Functions

### coach_step.generate_problems(ctx, prompts) → List[Problem]

Step 1. Calls `generate_samples(ctx.coach_vllm, ...)` to get coach rollouts. Parses each output using `extract_boxed()` to get ground_truth. Problem text is everything before the boxed answer. Computes embeddings via sentence-transformers for repetition penalty.

### rollout_step.generate_rollouts(ctx, problems) → Dict[int, List[Rollout]]

Step 2-4. For each problem, generates `rollouts_per_problem` rollouts per player via `generate_samples(ctx.player_vllm, ...)`. Then for each rollout:
- `extract_boxed(rollout.text)` → answer
- `check(answer, ground_truth)` → correct
- `compute_solve_reward(rollout)` → base reward
- `compute_overlong_penalty(len(rollout.tokens), ...)` → subtract from reward

Returns dict keyed by problem_idx.

### discussion_step.run_discussion(ctx, rollouts, problems) → (Dict[int, List[DiscussionResult]], Dict[Tuple[int,int], str])

Steps 5-6.

For each problem:
1. `majority_vote(alice_rollouts)` and `majority_vote(bob_rollouts)`
2. `should_discuss(majority_a, majority_b)` — if False, skip
3. `select_representatives(rollouts, majority_answers, ground_truth, problem_idx)`
4. Format discussion prompts using template from config: each player sees own solution + other's solution
5. `generate_samples(ctx.player_vllm, discussion_prompts)` — 1 rollout per player per problem
6. `parse_discussion_response(text)` → evaluation_text, final_answer
7. `check(final_answer, ground_truth)` → correct

Returns discussion results and majority_answers (needed by persuader bonus).

### Discussion prompt template

Stored as `discussion_template: str` on `CoachConfig` (or a new field). Uses placeholders:

```
You previously solved this problem:
{problem}

Your solution: {own_solution}
Another student's solution: {other_solution}

EVALUATION:
[Analyze both solutions]

FINAL ANSWER: \boxed{...}
```

### train_step.train_player(ctx, rollouts, discussion_results, problems) → float

Steps 7-10.5.

1. `apply_persuader_bonus(rollouts, discussion_results, majority_answers, problems, gamma)` — in-place, once
2. `filter_dynamic_sampling(all_rollouts)` — remove zero-variance problems
3. `compute_player_advantages(pre_rewards, post_rewards, ctx.player_ema)` — two-pool normalization
4. `build_player_batch(rollouts, advantages, ...)` → TrainingBatch
5. `ref_model.forward(batch.sequences)` → ref_log_probs
6. `compute_grpo_loss(current_lp, old_lp, ref_lp, advantages, mask)` → loss
7. `ds_player.backward(loss)` + `ds_player.optimizer_step()`
8. `broadcast_weights_to_vllm(player_model, ctx.player_vllm)`

### train_step.train_coach(ctx, problems, rollouts, discussion_info) → float

Steps 11-12.5.

1. For each problem: `compute_coach_reward(problem, ...)` using rep_buffer.compute_penalty() (called BEFORE push)
2. `compute_coach_advantages(rewards, ctx.coach_ema)`
3. `build_coach_batch(problems, advantages, ...)` → TrainingBatch
4. `compute_grpo_loss(..., js_beta=0)` — no JS-divergence for coach
5. `ds_coach.backward(loss)` + `ds_coach.optimizer_step()`
6. `broadcast_weights_to_vllm(coach_model, ctx.coach_vllm)`

## main_loop.step()

```python
def step(ctx: WorkflowContext) -> StepResult:
    # --- Generation phase (vLLM active, DS offloaded) ---
    offload_deepspeed_states(ctx.ds_player)
    offload_deepspeed_states(ctx.ds_coach)

    problems = coach_step.generate_problems(ctx, coach_prompts)
    rollouts = rollout_step.generate_rollouts(ctx, problems)
    discussion_results, majority_answers = discussion_step.run_discussion(
        ctx, rollouts, problems
    )

    # --- Training phase ---
    reload_deepspeed_states(ctx.ds_player)
    # Coach DS states only reloaded on coach update iterations (saves GPU memory)

    player_loss = train_step.train_player(
        ctx, rollouts, discussion_results, majority_answers, problems
    )

    coach_loss = None
    is_coach_iter = ctx.iteration % ctx.config.coach.update_freq == 0
    if is_coach_iter:
        reload_deepspeed_states(ctx.ds_coach)
        coach_loss = train_step.train_coach(
            ctx, problems, rollouts, discussion_results
        )
        offload_deepspeed_states(ctx.ds_coach)

    # INVARIANT: push AFTER train_coach() so current batch's embeddings
    # are NOT included in their own cross-batch repetition penalty.
    # compute_coach_reward() calls rep_buffer.compute_penalty() inside
    # train_coach(), which runs before this push.
    ctx.rep_buffer.push([p.coach_embedding for p in problems])
    ctx.iteration += 1

    return StepResult(...)
```

### Coach update frequency

Coach trains every `ctx.config.coach.update_freq` iterations (default 5). On non-coach iterations:
- Coach DeepSpeed states stay offloaded (never reloaded)
- No coach backward/optimizer step
- Coach vLLM weights unchanged
- `coach_loss` is None in StepResult

## batch_builder changes

### build_player_batch(rollouts, advantages, ref_log_probs) → TrainingBatch

Pads token sequences to max length in batch. Creates attention mask. Stacks into tensors. Handles both pre-discussion rollouts and post-discussion results (distinguished by `is_post_discussion` flag on TrainingBatch).

### build_coach_batch(problems, advantages, ref_log_probs) → TrainingBatch

Same pattern for coach sequences. Coach sequences come from Problem.coach_sequence.

## Boundary Principles

1. **workflow/ never computes rewards directly** — it calls functions from crisp/rewards/
2. **workflow/ never parses answers directly** — it calls extract_boxed() and check()
3. **workflow/ never builds loss** — it calls compute_grpo_loss()
4. **workflow/ never manages GPU memory directly** — it calls offload/reload/broadcast via crisp/infra/
5. **infra/ never imports from workflow/** — dependency flows one way

## Testing Strategy

Mock infra (vLLM engines, DeepSpeed, ref_model), use real domain logic.

| Test file | What it covers |
|-----------|---------------|
| `test_context.py` | WorkflowContext creation, StepResult fields |
| `test_coach_step.py` | Problem generation: mock vLLM → verify Problem objects with correct ground_truth and embeddings |
| `test_rollout_step.py` | Rollout generation: mock vLLM → verify Rollout objects with answers, correctness, rewards |
| `test_discussion_step.py` | Discussion trigger + execution: mock vLLM → verify DiscussionResult with parsed evaluation/answer |
| `test_train_step.py` | Training orchestration: mock DeepSpeed/ref_model → verify loss computation, weight sync calls |
| `test_main_loop.py` | Full step(): mock infra → verify step sequence, coach frequency gating, rep_buffer push ordering |
| `test_batch_builder.py` | Padding, masking, tensor shapes for build_player_batch and build_coach_batch |

## Config Changes

- `CoachConfig.update_freq: int = 5` — coach trains every N player iterations
- `CoachConfig.discussion_template: str` — prompt template with {problem}, {own_solution}, {other_solution} placeholders
- `CoachConfig.coach_prompt_template: str` — prompt for problem generation
