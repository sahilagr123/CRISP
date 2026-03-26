# Smoke Test Design Document

**Date**: 2026-03-05
**Status**: Approved

## Overview

An end-to-end smoke test that runs the full CRISP pipeline on Modal GPUs with a small model (Qwen3-0.6B), producing a Markdown report with detailed visibility into every step: coach problem generation, player rollouts, discussion triggers, post-discussion responses, rewards, advantages, and loss values.

**Goal**: Confirm all code paths work with real model inference and real gradient updates. The report provides human-readable evidence that the system functions correctly.

## Architecture: Approach B — Observability hooks in `step()`

### IterationData and StepCollector

New dataclasses in `crisp/workflow/collector.py`:

```python
@dataclass
class IterationData:
    iteration: int
    problems: List[Problem]
    rollouts: Dict[int, List[Rollout]]
    majority_answers: Dict[Tuple[int, int], str]
    discussion_results: Dict[int, List[DiscussionResult]]
    player_loss: float
    coach_loss: Optional[float]
    coach_rewards: Optional[List[float]]
    result: StepResult

class StepCollector:
    iterations: List[IterationData]

    def __init__(self):
        self.iterations = []

    def record(self, data: IterationData) -> None:
        self.iterations.append(data)
```

### Changes to existing code

1. **`main_loop.step(ctx, collector=None)`** — Add optional `collector` parameter. At the end, if collector is not None, assemble `IterationData` from the variables already in scope (problems, rollouts, discussion_results, majority_answers, player_loss, coach_loss, coach_rewards) and call `collector.record()`.

2. **`train_step.train_coach()`** — Change return type from `float` to `Tuple[float, List[float]]` to also return the per-problem `coach_rewards` list. Update the single call site in `main_loop.step()`.

3. **`config.py`** — Change `CoachConfig.update_freq` default from 5 to 1 (coach should update every iteration).

### Smoke test script

**`scripts/smoke_test.py`**:
1. Parses args (config path, output path)
2. Loads config, calls `init_infra()`
3. Runs 10 iterations of `step(ctx, collector=collector)`
4. Calls `write_smoke_report(collector, config, output_path)`

### Smoke test config

**`configs/smoke_test.yaml`**:
```yaml
training:
  model_name: "Qwen/Qwen3-0.6B"
  coach_model_name: "Qwen/Qwen3-0.6B"
  num_iterations: 10
  eval_freq: 0
  save_freq: 0

infra:
  num_gpus_per_node: 1
  vllm_num_engines: 1
  vllm_enable_sleep: true
  max_model_len: 4096

player:
  rollouts_per_problem: 4

coach:
  batch_size: 4
```

### Report generator

**`scripts/write_smoke_report.py`** — Takes a `StepCollector` and writes structured Markdown:

- **Header**: date, model, iteration count, config summary
- **Summary table**: iteration, num_problems, num_discussions, accuracy, player_loss, coach_loss
- **Detailed sections for iterations 0, 1, 2, and 9** (first three + last):
  - Coach-generated problems (text + ground_truth)
  - Player rollouts per problem: table with player, rollout#, answer, correct, reward
  - Discussion triggers: majority votes, agree/disagree
  - Discussion responses: evaluation text, final answer, correct, reward, persuader identification
  - Coach rewards per problem: r_uncertainty, r_discussion, r_repetition, r_total
  - Loss values
- **Trend summary**: accuracy and loss across all 10 iterations

### Modal deployment

**`scripts/modal_smoke.py`** — Wraps the smoke test in a Modal function:
- Defines a Modal image with torch, vllm, ray, transformers, sentence-transformers
- Runs `smoke_test.main()` inside a GPU container
- Copies the report back to local filesystem

## Files to create/modify

| File | Action |
|------|--------|
| `crisp/config.py` | Change `update_freq` default: 5 -> 1 |
| `crisp/workflow/collector.py` | NEW: IterationData, StepCollector |
| `crisp/workflow/main_loop.py` | Add collector param to step() |
| `crisp/workflow/train_step.py` | train_coach returns (loss, coach_rewards) |
| `tests/workflow/test_main_loop.py` | Update for new train_coach return type |
| `configs/smoke_test.yaml` | NEW: smoke test config |
| `scripts/smoke_test.py` | NEW: smoke test entry point |
| `scripts/write_smoke_report.py` | NEW: Markdown report generator |
| `scripts/modal_smoke.py` | NEW: Modal wrapper |

## Testing

- Existing 280 unit tests must still pass after the refactor (collector=None is default, train_coach return type change is backward-compatible at call site)
- The smoke test itself IS the test — success = report generated with non-empty problems, rollouts, discussions, finite losses
