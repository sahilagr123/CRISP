# Smoke Test Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add observability hooks to `step()`, build a smoke test script that runs 10 iterations on Modal with Qwen3-0.6B, and generates a Markdown report showing problems, rollouts, discussions, rewards, and losses.

**Architecture:** Optional `StepCollector` parameter on `step()` captures intermediate data (problems, rollouts, discussions, rewards). A smoke test script calls `init_infra()` + `step()` loop, then a report generator writes structured Markdown. Modal wrapper deploys it on a GPU.

**Tech Stack:** Python dataclasses, Modal (GPU deployment), existing CRISP infra (Ray/vLLM/DeepSpeed)

---

### Task 1: Change CoachConfig.update_freq default to 1

**Files:**
- Modify: `crisp/config.py:27`
- Modify: `tests/workflow/test_main_loop.py:67` (test that hardcodes update_freq=3 is fine, but line 53 uses default — `train_coach.return_value` must match new return type)

**Step 1: Change the default**

In `crisp/config.py`, line 27, change:
```python
    update_freq: int = 5  # Coach trains every N player iterations
```
to:
```python
    update_freq: int = 1  # Coach trains every iteration
```

**Step 2: Run tests to verify nothing breaks**

Run: `pytest tests/test_config.py tests/workflow/test_main_loop.py -v`
Expected: All pass (tests that set `update_freq=3` explicitly still work; default tests now see `update_freq=1`)

**Step 3: Commit**

```bash
git add crisp/config.py
git commit -m "feat: change coach update_freq default from 5 to 1"
```

---

### Task 2: Change train_coach return type to include coach_rewards

**Files:**
- Modify: `crisp/workflow/train_step.py:125-219` (train_coach function)
- Modify: `crisp/workflow/main_loop.py:59` (call site)
- Modify: `tests/workflow/test_train_step.py:104-160` (train_coach tests)
- Modify: `tests/workflow/test_main_loop.py:52-53,80-81,121-122,151-152,186-187,213-214` (mock return values)

**Step 1: Update train_coach to return (loss, coach_rewards)**

In `crisp/workflow/train_step.py`, the `train_coach` function currently has two return points:

Line 184 (empty batch early return):
```python
    if not batch.sequences:
        return 0.0
```
Change to:
```python
    if not batch.sequences:
        return 0.0, coach_rewards
```

Line 219 (normal return):
```python
    return loss.item()
```
Change to:
```python
    return loss.item(), coach_rewards
```

Also update the type hint on line 130. Change:
```python
) -> float:
```
to:
```python
) -> Tuple[float, List[float]]:
```

**Step 2: Update the call site in main_loop.py**

In `crisp/workflow/main_loop.py`, line 59, change:
```python
        coach_loss = train_step.train_coach(
            ctx, problems, rollouts, discussion_results,
        )
```
to:
```python
        coach_loss, _coach_rewards = train_step.train_coach(
            ctx, problems, rollouts, discussion_results,
        )
```

(We'll use `_coach_rewards` later in Task 4 when adding collector support. For now it's discarded.)

**Step 3: Update test mocks**

In `tests/workflow/test_main_loop.py`, every `mock_train.train_coach.return_value` must change from a scalar to a tuple. Update these lines:

Line 53: `mock_train.train_coach.return_value = 0.1` -> `mock_train.train_coach.return_value = (0.1, [0.5])`
Line 81: `mock_train.train_coach.return_value = 0.1` -> `mock_train.train_coach.return_value = (0.1, [0.5])`
Line 122: `mock_train.train_coach.return_value = 0.1` -> `mock_train.train_coach.return_value = (0.1, [0.5])`
Line 152: Change the side_effect lambda. Currently:
```python
        mock_train.train_coach.side_effect = lambda *a, **kw: (call_order.append("train_coach"), 0.1)[1]
```
Change to:
```python
        mock_train.train_coach.side_effect = lambda *a, **kw: (call_order.append("train_coach"), (0.1, [0.5]))[1]
```
Line 187: `mock_train.train_coach.return_value = 0.1` -> `mock_train.train_coach.return_value = (0.1, [0.5])`
Line 214: `mock_train.train_coach.return_value = 0.1` -> `mock_train.train_coach.return_value = (0.1, [0.5])`

In `tests/workflow/test_train_step.py`, update the assertion for `train_coach` return:

Line 128: change:
```python
        loss = train_coach(ctx, problems, rollouts, discussion_results)
```
to:
```python
        loss, coach_rewards = train_coach(ctx, problems, rollouts, discussion_results)
```

Line 130: keep `assert isinstance(loss, float)` and add after it:
```python
    assert isinstance(coach_rewards, list)
    assert len(coach_rewards) == 1
```

Line 156: change:
```python
        train_coach(ctx, problems, rollouts, discussion_results)
```
to:
```python
        loss, coach_rewards = train_coach(ctx, problems, rollouts, discussion_results)
```

**Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: All 280 tests pass.

**Step 5: Commit**

```bash
git add crisp/workflow/train_step.py crisp/workflow/main_loop.py tests/workflow/test_train_step.py tests/workflow/test_main_loop.py
git commit -m "feat: train_coach returns (loss, coach_rewards) tuple"
```

---

### Task 3: Create StepCollector and IterationData

**Files:**
- Create: `crisp/workflow/collector.py`
- Create: `tests/workflow/test_collector.py`

**Step 1: Write the test**

Create `tests/workflow/test_collector.py`:
```python
"""Tests for StepCollector and IterationData."""
from crisp.workflow.collector import IterationData, StepCollector
from crisp.workflow.context import StepResult


def test_collector_starts_empty():
    collector = StepCollector()
    assert len(collector.iterations) == 0


def test_collector_records_iteration():
    collector = StepCollector()
    data = IterationData(
        iteration=0,
        problems=[],
        rollouts={0: [], 1: []},
        majority_answers={},
        discussion_results={0: [], 1: []},
        player_loss=0.5,
        coach_loss=0.1,
        coach_rewards=[0.5, 0.3],
        result=StepResult(
            player_loss=0.5, coach_loss=0.1, num_problems=2,
            num_discussions=1, player_accuracy=0.75, coach_iteration=True,
        ),
    )
    collector.record(data)
    assert len(collector.iterations) == 1
    assert collector.iterations[0].iteration == 0
    assert collector.iterations[0].coach_rewards == [0.5, 0.3]


def test_collector_accumulates_multiple():
    collector = StepCollector()
    for i in range(5):
        data = IterationData(
            iteration=i, problems=[], rollouts={0: [], 1: []},
            majority_answers={}, discussion_results={0: [], 1: []},
            player_loss=float(i), coach_loss=None, coach_rewards=None,
            result=StepResult(
                player_loss=float(i), coach_loss=None, num_problems=0,
                num_discussions=0, player_accuracy=0.0, coach_iteration=False,
            ),
        )
        collector.record(data)
    assert len(collector.iterations) == 5
    assert [d.iteration for d in collector.iterations] == [0, 1, 2, 3, 4]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_collector.py -v`
Expected: FAIL (ImportError — module doesn't exist yet)

**Step 3: Write the implementation**

Create `crisp/workflow/collector.py`:
```python
"""Observability hook for step() — captures intermediate data per iteration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from crisp.types import DiscussionResult, Problem, Rollout
from crisp.workflow.context import StepResult


@dataclass
class IterationData:
    """All intermediate data from one training iteration."""
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
    """Accumulates IterationData across the training loop."""

    def __init__(self) -> None:
        self.iterations: List[IterationData] = []

    def record(self, data: IterationData) -> None:
        self.iterations.append(data)
```

**Step 4: Run tests**

Run: `pytest tests/workflow/test_collector.py -v`
Expected: All 3 pass.

**Step 5: Commit**

```bash
git add crisp/workflow/collector.py tests/workflow/test_collector.py
git commit -m "feat: add StepCollector and IterationData for observability"
```

---

### Task 4: Wire collector into main_loop.step()

**Files:**
- Modify: `crisp/workflow/main_loop.py:10,55-78`
- Modify: `tests/workflow/test_main_loop.py` (add collector test)

**Step 1: Write the failing test**

Add to `tests/workflow/test_main_loop.py`:
```python
def test_step_populates_collector():
    """step() records IterationData when collector is provided."""
    from crisp.workflow.main_loop import step
    from crisp.workflow.collector import StepCollector

    ctx = _make_ctx()

    problems = [Problem(text="Q", ground_truth="1",
                        coach_embedding=np.zeros(384))]
    rollouts_dict = {
        0: [Rollout(problem_idx=0, player_id=0, tokens=[1], text="\\boxed{1}",
                     log_probs=[-0.1], answer="1", correct=True, reward=1.0)],
        1: [Rollout(problem_idx=0, player_id=1, tokens=[2], text="\\boxed{2}",
                     log_probs=[-0.2], answer="2", correct=False, reward=0.0)],
    }
    disc_results = {0: [], 1: []}
    majority = {(0, 0): "1", (1, 0): "2"}

    with patch("crisp.workflow.main_loop.coach_step") as mock_coach, \
         patch("crisp.workflow.main_loop.rollout_step") as mock_rollout, \
         patch("crisp.workflow.main_loop.discussion_step") as mock_disc, \
         patch("crisp.workflow.main_loop.train_step") as mock_train:

        mock_coach.generate_problems.return_value = problems
        mock_rollout.generate_all_rollouts.return_value = rollouts_dict
        mock_disc.run_discussion.return_value = (disc_results, majority)
        mock_train.train_player.return_value = 0.5
        mock_train.train_coach.return_value = (0.1, [0.75])

        collector = StepCollector()
        result = step(ctx, collector=collector)

    assert len(collector.iterations) == 1
    data = collector.iterations[0]
    assert data.iteration == 0
    assert data.problems is problems
    assert data.rollouts is rollouts_dict
    assert data.majority_answers is majority
    assert data.discussion_results is disc_results
    assert data.player_loss == 0.5
    assert data.coach_loss == 0.1
    assert data.coach_rewards == [0.75]
    assert data.result is result


def test_step_without_collector_still_works():
    """step() works normally when no collector is provided."""
    from crisp.workflow.main_loop import step

    ctx = _make_ctx()

    problems = [Problem(text="Q", ground_truth="1",
                        coach_embedding=np.zeros(384))]

    with patch("crisp.workflow.main_loop.coach_step") as mock_coach, \
         patch("crisp.workflow.main_loop.rollout_step") as mock_rollout, \
         patch("crisp.workflow.main_loop.discussion_step") as mock_disc, \
         patch("crisp.workflow.main_loop.train_step") as mock_train:

        mock_coach.generate_problems.return_value = problems
        mock_rollout.generate_all_rollouts.return_value = {0: [], 1: []}
        mock_disc.run_discussion.return_value = ({0: [], 1: []}, {})
        mock_train.train_player.return_value = 0.5
        mock_train.train_coach.return_value = (0.1, [0.5])

        result = step(ctx)

    assert result.player_loss == 0.5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_main_loop.py::test_step_populates_collector -v`
Expected: FAIL (step() doesn't accept collector param yet)

**Step 3: Update main_loop.step()**

Replace `crisp/workflow/main_loop.py` entirely:
```python
"""Steps 1-13 orchestration: the CRISP training main loop."""
from __future__ import annotations

from typing import Any, Optional

from crisp.workflow import coach_step, discussion_step, rollout_step, train_step
from crisp.workflow.collector import IterationData, StepCollector
from crisp.workflow.context import StepResult


def step(ctx: Any, collector: Optional[StepCollector] = None) -> StepResult:
    """Execute one full training iteration (Steps 1-13).

    Phase 1 (Generation): vLLM active, DeepSpeed offloaded
    Phase 2 (Training): DeepSpeed active, vLLM idle

    Coach training gated by ctx.config.coach.update_freq.

    Args:
        ctx: WorkflowContext with all infra handles.
        collector: Optional StepCollector to capture intermediate data.
    """
    # --- Generation phase (vLLM active, DS offloaded) ---
    ctx.ds_player.offload_states()
    ctx.ds_coach.offload_states()

    # Step 1: Coach generates problems
    problems = coach_step.generate_problems(ctx)

    # Step 2-4: Players generate rollouts, verify, compute rewards
    rollouts = rollout_step.generate_all_rollouts(ctx, problems)

    # Steps 5-6: Discussion trigger + execution
    discussion_results, majority_answers = discussion_step.run_discussion(
        ctx, rollouts, problems,
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

    # --- Training phase ---
    ctx.ds_player.reload_states()

    # Steps 7-10.5: Player training
    player_loss = train_step.train_player(
        ctx, rollouts, discussion_results, majority_answers, problems,
    )

    # Steps 11-12.5: Coach training (gated by update_freq)
    coach_loss = None
    coach_rewards = None
    is_coach_iter = ctx.iteration % ctx.config.coach.update_freq == 0
    if is_coach_iter:
        ctx.ds_coach.reload_states()
        coach_loss, coach_rewards = train_step.train_coach(
            ctx, problems, rollouts, discussion_results,
        )
        ctx.ds_coach.offload_states()

    # INVARIANT: push AFTER train_coach() so current batch's embeddings
    # are NOT included in their own cross-batch repetition penalty.
    ctx.rep_buffer.push([p.coach_embedding for p in problems])

    # Step 13: Increment iteration
    ctx.iteration += 1

    result = StepResult(
        player_loss=player_loss,
        coach_loss=coach_loss,
        num_problems=len(problems),
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
```

**Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass (280 existing + 5 new = 285).

**Step 5: Commit**

```bash
git add crisp/workflow/main_loop.py tests/workflow/test_main_loop.py
git commit -m "feat: wire StepCollector into main_loop.step()"
```

---

### Task 5: Create smoke test config

**Files:**
- Create: `configs/smoke_test.yaml`

**Step 1: Create the config**

Create `configs/smoke_test.yaml`:
```yaml
# Smoke test config: Qwen3-0.6B on single GPU, small batches.
# Usage: python scripts/smoke_test.py --config configs/smoke_test.yaml

training:
  model_name: "Qwen/Qwen3-0.6B"
  coach_model_name: "Qwen/Qwen3-0.6B"
  num_iterations: 10
  eval_freq: 0
  save_freq: 0
  attn_implementation: "eager"

infra:
  num_gpus_per_node: 1
  vllm_num_engines: 1
  vllm_enable_sleep: true
  max_model_len: 4096
  zero_stage: 2
  bf16: true
  learning_rate: 0.00001

player:
  rollouts_per_problem: 4

coach:
  batch_size: 4
```

**Step 2: Verify config loads**

Run: `python -c "from crisp.config_loader import load_config; c = load_config('configs/smoke_test.yaml'); print(c.training.model_name, c.coach.batch_size)"`
Expected: `Qwen/Qwen3-0.6B 4`

**Step 3: Commit**

```bash
git add configs/smoke_test.yaml
git commit -m "feat: add smoke test config for Qwen3-0.6B"
```

---

### Task 6: Create the report generator

**Files:**
- Create: `scripts/write_smoke_report.py`
- Create: `tests/test_smoke_report.py`

**Step 1: Write a test for the report generator**

Create `tests/test_smoke_report.py`:
```python
"""Tests for the smoke test report generator."""
import numpy as np

from crisp.config import CRISPConfig
from crisp.types import DiscussionResult, Problem, Rollout
from crisp.workflow.collector import IterationData, StepCollector
from crisp.workflow.context import StepResult


def _make_collector_with_data():
    """Build a StepCollector with 2 synthetic iterations."""
    collector = StepCollector()

    for i in range(2):
        problems = [
            Problem(text=f"What is {i}+1?", ground_truth=str(i + 1),
                    coach_embedding=np.zeros(384, dtype=np.float32)),
            Problem(text=f"What is {i}+2?", ground_truth=str(i + 2),
                    coach_embedding=np.zeros(384, dtype=np.float32)),
        ]
        rollouts = {
            0: [
                Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text=f"\\boxed{{{i+1}}}",
                        log_probs=[-0.1, -0.2], answer=str(i + 1), correct=True, reward=1.0),
                Rollout(problem_idx=1, player_id=0, tokens=[3, 4], text="\\boxed{99}",
                        log_probs=[-0.3, -0.4], answer="99", correct=False, reward=0.0),
            ],
            1: [
                Rollout(problem_idx=0, player_id=1, tokens=[5, 6], text="\\boxed{99}",
                        log_probs=[-0.5, -0.6], answer="99", correct=False, reward=0.0),
                Rollout(problem_idx=1, player_id=1, tokens=[7, 8], text=f"\\boxed{{{i+2}}}",
                        log_probs=[-0.7, -0.8], answer=str(i + 2), correct=True, reward=1.0),
            ],
        }
        disc_results = {
            0: [DiscussionResult(
                problem_idx=0, player_id=0, tokens=[9], text="EVALUATION: checking\nFINAL ANSWER: \\boxed{1}",
                log_probs=[-0.1], evaluation_text="checking", final_answer=str(i + 1),
                correct=True, reward=1.0,
            )],
            1: [DiscussionResult(
                problem_idx=0, player_id=1, tokens=[10], text="EVALUATION: ok\nFINAL ANSWER: \\boxed{1}",
                log_probs=[-0.2], evaluation_text="ok", final_answer=str(i + 1),
                correct=True, reward=1.0,
            )],
        }
        majority = {(0, 0): str(i + 1), (1, 0): "99", (0, 1): "99", (1, 1): str(i + 2)}

        result = StepResult(
            player_loss=0.5 - i * 0.1, coach_loss=0.3 - i * 0.05,
            num_problems=2, num_discussions=1,
            player_accuracy=0.5, coach_iteration=True,
        )
        collector.record(IterationData(
            iteration=i, problems=problems, rollouts=rollouts,
            majority_answers=majority, discussion_results=disc_results,
            player_loss=result.player_loss, coach_loss=result.coach_loss,
            coach_rewards=[0.75, 0.50], result=result,
        ))
    return collector


def test_report_generates_markdown():
    from scripts.write_smoke_report import generate_report
    collector = _make_collector_with_data()
    config = CRISPConfig()
    report = generate_report(collector, config)

    assert isinstance(report, str)
    assert "# CRISP Smoke Test Report" in report
    assert "## Summary" in report


def test_report_contains_problems():
    from scripts.write_smoke_report import generate_report
    collector = _make_collector_with_data()
    report = generate_report(collector, CRISPConfig())

    assert "What is 0+1?" in report
    assert "Ground Truth" in report


def test_report_contains_rollout_table():
    from scripts.write_smoke_report import generate_report
    collector = _make_collector_with_data()
    report = generate_report(collector, CRISPConfig())

    assert "| Player" in report
    assert "Alice" in report
    assert "Bob" in report


def test_report_contains_discussion():
    from scripts.write_smoke_report import generate_report
    collector = _make_collector_with_data()
    report = generate_report(collector, CRISPConfig())

    assert "Discussion" in report
    assert "DISAGREE" in report or "Disagree" in report or "disagree" in report


def test_report_contains_summary_table():
    from scripts.write_smoke_report import generate_report
    collector = _make_collector_with_data()
    report = generate_report(collector, CRISPConfig())

    # Summary table should have iteration numbers
    assert "| 0" in report
    assert "| 1" in report


def test_report_contains_coach_rewards():
    from scripts.write_smoke_report import generate_report
    collector = _make_collector_with_data()
    report = generate_report(collector, CRISPConfig())

    assert "Coach Rewards" in report
    assert "0.75" in report
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke_report.py -v`
Expected: FAIL (ImportError)

**Step 3: Write the implementation**

Create `scripts/__init__.py` (empty, for importability):
```python
```

Create `scripts/write_smoke_report.py`:
```python
"""Generate a Markdown smoke test report from StepCollector data."""
from __future__ import annotations

from datetime import datetime
from typing import List

from crisp.config import CRISPConfig
from crisp.workflow.collector import IterationData, StepCollector


PLAYER_NAMES = {0: "Alice", 1: "Bob"}

# Show full detail for these iterations (first 3 + last)
def _detail_iterations(total: int) -> List[int]:
    iters = list(range(min(3, total)))
    if total > 3 and (total - 1) not in iters:
        iters.append(total - 1)
    return iters


def generate_report(collector: StepCollector, config: CRISPConfig) -> str:
    """Generate a Markdown report string from collected iteration data."""
    lines: List[str] = []

    model = config.training.model_name
    total = len(collector.iterations)
    batch_size = config.coach.batch_size
    rollouts_per = config.player.rollouts_per_problem

    lines.append("# CRISP Smoke Test Report")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Model**: {model}")
    lines.append(f"**Iterations**: {total} | **Problems/iter**: {batch_size} | **Rollouts/problem**: {rollouts_per}")
    lines.append("")

    # --- Summary table ---
    lines.append("## Summary")
    lines.append("")
    lines.append("| Iter | Problems | Discussions | Accuracy | Player Loss | Coach Loss |")
    lines.append("|------|----------|-------------|----------|-------------|------------|")
    for data in collector.iterations:
        r = data.result
        cl = f"{r.coach_loss:.4f}" if r.coach_loss is not None else "N/A"
        lines.append(
            f"| {data.iteration} | {r.num_problems} | {r.num_discussions} "
            f"| {r.player_accuracy:.3f} | {r.player_loss:.4f} | {cl} |"
        )
    lines.append("")

    # --- Detailed sections ---
    detail_iters = _detail_iterations(total)
    for data in collector.iterations:
        if data.iteration not in detail_iters:
            continue
        lines.append(f"## Iteration {data.iteration}")
        lines.append("")
        _write_problems_section(lines, data)
        _write_rollouts_section(lines, data)
        _write_discussion_section(lines, data)
        _write_coach_rewards_section(lines, data)
        lines.append(f"**Player Loss**: {data.player_loss:.4f}")
        if data.coach_loss is not None:
            lines.append(f"**Coach Loss**: {data.coach_loss:.4f}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # --- Trend summary ---
    lines.append("## Trends")
    lines.append("")
    lines.append("**Accuracy**: " + " -> ".join(
        f"{d.result.player_accuracy:.3f}" for d in collector.iterations
    ))
    lines.append("**Player Loss**: " + " -> ".join(
        f"{d.player_loss:.4f}" for d in collector.iterations
    ))
    coach_losses = [d.coach_loss for d in collector.iterations if d.coach_loss is not None]
    if coach_losses:
        lines.append("**Coach Loss**: " + " -> ".join(f"{cl:.4f}" for cl in coach_losses))
    lines.append("")

    return "\n".join(lines)


def _write_problems_section(lines: List[str], data: IterationData) -> None:
    lines.append("### Problems Generated by Coach")
    lines.append("")
    for i, prob in enumerate(data.problems):
        gt = prob.ground_truth
        text = prob.text[:200] + "..." if len(prob.text) > 200 else prob.text
        lines.append(f"**Problem {i + 1}**: \"{text}\"")
        lines.append(f"**Ground Truth**: `{gt}`")
        lines.append("")


def _write_rollouts_section(lines: List[str], data: IterationData) -> None:
    for prob_idx, prob in enumerate(data.problems):
        lines.append(f"### Player Rollouts - Problem {prob_idx + 1}")
        lines.append("")
        lines.append("| Player | # | Answer | Correct | Reward |")
        lines.append("|--------|---|--------|---------|--------|")

        for pid in [0, 1]:
            player_name = PLAYER_NAMES[pid]
            rollouts = [r for r in data.rollouts.get(pid, []) if r.problem_idx == prob_idx]
            for j, r in enumerate(rollouts):
                ans = r.answer if r.answer is not None else "-"
                correct_str = "Y" if r.correct else ("N" if r.correct is not None else "-")
                lines.append(f"| {player_name} | {j + 1} | {ans} | {correct_str} | {r.reward:.2f} |")
        lines.append("")


def _write_discussion_section(lines: List[str], data: IterationData) -> None:
    # Find which problems had discussion
    discussed = set()
    for pid in data.discussion_results:
        for dr in data.discussion_results[pid]:
            discussed.add(dr.problem_idx)

    if not discussed:
        lines.append("### Discussion")
        lines.append("No discussions triggered (all problems had agreement).")
        lines.append("")
        return

    lines.append("### Discussion")
    lines.append("")

    for prob_idx in sorted(discussed):
        prob = data.problems[prob_idx] if prob_idx < len(data.problems) else None
        prob_text = prob.text[:100] if prob else "?"

        maj_a = data.majority_answers.get((0, prob_idx), "None")
        maj_b = data.majority_answers.get((1, prob_idx), "None")
        lines.append(f"**Problem {prob_idx + 1}** (\"{prob_text[:60]}...\")")
        lines.append(f"  Alice majority=`{maj_a}`, Bob majority=`{maj_b}` -> DISAGREE")
        lines.append("")

        for pid in [0, 1]:
            player_name = PLAYER_NAMES[pid]
            drs = [dr for dr in data.discussion_results.get(pid, []) if dr.problem_idx == prob_idx]
            for dr in drs:
                correct_str = "Y" if dr.correct else "N"
                lines.append(f"  **{player_name} post-discussion**:")
                eval_text = dr.evaluation_text[:150] if dr.evaluation_text else "(empty)"
                lines.append(f"  > Evaluation: {eval_text}")
                lines.append(f"  > Final Answer: `{dr.final_answer}` | Correct: {correct_str} | Reward: {dr.reward:.2f}")
                lines.append("")


def _write_coach_rewards_section(lines: List[str], data: IterationData) -> None:
    if data.coach_rewards is None:
        return

    lines.append("### Coach Rewards")
    lines.append("")
    lines.append("| Problem | Reward |")
    lines.append("|---------|--------|")
    for i, reward in enumerate(data.coach_rewards):
        lines.append(f"| {i + 1} | {reward:.4f} |")
    lines.append("")
```

**Step 4: Run tests**

Run: `pytest tests/test_smoke_report.py -v`
Expected: All 6 pass.

**Step 5: Commit**

```bash
git add scripts/__init__.py scripts/write_smoke_report.py tests/test_smoke_report.py
git commit -m "feat: add Markdown report generator for smoke test"
```

---

### Task 7: Create the smoke test entry point

**Files:**
- Create: `scripts/smoke_test.py`

**Step 1: Create the script**

Create `scripts/smoke_test.py`:
```python
"""CRISP end-to-end smoke test.

Runs the full pipeline for N iterations and generates a Markdown report
showing coach problems, player rollouts, discussions, rewards, and losses.

Usage:
    python scripts/smoke_test.py --config configs/smoke_test.yaml --output smoke_report.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, List

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CRISP smoke test")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output", type=str, default="smoke_report.md",
                        help="Output path for the Markdown report")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Config overrides as key=value",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        level=logging.INFO,
    )

    args = parse_args(argv)

    from crisp.config_loader import load_config
    from crisp.train import init_infra, parse_overrides
    from crisp.workflow.collector import StepCollector
    from crisp.workflow.main_loop import step
    from scripts.write_smoke_report import generate_report

    overrides = parse_overrides(args.override)
    config = load_config(args.config, overrides=overrides)
    n_iters = config.training.num_iterations

    logger.info("Smoke test: %d iterations, model=%s", n_iters, config.training.model_name)

    ctx = init_infra(config)
    collector = StepCollector()

    for i in range(n_iters):
        result = step(ctx, collector=collector)
        logger.info(
            "iter=%d problems=%d discussions=%d accuracy=%.3f "
            "player_loss=%.4f coach_loss=%s",
            i, result.num_problems, result.num_discussions,
            result.player_accuracy, result.player_loss,
            f"{result.coach_loss:.4f}" if result.coach_loss is not None else "N/A",
        )

    report = generate_report(collector, config)

    with open(args.output, "w") as f:
        f.write(report)

    logger.info("Report written to %s", args.output)


if __name__ == "__main__":
    main()
```

**Step 2: Verify it parses args**

Run: `python -c "from scripts.smoke_test import parse_args; a = parse_args(['--config', 'configs/smoke_test.yaml']); print(a.config, a.output)"`
Expected: `configs/smoke_test.yaml smoke_report.md`

**Step 3: Commit**

```bash
git add scripts/smoke_test.py
git commit -m "feat: add smoke test entry point script"
```

---

### Task 8: Create the Modal wrapper

**Files:**
- Create: `scripts/modal_smoke.py`

**Step 1: Create the Modal script**

Create `scripts/modal_smoke.py`:
```python
"""Modal wrapper for CRISP smoke test.

Usage:
    modal run scripts/modal_smoke.py
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "vllm>=0.8.2",
        "ray>=2.9",
        "deepspeed>=0.16.4",
        "transformers>=4.40",
        "sentence-transformers",
        "peft>=0.7",
        "numpy",
        "sympy",
        "scipy",
        "pyyaml",
        "packaging",
    )
    .copy_local_dir("crisp", "/root/project/crisp")
    .copy_local_dir("configs", "/root/project/configs")
    .copy_local_dir("scripts", "/root/project/scripts")
    .copy_local_file("pyproject.toml", "/root/project/pyproject.toml")
)

app = modal.App("crisp-smoke-test", image=image)
vol = modal.Volume.from_name("crisp-smoke-reports", create_if_missing=True)


@app.function(
    gpu="A100",
    timeout=1800,
    volumes={"/reports": vol},
)
def run_smoke_test():
    import os
    import subprocess
    import sys

    os.chdir("/root/project")
    sys.path.insert(0, "/root/project")

    from scripts.smoke_test import main

    output_path = "/reports/smoke_report.md"
    main(["--config", "configs/smoke_test.yaml", "--output", output_path])

    with open(output_path) as f:
        report = f.read()

    print(report)
    return report


@app.local_entrypoint()
def main():
    report = run_smoke_test.remote()

    # Also save locally
    with open("smoke_report.md", "w") as f:
        f.write(report)

    print(f"\nReport saved to smoke_report.md ({len(report)} chars)")
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('scripts/modal_smoke.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add scripts/modal_smoke.py
git commit -m "feat: add Modal wrapper for smoke test"
```

---

### Task 9: Run full test suite and verify no regressions

**Files:** None (verification only)

**Step 1: Run all existing tests**

Run: `pytest tests/ -v`
Expected: All tests pass (280 existing + 5 new collector/main_loop tests + 6 report tests = ~291)

**Step 2: Verify config loading**

Run: `python -c "from crisp.config_loader import load_config; c = load_config('configs/smoke_test.yaml'); assert c.coach.update_freq == 1; assert c.training.model_name == 'Qwen/Qwen3-0.6B'; print('Config OK')"`
Expected: `Config OK`

**Step 3: Final commit (if any fixups needed)**

Only commit if there are fixes from the verification run.
