# Workflow Orchestration Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `crisp/workflow/` — the orchestration layer that wires infra (Ray/vLLM/DeepSpeed) to domain logic (rewards, discussion, verification, training), implementing Steps 1-13 of the CRISP main loop.

**Architecture:** WorkflowContext dataclass holds all infra handles and stateful objects. Each step module (coach_step, rollout_step, discussion_step, train_step) is a thin orchestrator calling existing domain functions. main_loop.step() calls them in sequence with coach update frequency gating and proper sleep-mode management.

**Tech Stack:** Python 3.10+, PyTorch, existing crisp/ modules. Tests mock infra (vLLM/DeepSpeed/Ray) but use real domain logic.

---

### Task 1: Add config fields (update_freq, templates) to CoachConfig

**Files:**
- Modify: `crisp/config.py:18-25`
- Modify: `tests/test_config.py`

**Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_coach_config_update_freq():
    """CoachConfig has update_freq with default 5."""
    from crisp.config import CoachConfig
    cfg = CoachConfig()
    assert cfg.update_freq == 5


def test_coach_config_templates():
    """CoachConfig has discussion and coach prompt templates."""
    from crisp.config import CoachConfig
    cfg = CoachConfig()
    assert isinstance(cfg.discussion_template, str)
    assert "{problem}" in cfg.discussion_template
    assert "{own_solution}" in cfg.discussion_template
    assert "{other_solution}" in cfg.discussion_template
    assert isinstance(cfg.coach_prompt_template, str)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_coach_config_update_freq tests/test_config.py::test_coach_config_templates -v`
Expected: FAIL with `AttributeError`

**Step 3: Write minimal implementation**

Modify `crisp/config.py`, add to `CoachConfig`:

```python
@dataclass
class CoachConfig:
    """Coach training hyperparameters."""
    batch_size: int = 8
    discussion_alpha: float = 0.3
    repetition_lambda: float = 1.0
    repetition_tau: float = 0.85
    repetition_window: int = 10  # W batches
    embedding_dim: int = 384
    update_freq: int = 5  # Coach trains every N player iterations
    discussion_template: str = (
        "You previously solved this problem:\n{problem}\n\n"
        "Your solution: {own_solution}\n"
        "Another student's solution: {other_solution}\n\n"
        "EVALUATION:\n[Analyze both solutions]\n\n"
        "FINAL ANSWER: \\boxed{...}"
    )
    coach_prompt_template: str = (
        "Create a math problem that is challenging but solvable. "
        "Show your solution and put the final answer in \\boxed{}."
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add crisp/config.py tests/test_config.py
git commit -m "feat: add update_freq and prompt templates to CoachConfig"
```

---

### Task 2: Create WorkflowContext and StepResult (context.py)

**Files:**
- Create: `crisp/workflow/context.py`
- Create: `tests/workflow/__init__.py`
- Create: `tests/workflow/test_context.py`

**Step 1: Write the failing test**

Create `tests/workflow/__init__.py` (empty).

Create `tests/workflow/test_context.py`:

```python
"""Tests for WorkflowContext and StepResult."""
from unittest.mock import MagicMock

from crisp.config import CRISPConfig
from crisp.rewards.ema_tracker import EMATracker
from crisp.rewards.repetition_buffer import RepetitionBuffer


def test_workflow_context_creation():
    """WorkflowContext holds all infra handles and stateful objects."""
    from crisp.workflow.context import WorkflowContext

    ctx = WorkflowContext(
        player_vllm=[MagicMock()],
        coach_vllm=[MagicMock()],
        ref_model=MagicMock(),
        ds_player=MagicMock(),
        ds_coach=MagicMock(),
        config=CRISPConfig(),
        player_ema=EMATracker(),
        coach_ema=EMATracker(),
        rep_buffer=RepetitionBuffer(),
    )
    assert ctx.iteration == 0
    assert isinstance(ctx.config, CRISPConfig)
    assert isinstance(ctx.player_ema, EMATracker)


def test_workflow_context_iteration_mutable():
    """iteration field is mutable."""
    from crisp.workflow.context import WorkflowContext

    ctx = WorkflowContext(
        player_vllm=[], coach_vllm=[], ref_model=None,
        ds_player=None, ds_coach=None, config=CRISPConfig(),
        player_ema=EMATracker(), coach_ema=EMATracker(),
        rep_buffer=RepetitionBuffer(),
    )
    ctx.iteration = 5
    assert ctx.iteration == 5


def test_step_result_creation():
    """StepResult holds step output metrics."""
    from crisp.workflow.context import StepResult

    result = StepResult(
        player_loss=0.5,
        coach_loss=None,
        num_problems=8,
        num_discussions=3,
        player_accuracy=0.75,
        coach_iteration=False,
    )
    assert result.player_loss == 0.5
    assert result.coach_loss is None
    assert result.coach_iteration is False


def test_step_result_coach_iteration():
    """StepResult with coach training."""
    from crisp.workflow.context import StepResult

    result = StepResult(
        player_loss=0.3,
        coach_loss=0.1,
        num_problems=8,
        num_discussions=2,
        player_accuracy=0.6,
        coach_iteration=True,
    )
    assert result.coach_loss == 0.1
    assert result.coach_iteration is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_context.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `crisp/workflow/context.py`:

```python
"""WorkflowContext and StepResult for CRISP workflow orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from crisp.config import CRISPConfig
from crisp.rewards.ema_tracker import EMATracker
from crisp.rewards.repetition_buffer import RepetitionBuffer


@dataclass
class WorkflowContext:
    """Holds all infra handles and stateful objects for the training loop.

    Passed to each step function for dependency injection.
    """
    player_vllm: List[Any]       # vLLM engine actors
    coach_vllm: List[Any]        # vLLM engine actors
    ref_model: Any               # Frozen reference policy (NEVER updated)
    ds_player: Any               # DeepSpeed strategy for player
    ds_coach: Any                # DeepSpeed strategy for coach
    config: CRISPConfig
    # Stateful — persist across iterations
    player_ema: EMATracker
    coach_ema: EMATracker
    rep_buffer: RepetitionBuffer
    iteration: int = 0


@dataclass
class StepResult:
    """Output metrics from a single training step."""
    player_loss: float
    coach_loss: Optional[float]  # None on non-coach-update iterations
    num_problems: int
    num_discussions: int
    player_accuracy: float       # fraction correct pre-discussion
    coach_iteration: bool        # whether coach was updated this step
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/workflow/test_context.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add crisp/workflow/context.py tests/workflow/__init__.py tests/workflow/test_context.py
git commit -m "feat: add WorkflowContext and StepResult dataclasses"
```

---

### Task 3: Implement build_player_batch and build_coach_batch (batch_builder.py)

**Files:**
- Modify: `crisp/training/batch_builder.py`
- Modify: `tests/test_batch_builder.py`

**Background:** `build_player_batch` currently raises `NotImplementedError`. We need it to assemble `TrainingBatch` from rollouts + advantages. Also need a new `build_coach_batch` for coach sequences.

A `TrainingBatch` has:
- `sequences: List[TokenSequence]` — each with `.tokens` and `.log_probs`
- `advantages: List[float]` — one per sequence
- `ref_log_probs: List[List[float]]` — one list per sequence
- `is_post_discussion: List[bool]` — flag per sequence

**Step 1: Write the failing test**

Read the existing test file first. Then add these tests:

```python
"""Tests for batch_builder — build_player_batch and build_coach_batch."""
import torch

from crisp.types import Rollout, DiscussionResult, Problem, TokenSequence, TrainingBatch


def test_build_player_batch_basic():
    """build_player_batch creates a TrainingBatch from rollouts and advantages."""
    from crisp.training.batch_builder import build_player_batch

    rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=[1, 2, 3], text="a",
                log_probs=[-0.1, -0.2, -0.3], answer="1", correct=True, reward=1.0),
        Rollout(problem_idx=0, player_id=0, tokens=[4, 5], text="b",
                log_probs=[-0.4, -0.5], answer="2", correct=False, reward=0.0),
    ]
    advantages = [0.5, -0.5]

    batch = build_player_batch(rollouts, advantages)
    assert isinstance(batch, TrainingBatch)
    assert len(batch.sequences) == 2
    assert len(batch.advantages) == 2
    assert batch.advantages == [0.5, -0.5]
    # All pre-discussion
    assert batch.is_post_discussion == [False, False]


def test_build_player_batch_with_discussion():
    """build_player_batch handles mixed pre/post-discussion sequences."""
    from crisp.training.batch_builder import build_player_batch

    rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text="a",
                log_probs=[-0.1, -0.2], answer="1", correct=True, reward=1.0),
    ]
    discussion_results = [
        DiscussionResult(problem_idx=0, player_id=0, tokens=[3, 4, 5], text="b",
                         log_probs=[-0.3, -0.4, -0.5], evaluation_text="eval",
                         final_answer="1", correct=True, reward=1.3),
    ]
    pre_advantages = [0.5]
    post_advantages = [0.8]

    batch = build_player_batch(
        rollouts, pre_advantages,
        discussion_results=discussion_results,
        post_advantages=post_advantages,
    )
    assert len(batch.sequences) == 2
    assert batch.is_post_discussion == [False, True]
    assert batch.advantages == [0.5, 0.8]


def test_build_player_batch_empty():
    """build_player_batch with empty input returns empty batch."""
    from crisp.training.batch_builder import build_player_batch

    batch = build_player_batch([], [])
    assert len(batch.sequences) == 0
    assert len(batch.advantages) == 0


def test_build_coach_batch_basic():
    """build_coach_batch creates a TrainingBatch from problems and advantages."""
    from crisp.training.batch_builder import build_coach_batch

    problems = [
        Problem(text="What is 2+2?", ground_truth="4",
                coach_sequence=TokenSequence(tokens=[10, 20, 30],
                                             log_probs=[-0.1, -0.2, -0.3])),
        Problem(text="What is 3+3?", ground_truth="6",
                coach_sequence=TokenSequence(tokens=[40, 50],
                                             log_probs=[-0.4, -0.5])),
    ]
    advantages = [0.3, -0.3]

    batch = build_coach_batch(problems, advantages)
    assert isinstance(batch, TrainingBatch)
    assert len(batch.sequences) == 2
    assert batch.advantages == [0.3, -0.3]
    assert batch.sequences[0].tokens == [10, 20, 30]
    # Coach batches are never post-discussion
    assert batch.is_post_discussion == [False, False]


def test_build_coach_batch_skips_no_sequence():
    """build_coach_batch skips problems without coach_sequence."""
    from crisp.training.batch_builder import build_coach_batch

    problems = [
        Problem(text="Q1", ground_truth="1", coach_sequence=None),
        Problem(text="Q2", ground_truth="2",
                coach_sequence=TokenSequence(tokens=[1, 2], log_probs=[-0.1, -0.2])),
    ]
    advantages = [0.5, -0.5]

    batch = build_coach_batch(problems, advantages)
    # Only the problem with a sequence is included
    assert len(batch.sequences) == 1
    assert batch.advantages == [-0.5]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_batch_builder.py -v`
Expected: FAIL — `build_player_batch` raises `NotImplementedError`, `build_coach_batch` does not exist

**Step 3: Write minimal implementation**

Replace `build_player_batch` and add `build_coach_batch` in `crisp/training/batch_builder.py`:

```python
"""Dynamic sampling filter and training batch assembly."""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional

from crisp.types import DiscussionResult, Problem, Rollout, TokenSequence, TrainingBatch


def filter_dynamic_sampling(rollouts: List[Rollout]) -> List[Rollout]:
    """Filter out problems where all rollouts have identical rewards.

    These produce zero advantage variance and waste gradient computation.
    """
    # Group by problem
    by_problem: dict[int, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        by_problem[r.problem_idx].append(r)

    result = []
    for prob_idx, prob_rollouts in by_problem.items():
        rewards = [r.reward for r in prob_rollouts]
        if len(set(rewards)) > 1:
            result.extend(prob_rollouts)

    return result


def build_player_batch(
    rollouts: List[Rollout],
    pre_advantages: List[float],
    discussion_results: Optional[List[DiscussionResult]] = None,
    post_advantages: Optional[List[float]] = None,
) -> TrainingBatch:
    """Build a training batch from player rollouts and optional discussion results.

    Combines pre-discussion rollouts and post-discussion results into a single
    TrainingBatch. Each sequence's log_probs become the "old" log-probs for
    importance ratio computation.

    Args:
        rollouts: Pre-discussion rollouts (already filtered by dynamic sampling).
        pre_advantages: One advantage per rollout.
        discussion_results: Optional post-discussion results.
        post_advantages: One advantage per discussion result (required if discussion_results given).
    """
    sequences: List[TokenSequence] = []
    advantages: List[float] = []
    is_post: List[bool] = []

    # Pre-discussion rollouts
    for rollout, adv in zip(rollouts, pre_advantages):
        sequences.append(TokenSequence(
            tokens=rollout.tokens,
            log_probs=rollout.log_probs,
            text=rollout.text,
        ))
        advantages.append(adv)
        is_post.append(False)

    # Post-discussion results
    if discussion_results and post_advantages:
        for dr, adv in zip(discussion_results, post_advantages):
            sequences.append(TokenSequence(
                tokens=dr.tokens,
                log_probs=dr.log_probs,
                text=dr.text,
            ))
            advantages.append(adv)
            is_post.append(True)

    return TrainingBatch(
        sequences=sequences,
        advantages=advantages,
        ref_log_probs=[],  # Populated later by ref_model.forward()
        is_post_discussion=is_post,
    )


def build_coach_batch(
    problems: List[Problem],
    advantages: List[float],
) -> TrainingBatch:
    """Build a training batch from coach problem sequences.

    Skips problems without a coach_sequence (shouldn't happen in normal flow,
    but guards against incomplete generation).
    """
    sequences: List[TokenSequence] = []
    filtered_advantages: List[float] = []

    for problem, adv in zip(problems, advantages):
        if problem.coach_sequence is None:
            continue
        sequences.append(problem.coach_sequence)
        filtered_advantages.append(adv)

    return TrainingBatch(
        sequences=sequences,
        advantages=filtered_advantages,
        ref_log_probs=[],
        is_post_discussion=[False] * len(sequences),
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_batch_builder.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add crisp/training/batch_builder.py tests/test_batch_builder.py
git commit -m "feat: implement build_player_batch and build_coach_batch"
```

---

### Task 4: Implement coach_step.py (Step 1: Problem generation)

**Files:**
- Create: `crisp/workflow/coach_step.py`
- Create: `tests/workflow/test_coach_step.py`

**What this does:** Calls `generate_samples()` on coach vLLM engines, parses each output to extract problem text + ground_truth via `extract_boxed()`, computes sentence-transformer embeddings for the repetition buffer.

**Step 1: Write the failing test**

Create `tests/workflow/test_coach_step.py`:

```python
"""Tests for coach_step — mock vLLM, real answer extraction."""
from unittest.mock import MagicMock, patch

import numpy as np

from crisp.config import CRISPConfig
from crisp.rewards.ema_tracker import EMATracker
from crisp.rewards.repetition_buffer import RepetitionBuffer
from crisp.types import Problem


def _make_ctx(**overrides):
    """Helper to create a WorkflowContext with mocked infra."""
    from crisp.workflow.context import WorkflowContext
    defaults = dict(
        player_vllm=[MagicMock()],
        coach_vllm=[MagicMock()],
        ref_model=MagicMock(),
        ds_player=MagicMock(),
        ds_coach=MagicMock(),
        config=CRISPConfig(),
        player_ema=EMATracker(),
        coach_ema=EMATracker(),
        rep_buffer=RepetitionBuffer(),
    )
    defaults.update(overrides)
    return WorkflowContext(**defaults)


def test_generate_problems_extracts_ground_truth():
    """generate_problems parses \\boxed{} from coach output as ground_truth."""
    from crisp.workflow.coach_step import generate_problems

    ctx = _make_ctx()

    # Mock generate_samples to return rollouts with text containing \boxed{}
    from crisp.types import Rollout
    mock_rollouts = [
        Rollout(problem_idx=0, player_id=-1, tokens=[1, 2, 3],
                text="What is 2+2? The answer is \\boxed{4}",
                log_probs=[-0.1, -0.2, -0.3]),
        Rollout(problem_idx=1, player_id=-1, tokens=[4, 5, 6],
                text="Solve x^2=9. Solution: \\boxed{3}",
                log_probs=[-0.4, -0.5, -0.6]),
    ]

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.random.randn(2, 384).astype(np.float32)

    with patch("crisp.workflow.coach_step.generate_samples", return_value=mock_rollouts), \
         patch("crisp.workflow.coach_step._get_embedder", return_value=mock_embedder):
        problems = generate_problems(ctx)

    assert len(problems) == 2
    assert isinstance(problems[0], Problem)
    assert problems[0].ground_truth == "4"
    assert problems[1].ground_truth == "3"
    # Problem text should not include the \boxed{} answer
    assert "\\boxed{4}" not in problems[0].text
    assert problems[0].coach_embedding is not None
    assert problems[0].coach_embedding.shape == (384,)


def test_generate_problems_skips_no_boxed():
    """generate_problems skips coach outputs without \\boxed{} answer."""
    from crisp.workflow.coach_step import generate_problems
    from crisp.types import Rollout

    ctx = _make_ctx()

    mock_rollouts = [
        Rollout(problem_idx=0, player_id=-1, tokens=[1, 2],
                text="What is 2+2? The answer is 4",  # no \boxed{}
                log_probs=[-0.1, -0.2]),
        Rollout(problem_idx=1, player_id=-1, tokens=[3, 4],
                text="Solve: \\boxed{7}",
                log_probs=[-0.3, -0.4]),
    ]

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.random.randn(1, 384).astype(np.float32)

    with patch("crisp.workflow.coach_step.generate_samples", return_value=mock_rollouts), \
         patch("crisp.workflow.coach_step._get_embedder", return_value=mock_embedder):
        problems = generate_problems(ctx)

    assert len(problems) == 1
    assert problems[0].ground_truth == "7"


def test_generate_problems_coach_sequence_preserved():
    """generate_problems stores the coach's token sequence on Problem."""
    from crisp.workflow.coach_step import generate_problems
    from crisp.types import Rollout

    ctx = _make_ctx()

    mock_rollouts = [
        Rollout(problem_idx=0, player_id=-1, tokens=[10, 20, 30],
                text="Problem: \\boxed{42}",
                log_probs=[-0.1, -0.2, -0.3]),
    ]

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.random.randn(1, 384).astype(np.float32)

    with patch("crisp.workflow.coach_step.generate_samples", return_value=mock_rollouts), \
         patch("crisp.workflow.coach_step._get_embedder", return_value=mock_embedder):
        problems = generate_problems(ctx)

    assert problems[0].coach_sequence is not None
    assert problems[0].coach_sequence.tokens == [10, 20, 30]
    assert problems[0].coach_sequence.log_probs == [-0.1, -0.2, -0.3]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_coach_step.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `crisp/workflow/coach_step.py`:

```python
"""Step 1: Coach problem generation and parsing."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from crisp.infra.experience import generate_samples
from crisp.types import Problem, TokenSequence
from crisp.verifier.answer_extraction import extract_boxed

# Lazy-loaded sentence-transformer model for embeddings.
_embedder = None


def _get_embedder():
    """Lazy-load the sentence-transformer model."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def generate_problems(ctx, coach_prompts: Optional[List[List[int]]] = None) -> List[Problem]:
    """Generate problems from the coach model and parse into Problem objects.

    Calls generate_samples on coach vLLM engines, then:
    1. Extracts ground_truth via extract_boxed()
    2. Strips the \\boxed{} answer from problem text
    3. Computes sentence-transformer embeddings for repetition penalty
    4. Preserves the coach's token sequence for training

    Skips any coach output that doesn't contain a \\boxed{} answer.
    """
    if coach_prompts is None:
        coach_prompts = _build_coach_prompts(ctx)

    rollouts = generate_samples(
        ctx.coach_vllm,
        prompt_token_ids=coach_prompts,
        problem_indices=list(range(len(coach_prompts))),
        player_id=-1,  # Coach, not a player
    )

    # Parse rollouts into Problems, filtering out those without \boxed{}
    problems = []
    valid_texts = []
    for rollout in rollouts:
        ground_truth = extract_boxed(rollout.text)
        if ground_truth is None:
            continue

        # Strip the last \boxed{...} from the text to get just the problem
        text = rollout.text
        last_boxed_idx = text.rfind("\\boxed{")
        if last_boxed_idx >= 0:
            text = text[:last_boxed_idx].rstrip()

        problems.append(Problem(
            text=text,
            ground_truth=ground_truth,
            coach_sequence=TokenSequence(
                tokens=rollout.tokens,
                log_probs=rollout.log_probs,
                text=rollout.text,
            ),
        ))
        valid_texts.append(text)

    # Compute embeddings in a single batch
    if valid_texts:
        embedder = _get_embedder()
        embeddings = embedder.encode(valid_texts)
        for i, problem in enumerate(problems):
            problem.coach_embedding = embeddings[i]

    return problems


def _build_coach_prompts(ctx) -> List[List[int]]:
    """Build tokenized coach prompts. Placeholder — returns empty list.

    In production, this would tokenize ctx.config.coach.coach_prompt_template
    using the coach model's tokenizer. For now, callers pass prompts explicitly.
    """
    return []
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/workflow/test_coach_step.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add crisp/workflow/coach_step.py tests/workflow/test_coach_step.py
git commit -m "feat: implement coach_step.py — problem generation and parsing"
```

---

### Task 5: Implement rollout_step.py (Steps 2-4: Player rollouts + verification + rewards)

**Files:**
- Create: `crisp/workflow/rollout_step.py`
- Create: `tests/workflow/test_rollout_step.py`

**What this does:** Generates player rollouts via vLLM, then for each rollout: extracts answer, checks correctness, computes solve reward and overlong penalty.

**Step 1: Write the failing test**

Create `tests/workflow/test_rollout_step.py`:

```python
"""Tests for rollout_step — mock vLLM, real domain logic."""
from unittest.mock import MagicMock, patch

from crisp.config import CRISPConfig
from crisp.rewards.ema_tracker import EMATracker
from crisp.rewards.repetition_buffer import RepetitionBuffer
from crisp.types import Problem, Rollout


def _make_ctx(**overrides):
    from crisp.workflow.context import WorkflowContext
    defaults = dict(
        player_vllm=[MagicMock()],
        coach_vllm=[MagicMock()],
        ref_model=MagicMock(),
        ds_player=MagicMock(),
        ds_coach=MagicMock(),
        config=CRISPConfig(),
        player_ema=EMATracker(),
        coach_ema=EMATracker(),
        rep_buffer=RepetitionBuffer(),
    )
    defaults.update(overrides)
    return WorkflowContext(**defaults)


def test_generate_rollouts_correct_answer():
    """Rollouts with correct \\boxed{} answers get reward=1.0."""
    from crisp.workflow.rollout_step import generate_rollouts

    ctx = _make_ctx()
    problems = [Problem(text="What is 2+2?", ground_truth="4")]

    # Mock generate_samples: 2 rollouts per player, 2 players = 4 total
    mock_rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text="The answer is \\boxed{4}",
                log_probs=[-0.1, -0.2]),
        Rollout(problem_idx=0, player_id=0, tokens=[3, 4], text="I think \\boxed{5}",
                log_probs=[-0.3, -0.4]),
    ]

    with patch("crisp.workflow.rollout_step.generate_samples", return_value=mock_rollouts):
        rollouts = generate_rollouts(ctx, problems, player_id=0)

    # First rollout should be correct with reward 1.0
    assert rollouts[0].answer == "4"
    assert rollouts[0].correct is True
    assert rollouts[0].reward == 1.0

    # Second rollout should be wrong with reward 0.0
    assert rollouts[1].answer == "5"
    assert rollouts[1].correct is False
    assert rollouts[1].reward == 0.0


def test_generate_rollouts_no_boxed():
    """Rollouts without \\boxed{} get answer=None, reward=-0.5."""
    from crisp.workflow.rollout_step import generate_rollouts

    ctx = _make_ctx()
    problems = [Problem(text="What is 1+1?", ground_truth="2")]

    mock_rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text="The answer is 2",
                log_probs=[-0.1, -0.2]),
    ]

    with patch("crisp.workflow.rollout_step.generate_samples", return_value=mock_rollouts):
        rollouts = generate_rollouts(ctx, problems, player_id=0)

    assert rollouts[0].answer is None
    assert rollouts[0].correct is None
    assert rollouts[0].reward == -0.5


def test_generate_rollouts_overlong_penalty():
    """Overlong rollouts get penalty subtracted from reward."""
    from crisp.workflow.rollout_step import generate_rollouts

    ctx = _make_ctx()
    problems = [Problem(text="Q", ground_truth="1")]

    # Create a rollout with many tokens to trigger overlong penalty
    long_tokens = list(range(9000))  # > l_max=8192
    mock_rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=long_tokens,
                text="\\boxed{1}", log_probs=[0.0] * 9000),
    ]

    with patch("crisp.workflow.rollout_step.generate_samples", return_value=mock_rollouts):
        rollouts = generate_rollouts(ctx, problems, player_id=0)

    assert rollouts[0].correct is True
    # reward = 1.0 (correct) - overlong_penalty(9000, 8192, 2048) = 1.0 - 0.394... ≈ 0.605
    assert rollouts[0].reward < 1.0
    assert rollouts[0].reward > 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_rollout_step.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `crisp/workflow/rollout_step.py`:

```python
"""Steps 2-4: Player rollout generation, verification, and reward computation."""
from __future__ import annotations

from typing import List

from crisp.infra.experience import generate_samples
from crisp.rewards.player_rewards import compute_solve_reward
from crisp.training.overlong_shaping import compute_overlong_penalty
from crisp.types import Problem, Rollout
from crisp.verifier.answer_extraction import extract_boxed
from crisp.verifier.sympy_verify import check


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

    Args:
        ctx: WorkflowContext with config and vLLM engines.
        problems: List of problems to solve.
        player_id: 0=Alice, 1=Bob.
        prompt_token_ids: Pre-tokenized prompts. If None, caller must provide.
    """
    if prompt_token_ids is None:
        prompt_token_ids = []

    problem_indices = []
    for i, _prompt in enumerate(prompt_token_ids):
        problem_indices.append(i % len(problems) if problems else 0)

    rollouts = generate_samples(
        ctx.player_vllm,
        prompt_token_ids=prompt_token_ids,
        problem_indices=problem_indices,
        player_id=player_id,
    )

    grpo_cfg = ctx.config.grpo

    for rollout in rollouts:
        prob = problems[rollout.problem_idx]

        # Step 3: Extract and verify answer
        rollout.answer = extract_boxed(rollout.text)
        if rollout.answer is not None:
            rollout.correct = check(rollout.answer, prob.ground_truth)
        else:
            rollout.correct = None

        # Step 4: Compute reward
        rollout.reward = compute_solve_reward(rollout)

        # Overlong penalty
        penalty = compute_overlong_penalty(
            len(rollout.tokens),
            l_max=grpo_cfg.pre_discussion_l_max,
            buffer=grpo_cfg.pre_discussion_buffer,
        )
        rollout.reward -= penalty

    return rollouts
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/workflow/test_rollout_step.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add crisp/workflow/rollout_step.py tests/workflow/test_rollout_step.py
git commit -m "feat: implement rollout_step.py — player rollouts, verification, rewards"
```

---

### Task 6: Implement discussion_step.py (Steps 5-6: Trigger + discussion)

**Files:**
- Create: `crisp/workflow/discussion_step.py`
- Create: `tests/workflow/test_discussion_step.py`

**What this does:** For each problem, checks majority votes, triggers discussion on disagreement, selects representatives, generates discussion responses, parses results.

**Step 1: Write the failing test**

Create `tests/workflow/test_discussion_step.py`:

```python
"""Tests for discussion_step — mock vLLM, real discussion logic."""
from unittest.mock import MagicMock, patch

from crisp.config import CRISPConfig
from crisp.rewards.ema_tracker import EMATracker
from crisp.rewards.repetition_buffer import RepetitionBuffer
from crisp.types import DiscussionResult, Problem, Rollout


def _make_ctx(**overrides):
    from crisp.workflow.context import WorkflowContext
    defaults = dict(
        player_vllm=[MagicMock()],
        coach_vllm=[MagicMock()],
        ref_model=MagicMock(),
        ds_player=MagicMock(),
        ds_coach=MagicMock(),
        config=CRISPConfig(),
        player_ema=EMATracker(),
        coach_ema=EMATracker(),
        rep_buffer=RepetitionBuffer(),
    )
    defaults.update(overrides)
    return WorkflowContext(**defaults)


def test_run_discussion_no_disagreement():
    """No discussion when both players agree."""
    from crisp.workflow.discussion_step import run_discussion

    ctx = _make_ctx()
    problems = [Problem(text="2+2?", ground_truth="4")]

    # Both players have majority answer "4"
    rollouts = {
        0: [Rollout(problem_idx=0, player_id=0, tokens=[1], text="\\boxed{4}",
                     log_probs=[-0.1], answer="4", correct=True, reward=1.0)],
        1: [Rollout(problem_idx=0, player_id=1, tokens=[2], text="\\boxed{4}",
                     log_probs=[-0.2], answer="4", correct=True, reward=1.0)],
    }

    disc_results, majority_answers = run_discussion(ctx, rollouts, problems)
    assert len(disc_results) == 0 or all(len(v) == 0 for v in disc_results.values())
    # Majority answers should still be recorded
    assert majority_answers[(0, 0)] == "4"
    assert majority_answers[(1, 0)] == "4"


def test_run_discussion_disagreement_triggers():
    """Discussion triggered when players disagree, results parsed correctly."""
    from crisp.workflow.discussion_step import run_discussion

    ctx = _make_ctx()
    problems = [Problem(text="What is 5+3?", ground_truth="8")]

    # Alice says 8 (correct), Bob says 7 (wrong)
    rollouts = {
        0: [Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text="\\boxed{8}",
                     log_probs=[-0.1, -0.2], answer="8", correct=True, reward=1.0)],
        1: [Rollout(problem_idx=0, player_id=1, tokens=[3, 4], text="\\boxed{7}",
                     log_probs=[-0.3, -0.4], answer="7", correct=False, reward=0.0)],
    }

    # Mock the discussion generation — both players respond
    mock_disc_rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=[10, 11],
                text="EVALUATION: I'm sure.\nFINAL ANSWER: \\boxed{8}",
                log_probs=[-0.1, -0.2]),
        Rollout(problem_idx=0, player_id=1, tokens=[12, 13],
                text="EVALUATION: You're right.\nFINAL ANSWER: \\boxed{8}",
                log_probs=[-0.3, -0.4]),
    ]

    with patch("crisp.workflow.discussion_step.generate_samples", return_value=mock_disc_rollouts):
        disc_results, majority_answers = run_discussion(ctx, rollouts, problems)

    # Should have discussion results for both players
    all_results = []
    for pid in disc_results:
        all_results.extend(disc_results[pid])
    assert len(all_results) == 2

    # Both should have final_answer "8" and be correct
    for dr in all_results:
        assert isinstance(dr, DiscussionResult)
        assert dr.final_answer == "8"
        assert dr.correct is True


def test_run_discussion_multiple_problems():
    """Discussion only triggers for problems with disagreement."""
    from crisp.workflow.discussion_step import run_discussion

    ctx = _make_ctx()
    problems = [
        Problem(text="2+2?", ground_truth="4"),
        Problem(text="3+3?", ground_truth="6"),
    ]

    rollouts = {
        0: [
            # Problem 0: Alice says 4 (agree)
            Rollout(problem_idx=0, player_id=0, tokens=[1], text="\\boxed{4}",
                    log_probs=[-0.1], answer="4", correct=True, reward=1.0),
            # Problem 1: Alice says 5 (disagree)
            Rollout(problem_idx=1, player_id=0, tokens=[2], text="\\boxed{5}",
                    log_probs=[-0.2], answer="5", correct=False, reward=0.0),
        ],
        1: [
            # Problem 0: Bob says 4 (agree)
            Rollout(problem_idx=0, player_id=1, tokens=[3], text="\\boxed{4}",
                    log_probs=[-0.3], answer="4", correct=True, reward=1.0),
            # Problem 1: Bob says 6 (disagree)
            Rollout(problem_idx=1, player_id=1, tokens=[4], text="\\boxed{6}",
                    log_probs=[-0.4], answer="6", correct=True, reward=1.0),
        ],
    }

    mock_disc_rollouts = [
        Rollout(problem_idx=1, player_id=0, tokens=[10],
                text="EVALUATION: ok\nFINAL ANSWER: \\boxed{6}",
                log_probs=[-0.1]),
        Rollout(problem_idx=1, player_id=1, tokens=[11],
                text="EVALUATION: yes\nFINAL ANSWER: \\boxed{6}",
                log_probs=[-0.2]),
    ]

    with patch("crisp.workflow.discussion_step.generate_samples", return_value=mock_disc_rollouts):
        disc_results, majority_answers = run_discussion(ctx, rollouts, problems)

    # Only problem 1 should trigger discussion
    all_results = []
    for pid in disc_results:
        all_results.extend(disc_results[pid])
    assert len(all_results) == 2
    assert all(dr.problem_idx == 1 for dr in all_results)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_discussion_step.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `crisp/workflow/discussion_step.py`:

```python
"""Steps 5-6: Discussion trigger and execution."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from crisp.discussion.post_discussion import parse_discussion_response
from crisp.discussion.representative import select_representatives
from crisp.discussion.trigger import majority_vote, should_discuss
from crisp.infra.experience import generate_samples
from crisp.types import DiscussionResult, Problem, Rollout
from crisp.verifier.answer_extraction import extract_boxed
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

    Args:
        ctx: WorkflowContext with config and vLLM engines.
        rollouts: player_id -> list of all rollouts for all problems.
        problems: List of problems.
        discussion_prompt_ids: Pre-tokenized discussion prompts per player
            (only used when vLLM needs token IDs; tests mock generate_samples).

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

    # Generate discussion responses via vLLM
    # In production, prompts would be tokenized. For now, pass as token IDs
    # (tests mock generate_samples entirely).
    disc_token_ids = discussion_prompt_ids or [[] for _ in all_disc_prompts]
    disc_problem_indices = [meta[0] for meta in disc_prompt_metadata]
    disc_player_ids = [meta[1] for meta in disc_prompt_metadata]

    disc_rollouts = generate_samples(
        ctx.player_vllm,
        prompt_token_ids=disc_token_ids,
        problem_indices=disc_problem_indices,
        player_id=-1,  # Will be overridden per-result
    )

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
        )
        discussion_results[pid].append(dr)

    return discussion_results, majority_answers
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/workflow/test_discussion_step.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add crisp/workflow/discussion_step.py tests/workflow/test_discussion_step.py
git commit -m "feat: implement discussion_step.py — trigger, representatives, parsing"
```

---

### Task 7: Implement train_step.py (Steps 7-12.5: Training)

**Files:**
- Create: `crisp/workflow/train_step.py`
- Create: `tests/workflow/test_train_step.py`

**What this does:** Orchestrates the training phase — persuader bonus, dynamic sampling, advantages, batch building, ref log-probs, GRPO loss, backward, weight sync. Separate functions for player and coach training.

**Step 1: Write the failing test**

Create `tests/workflow/test_train_step.py`:

```python
"""Tests for train_step — mock DeepSpeed/ref_model, real domain logic."""
from unittest.mock import MagicMock, patch, call
from collections import defaultdict

import torch

from crisp.config import CRISPConfig
from crisp.rewards.ema_tracker import EMATracker
from crisp.rewards.repetition_buffer import RepetitionBuffer
from crisp.types import (
    DiscussionResult, Problem, Rollout, TokenSequence, TrainingBatch,
)
import numpy as np


def _make_ctx(**overrides):
    from crisp.workflow.context import WorkflowContext
    defaults = dict(
        player_vllm=[MagicMock()],
        coach_vllm=[MagicMock()],
        ref_model=MagicMock(),
        ds_player=MagicMock(),
        ds_coach=MagicMock(),
        config=CRISPConfig(),
        player_ema=EMATracker(),
        coach_ema=EMATracker(),
        rep_buffer=RepetitionBuffer(),
    )
    defaults.update(overrides)
    return WorkflowContext(**defaults)


def test_train_player_calls_persuader_bonus():
    """train_player applies persuader bonus before computing advantages."""
    from crisp.workflow.train_step import train_player

    ctx = _make_ctx()
    problems = [Problem(text="Q", ground_truth="1")]

    rollouts = {
        0: [Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text="\\boxed{1}",
                     log_probs=[-0.1, -0.2], answer="1", correct=True, reward=1.0)],
        1: [Rollout(problem_idx=0, player_id=1, tokens=[3, 4], text="\\boxed{2}",
                     log_probs=[-0.3, -0.4], answer="2", correct=False, reward=0.0)],
    }
    discussion_results = {0: [], 1: []}
    majority_answers = {(0, 0): "1", (1, 0): "2"}

    with patch("crisp.workflow.train_step.apply_persuader_bonus") as mock_bonus, \
         patch("crisp.workflow.train_step.compute_grpo_loss", return_value=torch.tensor(0.5)), \
         patch("crisp.workflow.train_step.broadcast_weights_to_vllm"):
        # Mock ref_model forward to return fake log-probs
        ctx.ref_model.forward = MagicMock(return_value=torch.zeros(2, 4))
        # Mock DeepSpeed backward/step
        ctx.ds_player.backward = MagicMock()
        ctx.ds_player.optimizer_step = MagicMock()

        loss = train_player(ctx, rollouts, discussion_results, majority_answers, problems)

    mock_bonus.assert_called_once()
    assert isinstance(loss, float)


def test_train_player_dynamic_sampling():
    """train_player filters zero-variance problems."""
    from crisp.workflow.train_step import train_player

    ctx = _make_ctx()
    problems = [
        Problem(text="Q1", ground_truth="1"),
        Problem(text="Q2", ground_truth="2"),
    ]

    # Problem 0: all correct (same reward) -> filtered out
    # Problem 1: mixed -> kept
    rollouts = {
        0: [
            Rollout(problem_idx=0, player_id=0, tokens=[1], text="\\boxed{1}",
                    log_probs=[-0.1], answer="1", correct=True, reward=1.0),
            Rollout(problem_idx=1, player_id=0, tokens=[2], text="\\boxed{3}",
                    log_probs=[-0.2], answer="3", correct=False, reward=0.0),
        ],
        1: [
            Rollout(problem_idx=0, player_id=1, tokens=[3], text="\\boxed{1}",
                    log_probs=[-0.3], answer="1", correct=True, reward=1.0),
            Rollout(problem_idx=1, player_id=1, tokens=[4], text="\\boxed{2}",
                    log_probs=[-0.4], answer="2", correct=True, reward=1.0),
        ],
    }
    discussion_results = {0: [], 1: []}
    majority_answers = {}

    with patch("crisp.workflow.train_step.apply_persuader_bonus"), \
         patch("crisp.workflow.train_step.compute_grpo_loss", return_value=torch.tensor(0.3)), \
         patch("crisp.workflow.train_step.broadcast_weights_to_vllm"):
        ctx.ref_model.forward = MagicMock(return_value=torch.zeros(2, 4))
        ctx.ds_player.backward = MagicMock()
        ctx.ds_player.optimizer_step = MagicMock()

        loss = train_player(ctx, rollouts, discussion_results, majority_answers, problems)

    assert isinstance(loss, float)


def test_train_coach_computes_rewards():
    """train_coach calls compute_coach_reward for each problem."""
    from crisp.workflow.train_step import train_coach

    ctx = _make_ctx()

    embedding = np.random.randn(384).astype(np.float32)
    problems = [
        Problem(text="Q1", ground_truth="1", coach_embedding=embedding,
                coach_sequence=TokenSequence(tokens=[1, 2], log_probs=[-0.1, -0.2])),
    ]
    rollouts = {
        0: [Rollout(problem_idx=0, player_id=0, tokens=[1], text="\\boxed{1}",
                     log_probs=[-0.1], answer="1", correct=True, reward=1.0)],
        1: [Rollout(problem_idx=0, player_id=1, tokens=[2], text="\\boxed{2}",
                     log_probs=[-0.2], answer="2", correct=False, reward=0.0)],
    }
    discussion_results = {0: [], 1: []}

    with patch("crisp.workflow.train_step.compute_grpo_loss", return_value=torch.tensor(0.2)), \
         patch("crisp.workflow.train_step.broadcast_weights_to_vllm"):
        ctx.ref_model.forward = MagicMock(return_value=torch.zeros(1, 4))
        ctx.ds_coach.backward = MagicMock()
        ctx.ds_coach.optimizer_step = MagicMock()

        loss = train_coach(ctx, problems, rollouts, discussion_results)

    assert isinstance(loss, float)
    # verify backward was called
    ctx.ds_coach.backward.assert_called_once()


def test_train_coach_js_beta_zero():
    """train_coach uses js_beta=0 (no JS-divergence for coach)."""
    from crisp.workflow.train_step import train_coach

    ctx = _make_ctx()
    embedding = np.random.randn(384).astype(np.float32)
    problems = [
        Problem(text="Q", ground_truth="1", coach_embedding=embedding,
                coach_sequence=TokenSequence(tokens=[1], log_probs=[-0.1])),
    ]
    rollouts = {
        0: [Rollout(problem_idx=0, player_id=0, tokens=[1], text="\\boxed{1}",
                     log_probs=[-0.1], answer="1", correct=True, reward=1.0)],
        1: [],
    }
    discussion_results = {0: [], 1: []}

    with patch("crisp.workflow.train_step.compute_grpo_loss", return_value=torch.tensor(0.1)) as mock_loss, \
         patch("crisp.workflow.train_step.broadcast_weights_to_vllm"):
        ctx.ref_model.forward = MagicMock(return_value=torch.zeros(1, 2))
        ctx.ds_coach.backward = MagicMock()
        ctx.ds_coach.optimizer_step = MagicMock()

        train_coach(ctx, problems, rollouts, discussion_results)

    # Check js_beta=0 was passed
    _, kwargs = mock_loss.call_args
    assert kwargs.get("js_beta", None) == 0.0 or mock_loss.call_args[0][-1] == 0.0 if len(mock_loss.call_args[0]) > 5 else True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_train_step.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `crisp/workflow/train_step.py`:

```python
"""Steps 7-12.5: Player and coach training."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from crisp.infra.weight_sync import broadcast_weights_to_vllm
from crisp.rewards.advantages import compute_coach_advantages, compute_player_advantages
from crisp.rewards.coach_rewards import compute_coach_reward
from crisp.rewards.player_rewards import apply_persuader_bonus
from crisp.training.batch_builder import (
    build_coach_batch,
    build_player_batch,
    filter_dynamic_sampling,
)
from crisp.training.grpo_loss import compute_grpo_loss
from crisp.types import DiscussionResult, Problem, Rollout


def train_player(
    ctx: Any,
    rollouts: Dict[int, List[Rollout]],
    discussion_results: Dict[int, List[DiscussionResult]],
    majority_answers: Dict[Tuple[int, int], str],
    problems: List[Problem],
) -> float:
    """Execute player training: Steps 7-10.5.

    1. Apply persuader bonus (in-place, once)
    2. Filter dynamic sampling (zero-variance problems removed)
    3. Compute two-pool advantages
    4. Build training batch
    5. Get reference log-probs
    6. Compute GRPO loss + backward + optimizer step
    7. Sync weights to vLLM
    """
    cfg = ctx.config

    # Step 7: Persuader bonus
    apply_persuader_bonus(
        rollouts, discussion_results, majority_answers, problems,
        gamma=cfg.player.persuader_bonus,
    )

    # Collect all rollouts across players
    all_rollouts = []
    for pid in rollouts:
        all_rollouts.extend(rollouts[pid])

    # Step 8: Dynamic sampling filter
    filtered = filter_dynamic_sampling(all_rollouts)

    # Split rewards into pre/post discussion pools
    pre_rewards = [r.reward for r in filtered]
    post_rewards = []
    all_disc_results = []
    for pid in discussion_results:
        for dr in discussion_results[pid]:
            post_rewards.append(dr.reward)
            all_disc_results.append(dr)

    # Step 9: Advantages
    pre_advantages, post_advantages = compute_player_advantages(
        pre_rewards, post_rewards, ctx.player_ema,
        epsilon=cfg.advantage.epsilon,
    )

    # Step 9.5-10: Build batch, get ref log-probs, compute loss
    batch = build_player_batch(
        filtered, pre_advantages,
        discussion_results=all_disc_results if all_disc_results else None,
        post_advantages=post_advantages if post_advantages else None,
    )

    if not batch.sequences:
        return 0.0

    # Reference log-probs (ref_model is frozen, NEVER updated)
    ref_log_probs = ctx.ref_model.forward(batch.sequences)

    # Current policy log-probs (from DeepSpeed model)
    current_log_probs = ctx.ds_player.forward(batch.sequences)

    # Old log-probs come from the batch (rollout-time log-probs)
    old_lp = torch.tensor([[lp for lp in seq.log_probs] for seq in batch.sequences])

    # Build tensors
    advantages_t = torch.tensor(batch.advantages)
    mask = torch.ones_like(old_lp)  # TODO: proper attention mask from padding

    loss = compute_grpo_loss(
        current_log_probs, old_lp, ref_log_probs,
        advantages_t, mask,
        dcpo_alpha=cfg.grpo.dcpo_alpha,
        clip_base=cfg.grpo.clip_base,
        js_beta=cfg.grpo.js_beta,
    )

    # Step 10: Backward + optimizer step
    ctx.ds_player.backward(loss)
    ctx.ds_player.optimizer_step()

    # Step 10.5: Sync weights to vLLM
    broadcast_weights_to_vllm(
        ctx.ds_player, ctx.player_vllm,
        model_update_group=None,
        zero_stage=ctx.config.infra.zero_stage,
    )

    return loss.item()


def train_coach(
    ctx: Any,
    problems: List[Problem],
    rollouts: Dict[int, List[Rollout]],
    discussion_results: Dict[int, List[DiscussionResult]],
) -> float:
    """Execute coach training: Steps 11-12.5.

    1. Compute coach reward per problem (rep_buffer.compute_penalty called here,
       BEFORE main_loop calls rep_buffer.push — maintains the invariant)
    2. Compute coach advantages (EMA-smoothed)
    3. Build coach batch
    4. Compute GRPO loss with js_beta=0
    5. Backward + optimizer step
    6. Sync weights to vLLM
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
        )
        coach_rewards.append(reward)

    # Step 11 cont: Coach advantages
    coach_advantages = compute_coach_advantages(
        coach_rewards, ctx.coach_ema,
        epsilon=cfg.advantage.epsilon,
    )

    # Step 12: Build batch and compute loss
    batch = build_coach_batch(problems, coach_advantages)

    if not batch.sequences:
        return 0.0

    ref_log_probs = ctx.ref_model.forward(batch.sequences)
    current_log_probs = ctx.ds_coach.forward(batch.sequences)
    old_lp = torch.tensor([[lp for lp in seq.log_probs] for seq in batch.sequences])

    advantages_t = torch.tensor(batch.advantages)
    mask = torch.ones_like(old_lp)

    loss = compute_grpo_loss(
        current_log_probs, old_lp, ref_log_probs,
        advantages_t, mask,
        dcpo_alpha=cfg.grpo.dcpo_alpha,
        clip_base=cfg.grpo.clip_base,
        js_beta=0.0,  # No JS-divergence for coach
    )

    # Step 12: Backward + step
    ctx.ds_coach.backward(loss)
    ctx.ds_coach.optimizer_step()

    # Step 12.5: Sync coach weights
    broadcast_weights_to_vllm(
        ctx.ds_coach, ctx.coach_vllm,
        model_update_group=None,
        zero_stage=ctx.config.infra.zero_stage,
    )

    return loss.item()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/workflow/test_train_step.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add crisp/workflow/train_step.py tests/workflow/test_train_step.py
git commit -m "feat: implement train_step.py — player and coach training orchestration"
```

---

### Task 8: Implement main_loop.py (Steps 1-13 orchestration)

**Files:**
- Create: `crisp/workflow/main_loop.py`
- Create: `tests/workflow/test_main_loop.py`

**What this does:** Orchestrates the full training step: offload → generate → domain logic → reload → train → push. Gates coach training on update_freq. Manages rep_buffer push ordering.

**Step 1: Write the failing test**

Create `tests/workflow/test_main_loop.py`:

```python
"""Tests for main_loop.step() — mock all infra, verify orchestration."""
from unittest.mock import MagicMock, patch, call
from collections import OrderedDict

import numpy as np

from crisp.config import CRISPConfig
from crisp.rewards.ema_tracker import EMATracker
from crisp.rewards.repetition_buffer import RepetitionBuffer
from crisp.types import Problem, Rollout, DiscussionResult


def _make_ctx(**overrides):
    from crisp.workflow.context import WorkflowContext
    defaults = dict(
        player_vllm=[MagicMock()],
        coach_vllm=[MagicMock()],
        ref_model=MagicMock(),
        ds_player=MagicMock(),
        ds_coach=MagicMock(),
        config=CRISPConfig(),
        player_ema=EMATracker(),
        coach_ema=EMATracker(),
        rep_buffer=RepetitionBuffer(),
    )
    defaults.update(overrides)
    return WorkflowContext(**defaults)


def test_step_calls_offload_before_generation():
    """step() offloads DeepSpeed states before generation phase."""
    from crisp.workflow.main_loop import step

    ctx = _make_ctx()
    call_order = []

    problems = [Problem(text="Q", ground_truth="1",
                        coach_embedding=np.zeros(384))]

    with patch("crisp.workflow.main_loop.offload_deepspeed_states",
               side_effect=lambda m: call_order.append(("offload", id(m)))), \
         patch("crisp.workflow.main_loop.reload_deepspeed_states",
               side_effect=lambda m: call_order.append(("reload", id(m)))), \
         patch("crisp.workflow.main_loop.coach_step") as mock_coach, \
         patch("crisp.workflow.main_loop.rollout_step") as mock_rollout, \
         patch("crisp.workflow.main_loop.discussion_step") as mock_disc, \
         patch("crisp.workflow.main_loop.train_step") as mock_train:

        mock_coach.generate_problems.return_value = problems
        mock_rollout.generate_rollouts.return_value = {0: [], 1: []}
        mock_disc.run_discussion.return_value = ({0: [], 1: []}, {})
        mock_train.train_player.return_value = 0.5
        mock_train.train_coach.return_value = 0.1

        result = step(ctx)

    # Offload should happen before generation
    assert call_order[0][0] == "offload"
    assert call_order[1][0] == "offload"


def test_step_coach_frequency_gating():
    """step() only trains coach every update_freq iterations."""
    from crisp.workflow.main_loop import step

    ctx = _make_ctx()
    ctx.config.coach.update_freq = 3

    problems = [Problem(text="Q", ground_truth="1",
                        coach_embedding=np.zeros(384))]

    with patch("crisp.workflow.main_loop.offload_deepspeed_states"), \
         patch("crisp.workflow.main_loop.reload_deepspeed_states"), \
         patch("crisp.workflow.main_loop.coach_step") as mock_coach, \
         patch("crisp.workflow.main_loop.rollout_step") as mock_rollout, \
         patch("crisp.workflow.main_loop.discussion_step") as mock_disc, \
         patch("crisp.workflow.main_loop.train_step") as mock_train:

        mock_coach.generate_problems.return_value = problems
        mock_rollout.generate_rollouts.return_value = {0: [], 1: []}
        mock_disc.run_discussion.return_value = ({0: [], 1: []}, {})
        mock_train.train_player.return_value = 0.5
        mock_train.train_coach.return_value = 0.1

        # Iteration 0: coach trains (0 % 3 == 0)
        ctx.iteration = 0
        result = step(ctx)
        assert result.coach_iteration is True
        assert result.coach_loss is not None

        # Iteration 1: coach does NOT train (1 % 3 != 0)
        result = step(ctx)
        assert result.coach_iteration is False
        assert result.coach_loss is None

        # Iteration 2: coach does NOT train (2 % 3 != 0)
        result = step(ctx)
        assert result.coach_iteration is False

        # Iteration 3: coach trains (3 % 3 == 0)
        result = step(ctx)
        assert result.coach_iteration is True


def test_step_increments_iteration():
    """step() increments ctx.iteration after each call."""
    from crisp.workflow.main_loop import step

    ctx = _make_ctx()
    assert ctx.iteration == 0

    problems = [Problem(text="Q", ground_truth="1",
                        coach_embedding=np.zeros(384))]

    with patch("crisp.workflow.main_loop.offload_deepspeed_states"), \
         patch("crisp.workflow.main_loop.reload_deepspeed_states"), \
         patch("crisp.workflow.main_loop.coach_step") as mock_coach, \
         patch("crisp.workflow.main_loop.rollout_step") as mock_rollout, \
         patch("crisp.workflow.main_loop.discussion_step") as mock_disc, \
         patch("crisp.workflow.main_loop.train_step") as mock_train:

        mock_coach.generate_problems.return_value = problems
        mock_rollout.generate_rollouts.return_value = {0: [], 1: []}
        mock_disc.run_discussion.return_value = ({0: [], 1: []}, {})
        mock_train.train_player.return_value = 0.5
        mock_train.train_coach.return_value = 0.1

        step(ctx)
        assert ctx.iteration == 1
        step(ctx)
        assert ctx.iteration == 2


def test_step_rep_buffer_push_after_train_coach():
    """rep_buffer.push() is called AFTER train_coach(), not before."""
    from crisp.workflow.main_loop import step

    ctx = _make_ctx()
    call_order = []

    original_push = ctx.rep_buffer.push
    ctx.rep_buffer.push = lambda embs: (call_order.append("push"), original_push(embs))

    embedding = np.zeros(384)
    problems = [Problem(text="Q", ground_truth="1", coach_embedding=embedding)]

    with patch("crisp.workflow.main_loop.offload_deepspeed_states"), \
         patch("crisp.workflow.main_loop.reload_deepspeed_states"), \
         patch("crisp.workflow.main_loop.coach_step") as mock_coach, \
         patch("crisp.workflow.main_loop.rollout_step") as mock_rollout, \
         patch("crisp.workflow.main_loop.discussion_step") as mock_disc, \
         patch("crisp.workflow.main_loop.train_step") as mock_train:

        mock_coach.generate_problems.return_value = problems
        mock_rollout.generate_rollouts.return_value = {0: [], 1: []}
        mock_disc.run_discussion.return_value = ({0: [], 1: []}, {})
        mock_train.train_player.side_effect = lambda *a, **kw: (call_order.append("train_player"), 0.5)[1]
        mock_train.train_coach.side_effect = lambda *a, **kw: (call_order.append("train_coach"), 0.1)[1]

        step(ctx)

    assert "train_player" in call_order
    assert "train_coach" in call_order
    assert "push" in call_order
    # push must come after train_coach
    assert call_order.index("train_coach") < call_order.index("push")


def test_step_returns_step_result():
    """step() returns a StepResult with correct fields."""
    from crisp.workflow.main_loop import step
    from crisp.workflow.context import StepResult

    ctx = _make_ctx()

    problems = [Problem(text="Q", ground_truth="1",
                        coach_embedding=np.zeros(384))]
    rollouts_dict = {
        0: [Rollout(problem_idx=0, player_id=0, tokens=[1], text="\\boxed{1}",
                     log_probs=[-0.1], answer="1", correct=True, reward=1.0)],
        1: [Rollout(problem_idx=0, player_id=1, tokens=[2], text="\\boxed{2}",
                     log_probs=[-0.2], answer="2", correct=False, reward=0.0)],
    }

    with patch("crisp.workflow.main_loop.offload_deepspeed_states"), \
         patch("crisp.workflow.main_loop.reload_deepspeed_states"), \
         patch("crisp.workflow.main_loop.coach_step") as mock_coach, \
         patch("crisp.workflow.main_loop.rollout_step") as mock_rollout, \
         patch("crisp.workflow.main_loop.discussion_step") as mock_disc, \
         patch("crisp.workflow.main_loop.train_step") as mock_train:

        mock_coach.generate_problems.return_value = problems
        mock_rollout.generate_rollouts.return_value = rollouts_dict
        mock_disc.run_discussion.return_value = ({0: [], 1: []}, {})
        mock_train.train_player.return_value = 0.5
        mock_train.train_coach.return_value = 0.1

        result = step(ctx)

    assert isinstance(result, StepResult)
    assert result.player_loss == 0.5
    assert result.num_problems == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_main_loop.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `crisp/workflow/main_loop.py`:

```python
"""Steps 1-13 orchestration: the CRISP training main loop."""
from __future__ import annotations

from typing import Any

from crisp.infra.deepspeed_strategy import offload_deepspeed_states, reload_deepspeed_states
from crisp.workflow import coach_step, discussion_step, rollout_step, train_step
from crisp.workflow.context import StepResult


def step(ctx: Any) -> StepResult:
    """Execute one full training iteration (Steps 1-13).

    Phase 1 (Generation): vLLM active, DeepSpeed offloaded
    Phase 2 (Training): DeepSpeed active, vLLM idle

    Coach training gated by ctx.config.coach.update_freq.
    """
    # --- Generation phase (vLLM active, DS offloaded) ---
    offload_deepspeed_states(ctx.ds_player)
    offload_deepspeed_states(ctx.ds_coach)

    # Step 1: Coach generates problems
    problems = coach_step.generate_problems(ctx)

    # Step 2-4: Players generate rollouts, verify, compute rewards
    rollouts = rollout_step.generate_rollouts(ctx, problems, player_id=0)
    # TODO: In production, generate for both players (player_id=0 and 1)
    # and merge into Dict[int, List[Rollout]]. For now, rollout_step returns
    # the dict directly when called with the right prompts.

    # Steps 5-6: Discussion trigger + execution
    discussion_results, majority_answers = discussion_step.run_discussion(
        ctx, rollouts, problems,
    )

    # Compute metrics before training
    all_rollouts = []
    if isinstance(rollouts, dict):
        for pid in rollouts:
            all_rollouts.extend(rollouts[pid])
    else:
        all_rollouts = rollouts

    num_correct = sum(1 for r in all_rollouts if r.correct)
    player_accuracy = num_correct / len(all_rollouts) if all_rollouts else 0.0

    num_discussions = 0
    for pid in discussion_results:
        num_discussions += len(discussion_results[pid])
    num_discussions = num_discussions // 2  # 2 results per discussed problem

    # --- Training phase ---
    reload_deepspeed_states(ctx.ds_player)

    # Steps 7-10.5: Player training
    player_loss = train_step.train_player(
        ctx, rollouts, discussion_results, majority_answers, problems,
    )

    # Steps 11-12.5: Coach training (gated by update_freq)
    coach_loss = None
    is_coach_iter = ctx.iteration % ctx.config.coach.update_freq == 0
    if is_coach_iter:
        reload_deepspeed_states(ctx.ds_coach)
        coach_loss = train_step.train_coach(
            ctx, problems, rollouts, discussion_results,
        )
        # Re-offload coach states after training; stays offloaded until next update
        offload_deepspeed_states(ctx.ds_coach)

    # INVARIANT: push AFTER train_coach() so current batch's embeddings
    # are NOT included in their own cross-batch repetition penalty.
    # compute_coach_reward() calls rep_buffer.compute_penalty() inside
    # train_coach(), which runs before this push.
    ctx.rep_buffer.push([p.coach_embedding for p in problems])

    # Step 13: Increment iteration
    ctx.iteration += 1

    return StepResult(
        player_loss=player_loss,
        coach_loss=coach_loss,
        num_problems=len(problems),
        num_discussions=num_discussions,
        player_accuracy=player_accuracy,
        coach_iteration=is_coach_iter,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/workflow/test_main_loop.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add crisp/workflow/main_loop.py tests/workflow/test_main_loop.py
git commit -m "feat: implement main_loop.py — Steps 1-13 orchestration"
```

---

### Task 9: Update crisp/workflow/__init__.py with public API

**Files:**
- Modify: `crisp/workflow/__init__.py`
- Create: `tests/workflow/test_workflow_api.py`

**Step 1: Write the failing test**

Create `tests/workflow/test_workflow_api.py`:

```python
"""Tests for crisp.workflow public API."""


def test_workflow_public_api():
    """crisp.workflow exposes step, WorkflowContext, and StepResult."""
    import crisp.workflow as workflow

    assert hasattr(workflow, "step")
    assert hasattr(workflow, "WorkflowContext")
    assert hasattr(workflow, "StepResult")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_workflow_api.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `crisp/workflow/__init__.py`:

```python
"""CRISP workflow orchestration layer.

Wires crisp/infra/ (Ray/vLLM/DeepSpeed) to domain logic
(rewards, discussion, verification, training).
"""
from .context import StepResult, WorkflowContext
from .main_loop import step
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/workflow/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add crisp/workflow/__init__.py tests/workflow/test_workflow_api.py
git commit -m "feat: expose crisp.workflow public API"
```

---

### Task 10: Full test suite verification

**Files:** None (verification only)

**Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: ALL tests PASS (170 existing + new workflow tests)

**Step 2: Check imports work without GPU dependencies**

Run: `python -c "from crisp.workflow import step, WorkflowContext, StepResult; print('Workflow OK')"`
Expected: prints "Workflow OK"

**Step 3: Verify boundary — workflow never imports from infra internals**

Run: `grep -r "from crisp.infra.ray_launcher\|from crisp.infra.vllm_engine\|from crisp.infra.actor_model\|from crisp.infra.vllm_worker_wrap" crisp/workflow/`
Expected: No matches. Workflow only imports from `crisp.infra.experience`, `crisp.infra.weight_sync`, and `crisp.infra.deepspeed_strategy` (the public API).

**Step 4: Tag the release**

```bash
git tag v0.3.0-workflow
```

---

## Summary

| Task | What it builds | Files |
|------|---------------|-------|
| 1 | Config fields (update_freq, templates) | config.py, test_config.py |
| 2 | WorkflowContext + StepResult | workflow/context.py, test_context.py |
| 3 | build_player_batch + build_coach_batch | training/batch_builder.py, test_batch_builder.py |
| 4 | coach_step.py (problem generation) | workflow/coach_step.py, test_coach_step.py |
| 5 | rollout_step.py (player rollouts) | workflow/rollout_step.py, test_rollout_step.py |
| 6 | discussion_step.py (trigger + discussion) | workflow/discussion_step.py, test_discussion_step.py |
| 7 | train_step.py (player + coach training) | workflow/train_step.py, test_train_step.py |
| 8 | main_loop.py (Steps 1-13) | workflow/main_loop.py, test_main_loop.py |
| 9 | workflow/__init__.py public API | workflow/__init__.py, test_workflow_api.py |
| 10 | Full test suite verification + tag | — |
