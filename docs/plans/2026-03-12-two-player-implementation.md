# Two-Player Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the single `ds_player` into `ds_alice` / `ds_bob` with independent training, independent EMA trackers, sequential rollout generation with weight sync, and per-player discussion generation — restoring the SystemWalkthrough's two-player design.

**Architecture:** Two independent DeepSpeed engines on GPU 2 (both 4B, ~27GB each), shared vLLM engine on GPU 0 with sequential weight sync before each player's rollouts. Shared frozen ref model on CPU. Independent EMA trackers per player (Dr. MAS). Persuader bonus computed in main_loop before per-player training split.

**Tech Stack:** DeepSpeed ZeRO-2, vLLM, Ray, PyTorch, pytest

---

### Task 1: Update WorkflowContext dataclass

**Files:**
- Modify: `crisp/workflow/context.py:13-32`
- Test: `tests/workflow/test_context.py`

**Step 1: Update the test to use the new field names**

Replace the entire contents of `tests/workflow/test_context.py`:

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
        ds_alice=MagicMock(),
        ds_bob=MagicMock(),
        ds_coach=MagicMock(),
        config=CRISPConfig(),
        alice_ema=EMATracker(),
        bob_ema=EMATracker(),
        coach_ema=EMATracker(),
        rep_buffer=RepetitionBuffer(),
    )
    assert ctx.iteration == 0
    assert isinstance(ctx.config, CRISPConfig)
    assert isinstance(ctx.alice_ema, EMATracker)
    assert isinstance(ctx.bob_ema, EMATracker)


def test_workflow_context_iteration_mutable():
    """iteration field is mutable."""
    from crisp.workflow.context import WorkflowContext

    ctx = WorkflowContext(
        player_vllm=[], coach_vllm=[], ref_model=None,
        ds_alice=None, ds_bob=None, ds_coach=None, config=CRISPConfig(),
        alice_ema=EMATracker(), bob_ema=EMATracker(), coach_ema=EMATracker(),
        rep_buffer=RepetitionBuffer(),
    )
    ctx.iteration = 5
    assert ctx.iteration == 5


def test_step_result_creation():
    """StepResult holds step output metrics."""
    from crisp.workflow.context import StepResult

    result = StepResult(
        alice_loss=0.5,
        bob_loss=0.4,
        coach_loss=None,
        num_problems=8,
        num_discussions=3,
        player_accuracy=0.75,
        coach_iteration=False,
    )
    assert result.alice_loss == 0.5
    assert result.bob_loss == 0.4
    assert result.coach_loss is None
    assert result.coach_iteration is False


def test_step_result_coach_iteration():
    """StepResult with coach training."""
    from crisp.workflow.context import StepResult

    result = StepResult(
        alice_loss=0.3,
        bob_loss=0.2,
        coach_loss=0.1,
        num_problems=8,
        num_discussions=2,
        player_accuracy=0.6,
        coach_iteration=True,
    )
    assert result.coach_loss == 0.1
    assert result.coach_iteration is True
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/workflow/test_context.py -v`
Expected: FAIL — `WorkflowContext` still expects `ds_player` and `player_ema`

**Step 3: Update WorkflowContext and StepResult**

Replace `crisp/workflow/context.py`:

```python
"""WorkflowContext and StepResult for CRISP workflow orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from crisp.config import CRISPConfig
from crisp.rewards.ema_tracker import EMATracker
from crisp.rewards.repetition_buffer import RepetitionBuffer


@dataclass
class WorkflowContext:
    """Holds all infra handles and stateful objects for the training loop.

    Passed to each step function for dependency injection.
    Two independent player strategies (Alice, Bob) with independent EMA trackers
    implement the Dr. MAS principle: each agent lives in its own reward universe.
    """
    player_vllm: List[Any]       # shared vLLM engines (same base model)
    coach_vllm: Optional[List[Any]]  # vLLM engine actors (None → HF generate)
    ref_model: Any               # Frozen reference policy (NEVER updated, shared)
    ds_alice: Any                # DeepSpeed strategy for Alice (player 0)
    ds_bob: Any                  # DeepSpeed strategy for Bob (player 1)
    ds_coach: Any                # DeepSpeed strategy for coach
    config: CRISPConfig
    # Independent per-player EMA trackers (Dr. MAS: separate reward universes)
    alice_ema: EMATracker
    bob_ema: EMATracker
    coach_ema: EMATracker
    rep_buffer: RepetitionBuffer
    iteration: int = 0
    pad_token_id: int = 0
    tokenizer: Any = None
    coach_tokenizer: Any = None  # separate tokenizer when coach model differs
    accuracy_history: List[float] = field(default_factory=list)


@dataclass
class StepResult:
    """Output metrics from a single training step."""
    alice_loss: float
    bob_loss: float
    coach_loss: Optional[float]  # None on non-coach-update iterations
    num_problems: int
    num_discussions: int
    player_accuracy: float       # fraction correct pre-discussion (both players)
    coach_iteration: bool        # whether coach was updated this step
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/workflow/test_context.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add crisp/workflow/context.py tests/workflow/test_context.py
git commit -m "refactor: split WorkflowContext into ds_alice/ds_bob with independent EMA"
```

---

### Task 2: Update train_step.py — per-player train_player signature

**Files:**
- Modify: `crisp/workflow/train_step.py:141-266`
- Test: `tests/workflow/test_train_step.py`

**Step 1: Write updated tests**

Replace the entire contents of `tests/workflow/test_train_step.py`:

```python
"""Tests for train_step — mock DeepSpeed/ref_model, real domain logic."""
from unittest.mock import MagicMock, patch, call

import torch
import numpy as np

from crisp.config import CRISPConfig
from crisp.rewards.ema_tracker import EMATracker
from crisp.rewards.repetition_buffer import RepetitionBuffer
from crisp.types import (
    DiscussionResult, Problem, Rollout, TokenSequence, TrainingBatch,
)


def _make_ctx(**overrides):
    from crisp.workflow.context import WorkflowContext
    ds_alice = MagicMock()
    ds_alice._engine = None
    ds_bob = MagicMock()
    ds_bob._engine = None
    ds_coach = MagicMock()
    ds_coach._engine = None
    defaults = dict(
        player_vllm=[MagicMock()],
        coach_vllm=[MagicMock()],
        ref_model=MagicMock(),
        ds_alice=ds_alice,
        ds_bob=ds_bob,
        ds_coach=ds_coach,
        config=CRISPConfig(),
        alice_ema=EMATracker(),
        bob_ema=EMATracker(),
        coach_ema=EMATracker(),
        rep_buffer=RepetitionBuffer(),
    )
    defaults.update(overrides)
    return WorkflowContext(**defaults)


def test_train_player_independent_alice():
    """train_player trains Alice with only Alice's rollouts and Alice's EMA."""
    from crisp.workflow.train_step import train_player

    ctx = _make_ctx()
    problems = [Problem(text="Q", ground_truth="1")]

    # Alice's rollouts only — mixed rewards so dynamic sampling keeps them
    alice_rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text="\\boxed{1}",
                log_probs=[-0.1, -0.2], answer="1", correct=True, reward=1.0,
                _persuader_bonus_applied=True),
        Rollout(problem_idx=0, player_id=0, tokens=[3, 4], text="\\boxed{2}",
                log_probs=[-0.3, -0.4], answer="2", correct=False, reward=0.0,
                _persuader_bonus_applied=True),
    ]
    alice_discussions = []

    with patch("crisp.workflow.train_step.compute_grpo_loss", return_value=torch.tensor(0.5)):
        ctx.ref_model.forward = MagicMock(return_value=torch.zeros(2, 4))
        ctx.ds_alice.forward = MagicMock(return_value=torch.zeros(2, 4))
        ctx.ds_alice.backward = MagicMock()
        ctx.ds_alice.optimizer_step = MagicMock()

        loss = train_player(
            ctx, player_id=0,
            rollouts=alice_rollouts,
            discussion_results=alice_discussions,
            problems=problems,
            ds_model=ctx.ds_alice,
            ema_tracker=ctx.alice_ema,
        )

    assert isinstance(loss, float)
    ctx.ds_alice.optimizer_step.assert_called_once()
    # Bob's model should NOT be touched
    ctx.ds_bob.forward.assert_not_called()
    ctx.ds_bob.backward.assert_not_called()


def test_train_player_independent_bob():
    """train_player trains Bob with only Bob's rollouts and Bob's EMA."""
    from crisp.workflow.train_step import train_player

    ctx = _make_ctx()
    problems = [Problem(text="Q", ground_truth="1")]

    bob_rollouts = [
        Rollout(problem_idx=0, player_id=1, tokens=[1, 2], text="\\boxed{1}",
                log_probs=[-0.1, -0.2], answer="1", correct=True, reward=1.0,
                _persuader_bonus_applied=True),
        Rollout(problem_idx=0, player_id=1, tokens=[3, 4], text="\\boxed{2}",
                log_probs=[-0.3, -0.4], answer="2", correct=False, reward=0.0,
                _persuader_bonus_applied=True),
    ]
    bob_discussions = []

    with patch("crisp.workflow.train_step.compute_grpo_loss", return_value=torch.tensor(0.3)):
        ctx.ref_model.forward = MagicMock(return_value=torch.zeros(2, 4))
        ctx.ds_bob.forward = MagicMock(return_value=torch.zeros(2, 4))
        ctx.ds_bob.backward = MagicMock()
        ctx.ds_bob.optimizer_step = MagicMock()

        loss = train_player(
            ctx, player_id=1,
            rollouts=bob_rollouts,
            discussion_results=bob_discussions,
            problems=problems,
            ds_model=ctx.ds_bob,
            ema_tracker=ctx.bob_ema,
        )

    assert isinstance(loss, float)
    ctx.ds_bob.optimizer_step.assert_called_once()
    ctx.ds_alice.forward.assert_not_called()


def test_train_player_dynamic_sampling():
    """train_player filters zero-variance problems per player."""
    from crisp.workflow.train_step import train_player

    ctx = _make_ctx()
    problems = [
        Problem(text="Q1", ground_truth="1"),
        Problem(text="Q2", ground_truth="2"),
    ]

    # Problem 0: all correct (same reward) -> filtered out
    # Problem 1: mixed -> kept
    alice_rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=[1], text="\\boxed{1}",
                log_probs=[-0.1], answer="1", correct=True, reward=1.0,
                _persuader_bonus_applied=True),
        Rollout(problem_idx=0, player_id=0, tokens=[2], text="\\boxed{1}",
                log_probs=[-0.2], answer="1", correct=True, reward=1.0,
                _persuader_bonus_applied=True),
        Rollout(problem_idx=1, player_id=0, tokens=[3], text="\\boxed{3}",
                log_probs=[-0.3], answer="3", correct=False, reward=0.0,
                _persuader_bonus_applied=True),
        Rollout(problem_idx=1, player_id=0, tokens=[4], text="\\boxed{2}",
                log_probs=[-0.4], answer="2", correct=True, reward=1.0,
                _persuader_bonus_applied=True),
    ]

    with patch("crisp.workflow.train_step.compute_grpo_loss", return_value=torch.tensor(0.3)):
        ctx.ref_model.forward = MagicMock(return_value=torch.zeros(2, 4))
        ctx.ds_alice.forward = MagicMock(return_value=torch.zeros(2, 4))
        ctx.ds_alice.backward = MagicMock()
        ctx.ds_alice.optimizer_step = MagicMock()

        loss = train_player(
            ctx, player_id=0,
            rollouts=alice_rollouts,
            discussion_results=[],
            problems=problems,
            ds_model=ctx.ds_alice,
            ema_tracker=ctx.alice_ema,
        )

    assert isinstance(loss, float)


def test_train_player_no_persuader_bonus_call():
    """train_player does NOT call apply_persuader_bonus (moved to main_loop)."""
    from crisp.workflow.train_step import train_player

    ctx = _make_ctx()
    problems = [Problem(text="Q", ground_truth="1")]

    rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text="\\boxed{1}",
                log_probs=[-0.1, -0.2], answer="1", correct=True, reward=1.0,
                _persuader_bonus_applied=True),
        Rollout(problem_idx=0, player_id=0, tokens=[3, 4], text="\\boxed{2}",
                log_probs=[-0.3, -0.4], answer="2", correct=False, reward=0.0,
                _persuader_bonus_applied=True),
    ]

    with patch("crisp.workflow.train_step.apply_persuader_bonus") as mock_bonus, \
         patch("crisp.workflow.train_step.compute_grpo_loss", return_value=torch.tensor(0.5)):
        ctx.ref_model.forward = MagicMock(return_value=torch.zeros(2, 4))
        ctx.ds_alice.forward = MagicMock(return_value=torch.zeros(2, 4))
        ctx.ds_alice.backward = MagicMock()
        ctx.ds_alice.optimizer_step = MagicMock()

        loss = train_player(
            ctx, player_id=0,
            rollouts=rollouts,
            discussion_results=[],
            problems=problems,
            ds_model=ctx.ds_alice,
            ema_tracker=ctx.alice_ema,
        )

    # Persuader bonus should NOT be called inside train_player
    mock_bonus.assert_not_called()


def test_train_player_passes_tensors():
    """train_player passes torch.Tensor to forward(), not List[TokenSequence]."""
    from crisp.workflow.train_step import train_player

    ctx = _make_ctx()
    ctx.pad_token_id = 0

    problems = [Problem(text="Q", ground_truth="1")]
    rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text="\\boxed{1}",
                log_probs=[-0.1, -0.2], answer="1", correct=True, reward=1.0,
                _persuader_bonus_applied=True),
        Rollout(problem_idx=0, player_id=0, tokens=[3, 4], text="\\boxed{2}",
                log_probs=[-0.3, -0.4], answer="2", correct=False, reward=0.0,
                _persuader_bonus_applied=True),
    ]

    with patch("crisp.workflow.train_step.compute_grpo_loss", return_value=torch.tensor(0.5)):
        ctx.ref_model.forward = MagicMock(return_value=torch.zeros(2, 2))
        ctx.ds_alice.forward = MagicMock(return_value=torch.zeros(2, 2))
        ctx.ds_alice.backward = MagicMock()
        ctx.ds_alice.optimizer_step = MagicMock()

        train_player(
            ctx, player_id=0,
            rollouts=rollouts,
            discussion_results=[],
            problems=problems,
            ds_model=ctx.ds_alice,
            ema_tracker=ctx.alice_ema,
        )

    call_args = ctx.ds_alice.forward.call_args
    assert call_args is not None, "ds_alice.forward was never called"
    assert isinstance(call_args[0][0], torch.Tensor)


def test_train_coach_computes_rewards():
    """train_coach calls compute_coach_reward for each problem."""
    from crisp.workflow.train_step import train_coach

    ctx = _make_ctx()

    embedding1 = np.random.randn(384).astype(np.float32)
    embedding2 = np.random.randn(384).astype(np.float32)
    problems = [
        Problem(text="Q1", ground_truth="1", coach_embedding=embedding1,
                coach_sequence=TokenSequence(tokens=[1, 2], log_probs=[-0.1, -0.2])),
        Problem(text="Q2", ground_truth="2", coach_embedding=embedding2,
                coach_sequence=TokenSequence(tokens=[3, 4], log_probs=[-0.3, -0.4])),
    ]
    rollouts = {
        0: [Rollout(problem_idx=0, player_id=0, tokens=[1], text="\\boxed{1}",
                     log_probs=[-0.1], answer="1", correct=True, reward=1.0),
             Rollout(problem_idx=1, player_id=0, tokens=[5], text="\\boxed{3}",
                     log_probs=[-0.5], answer="3", correct=False, reward=0.0)],
        1: [Rollout(problem_idx=0, player_id=1, tokens=[2], text="\\boxed{2}",
                     log_probs=[-0.2], answer="2", correct=False, reward=0.0),
             Rollout(problem_idx=1, player_id=1, tokens=[6], text="\\boxed{4}",
                     log_probs=[-0.6], answer="4", correct=False, reward=0.0)],
    }
    discussion_results = {0: [], 1: []}

    with patch("crisp.workflow.train_step.compute_grpo_loss", return_value=torch.tensor(0.2)):
        ctx.ref_model.forward = MagicMock(return_value=torch.zeros(2, 4))
        ctx.ds_coach.backward = MagicMock()
        ctx.ds_coach.optimizer_step = MagicMock()

        loss, coach_rewards = train_coach(ctx, problems, rollouts, discussion_results)

    assert isinstance(loss, float)
    assert isinstance(coach_rewards, list)
    assert len(coach_rewards) == 2
    ctx.ds_coach.backward.assert_called_once()
    ctx.ref_model.forward.assert_not_called()


def test_train_coach_uses_config_js_beta():
    """train_coach uses js_beta from config (KL constraint prevents mode collapse)."""
    from crisp.workflow.train_step import train_coach

    ctx = _make_ctx()
    embedding1 = np.random.randn(384).astype(np.float32)
    embedding2 = np.random.randn(384).astype(np.float32)
    problems = [
        Problem(text="Q1", ground_truth="1", coach_embedding=embedding1,
                coach_sequence=TokenSequence(tokens=[1], log_probs=[-0.1])),
        Problem(text="Q2", ground_truth="2", coach_embedding=embedding2,
                coach_sequence=TokenSequence(tokens=[2], log_probs=[-0.2])),
    ]
    rollouts = {
        0: [Rollout(problem_idx=0, player_id=0, tokens=[1], text="\\boxed{1}",
                     log_probs=[-0.1], answer="1", correct=True, reward=1.0),
             Rollout(problem_idx=1, player_id=0, tokens=[3], text="\\boxed{3}",
                     log_probs=[-0.3], answer="3", correct=False, reward=0.0)],
        1: [Rollout(problem_idx=1, player_id=1, tokens=[4], text="\\boxed{4}",
                     log_probs=[-0.4], answer="4", correct=False, reward=0.0)],
    }
    discussion_results = {0: [], 1: []}

    with patch("crisp.workflow.train_step.compute_grpo_loss", return_value=torch.tensor(0.1)) as mock_loss:
        ctx.ref_model.forward = MagicMock(return_value=torch.zeros(2, 2))
        ctx.ds_coach.backward = MagicMock()
        ctx.ds_coach.optimizer_step = MagicMock()

        loss, coach_rewards = train_coach(ctx, problems, rollouts, discussion_results)

    _, kwargs = mock_loss.call_args
    assert kwargs.get("js_beta") == ctx.config.grpo.js_beta
    ctx.ref_model.forward.assert_not_called()


def test_train_coach_skips_on_zero_variance_rewards():
    """train_coach returns early when all coach rewards are identical (no signal)."""
    from crisp.workflow.train_step import train_coach

    ctx = _make_ctx()
    embedding = np.random.randn(384).astype(np.float32)
    problems = [
        Problem(text="Q", ground_truth="1", coach_embedding=embedding,
                coach_sequence=TokenSequence(tokens=[1, 2], log_probs=[-0.1, -0.2])),
    ]
    rollouts = {
        0: [Rollout(problem_idx=0, player_id=0, tokens=[1], text="\\boxed{1}",
                     log_probs=[-0.1], answer="1", correct=True, reward=1.0)],
        1: [Rollout(problem_idx=0, player_id=1, tokens=[2], text="\\boxed{2}",
                     log_probs=[-0.2], answer="2", correct=False, reward=0.0)],
    }
    discussion_results = {0: [], 1: []}

    ctx.ds_coach.backward = MagicMock()
    ctx.ds_coach.optimizer_step = MagicMock()

    loss, coach_rewards = train_coach(ctx, problems, rollouts, discussion_results)

    assert loss == 0.0
    assert isinstance(coach_rewards, list)
    assert len(coach_rewards) == 1
    ctx.ds_coach.backward.assert_not_called()
    ctx.ds_coach.optimizer_step.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/workflow/test_train_step.py -v`
Expected: FAIL — `train_player` has wrong signature, `WorkflowContext` has wrong fields

**Step 3: Update train_player in train_step.py**

Change the `train_player` function signature and body. Remove the `apply_persuader_bonus` call, accept flat lists instead of dicts, accept explicit `ds_model` and `ema_tracker` params.

In `crisp/workflow/train_step.py`, replace the `train_player` function (lines 141-266) with:

```python
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
    """Execute player training: Steps 7-10.5 for a SINGLE player.

    Persuader bonus must be applied BEFORE calling this function (in main_loop).
    Each player has independent dynamic sampling, advantages, and gradient updates.

    1. Filter dynamic sampling (zero-variance problems removed)
    2. Compute two-pool advantages (player's own EMA)
    3. Build training batch
    4. Get reference log-probs (shared ref model)
    5. Compute GRPO loss + backward + optimizer step
    6. Sync weights to vLLM (unless sync_weights=False)
    """
    cfg = ctx.config

    # Step 8: Dynamic sampling filter (per-player)
    filtered = filter_dynamic_sampling(rollouts)

    # Split rewards into pre/post discussion pools
    pre_rewards = [r.reward for r in filtered]
    post_rewards = [dr.reward for dr in discussion_results]

    # Step 9: Advantages (player's own EMA — Dr. MAS independent stats)
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

        _alloc = torch.cuda.memory_allocated(device) / 1e9
        _mem_log.info(
            "player%d split %d/%d: %d seqs, max_len=%d, after pad_to_gpu: %.1fGB",
            player_id, split_idx + 1, n_splits, end - start,
            input_ids.shape[1], _alloc,
        )

        ref_log_probs = _chunked_ref_forward(ctx.ref_model, input_ids, attention_mask)
        torch.cuda.empty_cache()

        _alloc = torch.cuda.memory_allocated(device) / 1e9
        _mem_log.info("player%d split %d/%d: after ref_forward+empty_cache: %.1fGB",
                      player_id, split_idx + 1, n_splits, _alloc)

        old_lp = old_lp[:, 1:]
        mask = attention_mask[:, 1:].float()
        advantages_t = torch.tensor(sub_advs, device=device)

        split_loss = _chunked_grpo_backward(
            ds_model, ref_log_probs,
            input_ids, attention_mask, old_lp, advantages_t, mask,
            dcpo_alpha=cfg.grpo.dcpo_alpha,
            clip_base=cfg.grpo.clip_base,
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
```

Also remove the `apply_persuader_bonus` import from the top of the file (line 13) since `train_player` no longer calls it. Keep the import available at module level for other callers if needed — actually, remove it entirely from `train_step.py` since `main_loop.py` will import it directly.

In `crisp/workflow/train_step.py` line 13, remove:
```python
from crisp.rewards.player_rewards import apply_persuader_bonus
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/workflow/test_train_step.py -v`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add crisp/workflow/train_step.py crisp/workflow/context.py tests/workflow/test_train_step.py
git commit -m "refactor: train_player takes per-player rollouts, ds_model, ema_tracker"
```

---

### Task 3: Update discussion_step.py — per-player discussion generation

**Files:**
- Modify: `crisp/workflow/discussion_step.py:98-114`
- Test: `tests/workflow/test_discussion_step.py`

**Step 1: Write a new test for per-player discussion generation**

Add to `tests/workflow/test_discussion_step.py`:

```python
def test_run_discussion_syncs_weights_per_player():
    """Discussion generation syncs correct player weights before generating."""
    from crisp.workflow.discussion_step import run_discussion

    ctx = _make_ctx()
    # Add ds_alice/ds_bob with sync_weights method
    ctx.ds_alice = MagicMock()
    ctx.ds_bob = MagicMock()

    problems = [Problem(text="What is 5+3?", ground_truth="8")]

    rollouts = {
        0: [Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text="\\boxed{8}",
                     log_probs=[-0.1, -0.2], answer="8", correct=True, reward=1.0)],
        1: [Rollout(problem_idx=0, player_id=1, tokens=[3, 4], text="\\boxed{7}",
                     log_probs=[-0.3, -0.4], answer="7", correct=False, reward=0.0)],
    }

    mock_disc_rollouts_alice = [
        Rollout(problem_idx=0, player_id=0, tokens=[10, 11],
                text="FINAL ANSWER: \\boxed{8}",
                log_probs=[-0.1, -0.2]),
    ]
    mock_disc_rollouts_bob = [
        Rollout(problem_idx=0, player_id=1, tokens=[12, 13],
                text="FINAL ANSWER: \\boxed{8}",
                log_probs=[-0.3, -0.4]),
    ]

    call_order = []
    def mock_generate(engines, **kwargs):
        call_order.append("generate")
        # Return based on call count
        if len([c for c in call_order if c == "generate"]) == 1:
            return mock_disc_rollouts_alice
        return mock_disc_rollouts_bob

    ctx.ds_alice.sync_weights = MagicMock(side_effect=lambda v: call_order.append("sync_alice"))
    ctx.ds_bob.sync_weights = MagicMock(side_effect=lambda v: call_order.append("sync_bob"))

    with patch("crisp.workflow.discussion_step.generate_samples", side_effect=mock_generate):
        disc_results, majority_answers = run_discussion(ctx, rollouts, problems)

    # Verify sync happened before each generate
    assert "sync_alice" in call_order
    assert "sync_bob" in call_order
    all_results = []
    for pid in disc_results:
        all_results.extend(disc_results[pid])
    assert len(all_results) == 2
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_discussion_step.py::test_run_discussion_syncs_weights_per_player -v`
Expected: FAIL — `ctx` doesn't have `ds_alice`/`ds_bob`, discussion doesn't sync weights

**Step 3: Update discussion_step.py**

The key change: instead of batching all discussion prompts into one `generate_samples` call, split by player and sync weights before each.

In `crisp/workflow/discussion_step.py`, replace the generation section (lines 95-141) with logic that groups prompts by player_id, syncs weights, and generates per-player:

```python
    if not all_disc_prompts:
        return discussion_results, majority_answers

    # Group prompts by player_id for per-player weight sync
    alice_prompts = []
    alice_metadata = []
    bob_prompts = []
    bob_metadata = []
    for prompt, (prob_idx, pid) in zip(all_disc_prompts, disc_prompt_metadata):
        if pid == 0:
            alice_prompts.append(prompt)
            alice_metadata.append((prob_idx, pid))
        else:
            bob_prompts.append(prompt)
            bob_metadata.append((prob_idx, pid))

    def _generate_for_player(prompts, metadata, ds_model):
        """Sync weights and generate discussion responses for one player."""
        if not prompts:
            return []

        # Sync this player's weights to vLLM before generating
        if ds_model is not None:
            ds_model.sync_weights(ctx.player_vllm)

        if discussion_prompt_ids is not None:
            disc_token_ids = discussion_prompt_ids
        elif getattr(ctx, 'tokenizer', None) is not None:
            from crisp.workflow.tokenizer import build_discussion_prompts
            disc_token_ids = build_discussion_prompts(ctx.tokenizer, ctx.config, prompts)
        else:
            disc_token_ids = [[] for _ in prompts]
        disc_problem_indices = [meta[0] for meta in metadata]

        return generate_samples(
            ctx.player_vllm,
            prompt_token_ids=disc_token_ids,
            problem_indices=disc_problem_indices,
            player_id=-1,
            max_new_tokens=2048,
        )

    # Get per-player ds_model (None for legacy single-player contexts)
    ds_alice = getattr(ctx, 'ds_alice', None)
    ds_bob = getattr(ctx, 'ds_bob', None)

    alice_rollouts_disc = _generate_for_player(alice_prompts, alice_metadata, ds_alice)
    bob_rollouts_disc = _generate_for_player(bob_prompts, bob_metadata, ds_bob)

    # Combine and parse results
    all_gen_rollouts = list(zip(alice_rollouts_disc, alice_metadata)) + \
                       list(zip(bob_rollouts_disc, bob_metadata))

    for rollout, (prob_idx, pid) in all_gen_rollouts:
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
            reward=1.0 if correct else 0.0,
        )
        grpo_cfg = ctx.config.grpo
        penalty = compute_overlong_penalty(
            len(rollout.tokens),
            l_max=grpo_cfg.post_discussion_l_max,
            buffer=grpo_cfg.post_discussion_buffer,
        )
        dr.reward -= penalty

        discussion_results[pid].append(dr)

    return discussion_results, majority_answers
```

**Step 4: Update _make_ctx in test_discussion_step.py to use new context fields**

Update the `_make_ctx` helper at the top of `tests/workflow/test_discussion_step.py`:

```python
def _make_ctx(**overrides):
    from crisp.workflow.context import WorkflowContext
    defaults = dict(
        player_vllm=[MagicMock()],
        coach_vllm=[MagicMock()],
        ref_model=MagicMock(),
        ds_alice=MagicMock(),
        ds_bob=MagicMock(),
        ds_coach=MagicMock(),
        config=CRISPConfig(),
        alice_ema=EMATracker(),
        bob_ema=EMATracker(),
        coach_ema=EMATracker(),
        rep_buffer=RepetitionBuffer(),
    )
    defaults.update(overrides)
    return WorkflowContext(**defaults)
```

**Step 5: Run all discussion tests**

Run: `python -m pytest tests/workflow/test_discussion_step.py -v`
Expected: PASS (5 tests)

**Step 6: Commit**

```bash
git add crisp/workflow/discussion_step.py tests/workflow/test_discussion_step.py
git commit -m "feat: per-player weight sync in discussion generation"
```

---

### Task 4: Update main_loop.py — orchestrate two-player flow

**Files:**
- Modify: `crisp/workflow/main_loop.py`
- Test: `tests/workflow/test_main_loop.py`

**Step 1: Write updated tests**

Replace the entire contents of `tests/workflow/test_main_loop.py`:

```python
"""Tests for main_loop.step() — mock all infra, verify orchestration."""
from unittest.mock import MagicMock, patch, call

import numpy as np

from crisp.config import CRISPConfig
from crisp.rewards.ema_tracker import EMATracker
from crisp.rewards.repetition_buffer import RepetitionBuffer
from crisp.types import Problem, Rollout


def _make_ctx(**overrides):
    from crisp.workflow.context import WorkflowContext
    config = CRISPConfig()
    config.infra.vllm_enable_sleep = False
    player_vllm = [MagicMock()]
    ds_alice = MagicMock()
    ds_alice._engine = None
    ds_bob = MagicMock()
    ds_bob._engine = None
    defaults = dict(
        player_vllm=player_vllm,
        coach_vllm=None,
        ref_model=MagicMock(),
        ds_alice=ds_alice,
        ds_bob=ds_bob,
        ds_coach=MagicMock(),
        config=config,
        alice_ema=EMATracker(),
        bob_ema=EMATracker(),
        coach_ema=EMATracker(),
        rep_buffer=RepetitionBuffer(),
    )
    defaults.update(overrides)
    return WorkflowContext(**defaults)


def test_step_trains_both_players_independently():
    """step() calls train_player twice — once for Alice, once for Bob."""
    from crisp.workflow.main_loop import step

    ctx = _make_ctx()
    problems = [Problem(text="Q", ground_truth="1",
                        coach_embedding=np.zeros(384))]

    with patch("crisp.workflow.main_loop.coach_step") as mock_coach, \
         patch("crisp.workflow.main_loop.rollout_step") as mock_rollout, \
         patch("crisp.workflow.main_loop.discussion_step") as mock_disc, \
         patch("crisp.workflow.main_loop.train_step") as mock_train, \
         patch("crisp.workflow.main_loop.apply_persuader_bonus"):

        mock_coach.generate_problems.return_value = problems
        mock_rollout.generate_rollouts.return_value = []
        mock_disc.run_discussion.return_value = ({0: [], 1: []}, {})
        mock_train.train_player.return_value = 0.5
        mock_train.train_coach.return_value = (0.1, [0.5])

        result = step(ctx)

    # train_player called twice (Alice and Bob)
    assert mock_train.train_player.call_count == 2

    # First call is Alice (player_id=0), second is Bob (player_id=1)
    alice_call = mock_train.train_player.call_args_list[0]
    bob_call = mock_train.train_player.call_args_list[1]
    assert alice_call.kwargs.get("player_id") == 0 or alice_call.args[1] == 0
    assert bob_call.kwargs.get("player_id") == 1 or bob_call.args[1] == 1


def test_step_sequential_rollouts_with_weight_sync():
    """step() syncs weights before each player's rollout generation."""
    from crisp.workflow.main_loop import step

    ctx = _make_ctx()
    call_order = []

    problems = [Problem(text="Q", ground_truth="1",
                        coach_embedding=np.zeros(384))]

    ctx.ds_alice.sync_weights = MagicMock(
        side_effect=lambda v: call_order.append("sync_alice"))
    ctx.ds_bob.sync_weights = MagicMock(
        side_effect=lambda v: call_order.append("sync_bob"))

    with patch("crisp.workflow.main_loop.coach_step") as mock_coach, \
         patch("crisp.workflow.main_loop.rollout_step") as mock_rollout, \
         patch("crisp.workflow.main_loop.discussion_step") as mock_disc, \
         patch("crisp.workflow.main_loop.train_step") as mock_train, \
         patch("crisp.workflow.main_loop.apply_persuader_bonus"):

        mock_coach.generate_problems.return_value = problems

        def mock_gen_rollouts(ctx, problems, player_id, **kw):
            call_order.append(f"rollout_{player_id}")
            return []
        mock_rollout.generate_rollouts.side_effect = mock_gen_rollouts
        mock_disc.run_discussion.return_value = ({0: [], 1: []}, {})
        mock_train.train_player.return_value = 0.5
        mock_train.train_coach.return_value = (0.1, [0.5])

        step(ctx)

    # Verify: sync_alice -> rollout_0 -> sync_bob -> rollout_1
    assert call_order.index("sync_alice") < call_order.index("rollout_0")
    assert call_order.index("sync_bob") < call_order.index("rollout_1")
    assert call_order.index("rollout_0") < call_order.index("sync_bob")


def test_step_persuader_bonus_before_training():
    """step() calls apply_persuader_bonus once before per-player training."""
    from crisp.workflow.main_loop import step

    ctx = _make_ctx()
    call_order = []

    problems = [Problem(text="Q", ground_truth="1",
                        coach_embedding=np.zeros(384))]

    with patch("crisp.workflow.main_loop.coach_step") as mock_coach, \
         patch("crisp.workflow.main_loop.rollout_step") as mock_rollout, \
         patch("crisp.workflow.main_loop.discussion_step") as mock_disc, \
         patch("crisp.workflow.main_loop.train_step") as mock_train, \
         patch("crisp.workflow.main_loop.apply_persuader_bonus") as mock_bonus:

        mock_coach.generate_problems.return_value = problems
        mock_rollout.generate_rollouts.return_value = []
        mock_disc.run_discussion.return_value = ({0: [], 1: []}, {})
        mock_bonus.side_effect = lambda *a, **kw: call_order.append("bonus")
        mock_train.train_player.side_effect = lambda *a, **kw: (
            call_order.append("train"), 0.5)[1]
        mock_train.train_coach.return_value = (0.1, [0.5])

        step(ctx)

    mock_bonus.assert_called_once()
    assert call_order.index("bonus") < call_order.index("train")


def test_step_returns_both_player_losses():
    """step() returns StepResult with alice_loss and bob_loss."""
    from crisp.workflow.main_loop import step
    from crisp.workflow.context import StepResult

    ctx = _make_ctx()
    problems = [Problem(text="Q", ground_truth="1",
                        coach_embedding=np.zeros(384))]

    with patch("crisp.workflow.main_loop.coach_step") as mock_coach, \
         patch("crisp.workflow.main_loop.rollout_step") as mock_rollout, \
         patch("crisp.workflow.main_loop.discussion_step") as mock_disc, \
         patch("crisp.workflow.main_loop.train_step") as mock_train, \
         patch("crisp.workflow.main_loop.apply_persuader_bonus"):

        mock_coach.generate_problems.return_value = problems
        mock_rollout.generate_rollouts.return_value = []
        mock_disc.run_discussion.return_value = ({0: [], 1: []}, {})
        # Return different losses for Alice and Bob
        mock_train.train_player.side_effect = [0.5, 0.3]
        mock_train.train_coach.return_value = (0.1, [0.5])

        result = step(ctx)

    assert isinstance(result, StepResult)
    assert result.alice_loss == 0.5
    assert result.bob_loss == 0.3


def test_step_coach_frequency_gating():
    """step() only trains coach every update_freq iterations."""
    from crisp.workflow.main_loop import step

    ctx = _make_ctx()
    ctx.config.coach.update_freq = 3

    problems = [Problem(text="Q", ground_truth="1",
                        coach_embedding=np.zeros(384))]

    with patch("crisp.workflow.main_loop.coach_step") as mock_coach, \
         patch("crisp.workflow.main_loop.rollout_step") as mock_rollout, \
         patch("crisp.workflow.main_loop.discussion_step") as mock_disc, \
         patch("crisp.workflow.main_loop.train_step") as mock_train, \
         patch("crisp.workflow.main_loop.apply_persuader_bonus"):

        mock_coach.generate_problems.return_value = problems
        mock_rollout.generate_rollouts.return_value = []
        mock_disc.run_discussion.return_value = ({0: [], 1: []}, {})
        mock_train.train_player.return_value = 0.5
        mock_train.train_coach.return_value = (0.1, [0.5])

        ctx.iteration = 0
        result = step(ctx)
        assert result.coach_iteration is True

        result = step(ctx)
        assert result.coach_iteration is False

        result = step(ctx)
        assert result.coach_iteration is False

        result = step(ctx)
        assert result.coach_iteration is True


def test_step_increments_iteration():
    """step() increments ctx.iteration after each call."""
    from crisp.workflow.main_loop import step

    ctx = _make_ctx()
    assert ctx.iteration == 0

    problems = [Problem(text="Q", ground_truth="1",
                        coach_embedding=np.zeros(384))]

    with patch("crisp.workflow.main_loop.coach_step") as mock_coach, \
         patch("crisp.workflow.main_loop.rollout_step") as mock_rollout, \
         patch("crisp.workflow.main_loop.discussion_step") as mock_disc, \
         patch("crisp.workflow.main_loop.train_step") as mock_train, \
         patch("crisp.workflow.main_loop.apply_persuader_bonus"):

        mock_coach.generate_problems.return_value = problems
        mock_rollout.generate_rollouts.return_value = []
        mock_disc.run_discussion.return_value = ({0: [], 1: []}, {})
        mock_train.train_player.return_value = 0.5
        mock_train.train_coach.return_value = (0.1, [0.5])

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

    with patch("crisp.workflow.main_loop.coach_step") as mock_coach, \
         patch("crisp.workflow.main_loop.rollout_step") as mock_rollout, \
         patch("crisp.workflow.main_loop.discussion_step") as mock_disc, \
         patch("crisp.workflow.main_loop.train_step") as mock_train, \
         patch("crisp.workflow.main_loop.apply_persuader_bonus"):

        mock_coach.generate_problems.return_value = problems
        mock_rollout.generate_rollouts.return_value = []
        mock_disc.run_discussion.return_value = ({0: [], 1: []}, {})
        mock_train.train_player.side_effect = lambda *a, **kw: (call_order.append("train_player"), 0.5)[1]
        mock_train.train_coach.side_effect = lambda *a, **kw: (call_order.append("train_coach"), (0.1, [0.5]))[1]

        step(ctx)

    assert call_order.index("train_coach") < call_order.index("push")
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/workflow/test_main_loop.py -v`
Expected: FAIL — `main_loop.step()` still uses old single-player pattern

**Step 3: Rewrite main_loop.py**

Replace `crisp/workflow/main_loop.py` with the two-player orchestration. Key changes:
- Import `apply_persuader_bonus` directly
- Sequential rollout generation with per-player weight sync
- Call `train_player` twice (Alice, Bob) with per-player args
- Return `StepResult` with `alice_loss` and `bob_loss`
- Coach weight sync on dedicated GPU 1 (unchanged)

```python
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

    Two-player design: Alice and Bob are independent agents with separate
    DeepSpeed models, separate EMA trackers, and separate training steps.
    vLLM engine is shared — weight sync happens before each player's rollouts.

    Args:
        ctx: WorkflowContext with all infra handles.
        collector: Optional StepCollector to capture intermediate data.
    """
    # === GENERATION PHASE ===
    _shared_gpu = (ctx.config.infra.vllm_enable_sleep
                   and ctx.coach_vllm is not None
                   and ctx.coach_vllm is not ctx.player_vllm)

    # Step 1: Coach generates problems
    if _shared_gpu:
        _wake_vllm(ctx.coach_vllm)

    problems = coach_step.generate_problems(
        ctx, accuracy_history=ctx.accuracy_history,
    )

    if _shared_gpu:
        _sleep_vllm(ctx.coach_vllm)
        _wake_vllm(ctx.player_vllm)

    if not problems:
        logger.warning("iter=%d: Coach produced 0 valid problems, skipping iteration",
                       ctx.iteration)
        if _shared_gpu:
            _sleep_vllm(ctx.player_vllm)
        ctx.iteration += 1
        return StepResult(
            alice_loss=0.0, bob_loss=0.0, coach_loss=None, num_problems=0,
            num_discussions=0, player_accuracy=0.0, coach_iteration=False,
        )

    solvable_problems = [p for p in problems if p.self_solvable]
    n_unsolvable = len(problems) - len(solvable_problems)
    if n_unsolvable:
        logger.info("iter=%d: %d/%d problems unsolvable by coach (will penalize)",
                     ctx.iteration, n_unsolvable, len(problems))

    for i, p in enumerate(solvable_problems):
        logger.info("iter=%d problem[%d] (answer=%s): %s", ctx.iteration, i,
                     p.ground_truth, p.text[:200].replace("\n", " "))

    # Steps 2-4: Sequential per-player rollouts with weight sync
    if solvable_problems:
        # Sync Alice weights → generate Alice rollouts
        ctx.ds_alice.sync_weights(ctx.player_vllm)
        alice_rollouts = rollout_step.generate_rollouts(
            ctx, solvable_problems, player_id=0)

        # Sync Bob weights → generate Bob rollouts
        ctx.ds_bob.sync_weights(ctx.player_vllm)
        bob_rollouts = rollout_step.generate_rollouts(
            ctx, solvable_problems, player_id=1)

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

    # Steps 5-6: Discussion (per-player weight sync happens inside discussion_step)
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
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    if _shared_gpu:
        _sleep_vllm(ctx.player_vllm)
    elif ctx.config.infra.vllm_enable_sleep:
        all_engines = list(ctx.player_vllm)
        if ctx.coach_vllm is not None and ctx.coach_vllm is not ctx.player_vllm:
            all_engines.extend(ctx.coach_vllm)
        _sleep_vllm(all_engines)

    _log_gpu_memory("before player training")

    # Step 7: Persuader bonus (needs both players' data — call once before split)
    apply_persuader_bonus(
        rollouts, discussion_results, majority_answers, solvable_problems,
        gamma=ctx.config.player.persuader_bonus,
    )

    # Steps 7-10.5: Per-player training (Dr. MAS: independent reward universes)
    alice_loss = train_step.train_player(
        ctx, player_id=0,
        rollouts=rollouts[0],
        discussion_results=discussion_results[0],
        problems=solvable_problems,
        ds_model=ctx.ds_alice,
        ema_tracker=ctx.alice_ema,
        sync_weights=False,
    )

    torch.cuda.empty_cache()

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
    torch.cuda.empty_cache()

    if _shared_gpu:
        # Shared-GPU: wake, sync each player, sleep
        _wake_vllm(ctx.player_vllm)
        ctx.ds_alice.sync_weights(ctx.player_vllm)
        # Bob sync will happen at start of next iteration before Bob's rollouts
        _sleep_vllm(ctx.player_vllm)

        if is_coach_iter:
            _wake_vllm(ctx.coach_vllm)
            ctx.ds_coach.sync_weights(ctx.coach_vllm)
            _sleep_vllm(ctx.coach_vllm)
    elif ctx.config.infra.vllm_enable_sleep:
        _log_gpu_memory("after training")
        torch.cuda.empty_cache()
        _log_gpu_memory("after empty_cache")
        _wake_vllm(all_engines)
        # Weight sync deferred to generation phase (sync before each player's rollouts)
        if ctx.coach_vllm is not None:
            ctx.ds_coach.sync_weights(ctx.coach_vllm)
    else:
        # No sleep mode (dedicated GPUs) — coach sync only
        # Player weight sync deferred to generation phase
        if is_coach_iter and ctx.coach_vllm is not None:
            ctx.ds_coach.sync_weights(ctx.coach_vllm)

    # INVARIANT: push AFTER train_coach() so current batch's embeddings
    # are NOT included in their own cross-batch repetition penalty.
    ctx.rep_buffer.push([p.coach_embedding for p in problems])

    # Track accuracy for coach calibration prompt
    ctx.accuracy_history.append(player_accuracy)

    # Step 13: Increment iteration
    ctx.iteration += 1

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
            player_loss=(alice_loss + bob_loss) / 2,
            coach_loss=coach_loss,
            coach_rewards=coach_rewards,
            result=result,
        ))

    return result
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/workflow/test_main_loop.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add crisp/workflow/main_loop.py tests/workflow/test_main_loop.py
git commit -m "feat: two-player orchestration with per-player weight sync and training"
```

---

### Task 5: Update init_infra in train.py

**Files:**
- Modify: `crisp/train.py:60-400` (init_infra), `crisp/train.py:403-431` (save_checkpoint), `crisp/train.py:434-470` (load_checkpoint), `crisp/train.py:473-502` (save_hf_weights), `crisp/train.py:531-646` (run)

This task has no dedicated test file (init_infra requires GPU). Verify by running the full test suite.

**Step 1: Update init_infra — 4-GPU path (lines 228-275)**

Replace the 4-GPU block with two player models:

```python
    if icfg.num_gpus_per_node == 4:
        from crisp.infra.actor_model import Actor

        # GPU 2: Alice
        ds_alice = _make_strategy()
        ds_alice.setup_distributed(dist_backend="gloo")
        alice_model = Actor(tcfg.model_name, **actor_kwargs)
        ds_alice.prepare(alice_model)
        alice_model.gradient_checkpointing_enable()
        _inner = getattr(alice_model.model, 'model', alice_model.model)
        logger.info("Alice model on GPU 2: gc=%s, training=%s",
                     getattr(_inner, 'gradient_checkpointing', False), _inner.training)

        # GPU 2: Bob (same GPU, independent weights + optimizer)
        ds_bob = _make_strategy()
        bob_model = Actor(tcfg.model_name, **actor_kwargs)
        ds_bob.prepare(bob_model)
        bob_model.gradient_checkpointing_enable()
        _inner_b = getattr(bob_model.model, 'model', bob_model.model)
        logger.info("Bob model on GPU 2: gc=%s, training=%s",
                     getattr(_inner_b, 'gradient_checkpointing', False), _inner_b.training)

        # GPU 2: Reference model (parked on CPU, shared by both players)
        ref_strategy = _make_strategy()
        ref_model_actor = Actor(tcfg.model_name, **actor_kwargs)
        ref_strategy.prepare(ref_model_actor, is_rlhf=True)
        ref_model = ref_strategy
        ref_module = getattr(ref_strategy, '_engine', ref_strategy)
        ref_module = getattr(ref_module, 'module', ref_module)
        ref_module.to('cpu')
        _torch.cuda.empty_cache()
        logger.info("Reference model initialized and parked on CPU")

        # GPU 3: Coach model
        os.environ["LOCAL_RANK"] = "3"
        _torch.cuda.set_device(3)
        ds_coach = _make_strategy(learning_rate=coach_lr)
        coach_model_actor = Actor(coach_model, **actor_kwargs)
        ds_coach.prepare(coach_model_actor)
        coach_model_actor.gradient_checkpointing_enable()
        _inner_c = getattr(coach_model_actor.model, 'model', coach_model_actor.model)
        logger.info("Coach model on GPU 3: gc=%s, training=%s",
                     getattr(_inner_c, 'gradient_checkpointing', False), _inner_c.training)

        # Reset default device to GPU 2
        os.environ["LOCAL_RANK"] = "2"
        _torch.cuda.set_device(2)
```

**Step 2: Update single/2-GPU path (lines 328-362)**

Same pattern — create `ds_alice` and `ds_bob` instead of `ds_player`:

```python
    else:
        from crisp.infra.actor_model import Actor

        ds_alice = _make_strategy()
        ds_alice.setup_distributed()
        alice_model = Actor(tcfg.model_name, **actor_kwargs)
        ds_alice.prepare(alice_model)
        alice_model.gradient_checkpointing_enable()
        logger.info("Alice model initialized (gradient checkpointing ON)")

        ds_bob = _make_strategy()
        bob_model = Actor(tcfg.model_name, **actor_kwargs)
        ds_bob.prepare(bob_model)
        bob_model.gradient_checkpointing_enable()
        logger.info("Bob model initialized (gradient checkpointing ON)")

        ds_coach = _make_strategy(learning_rate=coach_lr)
        coach_model_actor = Actor(coach_model, **actor_kwargs)
        ds_coach.prepare(coach_model_actor)
        coach_model_actor.gradient_checkpointing_enable()
        logger.info("Coach model initialized (gradient checkpointing ON)")

        ref_strategy = _make_strategy()
        ref_model_actor = Actor(tcfg.model_name, **actor_kwargs)
        ref_strategy.prepare(ref_model_actor, is_rlhf=True)
        ref_model = ref_strategy
        logger.info("Reference model initialized (frozen)")

        ref_module = getattr(ref_strategy, '_engine', ref_strategy)
        ref_module = getattr(ref_module, 'module', ref_module)
        ref_module.to('cpu')
        import torch as _torch_park
        _torch_park.cuda.empty_cache()
        logger.info("Reference model parked on CPU")
```

**Step 3: Update EMA tracker creation (lines 364-376)**

```python
    alice_ema = EMATracker(
        mu=acfg.ema_init_mu, sigma_sq=acfg.ema_init_sigma_sq, eta=acfg.ema_eta,
    )
    bob_ema = EMATracker(
        mu=acfg.ema_init_mu, sigma_sq=acfg.ema_init_sigma_sq, eta=acfg.ema_eta,
    )
    coach_ema = EMATracker(
        mu=acfg.coach_ema_init_mu, sigma_sq=acfg.coach_ema_init_sigma_sq, eta=acfg.ema_eta,
    )
```

**Step 4: Update WorkflowContext construction (lines 387-400)**

```python
    return WorkflowContext(
        player_vllm=player_vllm,
        coach_vllm=coach_vllm,
        ref_model=ref_model,
        ds_alice=ds_alice,
        ds_bob=ds_bob,
        ds_coach=ds_coach,
        config=config,
        alice_ema=alice_ema,
        bob_ema=bob_ema,
        coach_ema=coach_ema,
        rep_buffer=rep_buffer,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        tokenizer=tokenizer,
        coach_tokenizer=coach_tokenizer,
    )
```

**Step 5: Update save_checkpoint (3 subdirs)**

```python
def save_checkpoint(ctx: Any, path: str) -> None:
    import shutil
    os.makedirs(path, exist_ok=True)
    alice_path = os.path.join(path, "alice")
    bob_path = os.path.join(path, "bob")
    coach_path = os.path.join(path, "coach")
    for sub in (alice_path, bob_path, coach_path):
        latest = os.path.join(sub, "latest")
        if os.path.isdir(latest):
            shutil.rmtree(latest)
            logger.warning("Removed stale 'latest' directory: %s", latest)
    if hasattr(ctx.ds_alice, "_engine") and ctx.ds_alice._engine is not None:
        ctx.ds_alice._engine.save_checkpoint(
            alice_path, tag="ckpt",
            client_state={"iteration": ctx.iteration},
        )
    if hasattr(ctx.ds_bob, "_engine") and ctx.ds_bob._engine is not None:
        ctx.ds_bob._engine.save_checkpoint(bob_path, tag="ckpt")
    if hasattr(ctx.ds_coach, "_engine") and ctx.ds_coach._engine is not None:
        ctx.ds_coach._engine.save_checkpoint(coach_path, tag="ckpt")
    logger.info("Checkpoint saved at iteration %d to %s", ctx.iteration, path)
```

**Step 6: Update load_checkpoint**

```python
def load_checkpoint(ctx: Any, path: str) -> None:
    alice_path = os.path.join(path, "alice")
    bob_path = os.path.join(path, "bob")
    coach_path = os.path.join(path, "coach")

    # Backwards compat: try old "player" dir if "alice" doesn't exist
    if not os.path.isdir(alice_path) and os.path.isdir(os.path.join(path, "player")):
        alice_path = os.path.join(path, "player")
        logger.info("Legacy checkpoint: loading 'player' dir as Alice")

    def _load(engine, ckpt_path):
        try:
            return engine.load_checkpoint(ckpt_path, tag=None)
        except Exception:
            pass
        for tag in ("ckpt", "latest"):
            tag_dir = os.path.join(ckpt_path, tag)
            if os.path.isdir(tag_dir):
                logger.info("Loading checkpoint with explicit tag=%s from %s", tag, ckpt_path)
                return engine.load_checkpoint(ckpt_path, tag=tag)
        logger.warning("No checkpoint found at %s", ckpt_path)
        return None, None

    if hasattr(ctx.ds_alice, "_engine") and ctx.ds_alice._engine is not None:
        if os.path.isdir(alice_path):
            _, client_state = _load(ctx.ds_alice._engine, alice_path)
            if client_state and "iteration" in client_state:
                ctx.iteration = client_state["iteration"]
                logger.info("Resumed Alice from iteration %d", ctx.iteration)
    if hasattr(ctx.ds_bob, "_engine") and ctx.ds_bob._engine is not None:
        if os.path.isdir(bob_path):
            _load(ctx.ds_bob._engine, bob_path)
            logger.info("Resumed Bob")
        else:
            logger.warning("No Bob checkpoint at %s — starting Bob from scratch", bob_path)
    if hasattr(ctx.ds_coach, "_engine") and ctx.ds_coach._engine is not None:
        if os.path.isdir(coach_path):
            _load(ctx.ds_coach._engine, coach_path)
        else:
            logger.warning("No coach checkpoint at %s — starting coach from scratch", coach_path)
```

**Step 7: Update save_hf_weights**

```python
def save_hf_weights(ctx: Any, path: str) -> None:
    import torch
    alice_path = os.path.join(path, "alice_hf")
    bob_path = os.path.join(path, "bob_hf")
    coach_path = os.path.join(path, "coach_hf")
    os.makedirs(alice_path, exist_ok=True)
    os.makedirs(bob_path, exist_ok=True)
    os.makedirs(coach_path, exist_ok=True)

    alice_model = ctx.ds_alice._engine.module.model
    bob_model = ctx.ds_bob._engine.module.model
    coach_model = ctx.ds_coach._engine.module.model

    alice_model.save_pretrained(alice_path, safe_serialization=True)
    bob_model.save_pretrained(bob_path, safe_serialization=True)
    coach_model.save_pretrained(coach_path, safe_serialization=True)
    ctx.tokenizer.save_pretrained(alice_path)
    ctx.tokenizer.save_pretrained(bob_path)
    if ctx.coach_tokenizer is not None:
        ctx.coach_tokenizer.save_pretrained(coach_path)
    else:
        ctx.tokenizer.save_pretrained(coach_path)

    with open(os.path.join(path, "iteration.txt"), "w") as f:
        f.write(str(ctx.iteration))

    logger.info("HF weights saved at iteration %d to %s", ctx.iteration, path)
```

**Step 8: Update run() — resume-hf path and logging**

In the `run()` function, update the resume-hf section to handle alice/bob:

```python
    if resume_hf_path:
        alice_hf = os.path.join(resume_hf_path, "alice_hf")
        bob_hf = os.path.join(resume_hf_path, "bob_hf")
        coach_hf = os.path.join(resume_hf_path, "coach_hf")
        # Backwards compat: try old "player_hf"
        if not os.path.isdir(alice_hf) and os.path.isdir(os.path.join(resume_hf_path, "player_hf")):
            alice_hf = os.path.join(resume_hf_path, "player_hf")
            bob_hf = alice_hf  # Both start from same checkpoint
            logger.info("Legacy HF weights: loading player_hf for both Alice and Bob")
        if os.path.isdir(alice_hf):
            config.training.model_name = alice_hf
            logger.info("resume-hf: Alice model → %s", alice_hf)
        # Bob uses same model_name since both are loaded from same base
        # (init_infra creates two independent copies from model_name)
        if os.path.isdir(coach_hf):
            config.training.coach_model_name = coach_hf
            logger.info("resume-hf: coach model → %s", coach_hf)
```

Update the logging in the training loop:

```python
        if ctx.iteration % tcfg.log_freq == 0:
            logger.info(
                "iter=%d alice_loss=%.4f bob_loss=%.4f coach_loss=%s accuracy=%.3f "
                "problems=%d discussions=%d",
                ctx.iteration, result.alice_loss, result.bob_loss,
                f"{result.coach_loss:.4f}" if result.coach_loss is not None else "N/A",
                result.player_accuracy, result.num_problems, result.num_discussions,
            )
```

**Step 9: Run the full test suite**

Run: `python -m pytest tests/ -q --ignore=tests/infra/ --ignore=tests/test_config.py`
Expected: All tests pass. Fix any remaining references to `ds_player` or `player_ema` that surface.

**Step 10: Commit**

```bash
git add crisp/train.py
git commit -m "feat: init_infra creates two player models, checkpoint save/load for alice/bob"
```

---

### Task 6: Update remaining test helpers and config

**Files:**
- Modify: `tests/workflow/test_rollout_step.py` — update `_make_ctx`
- Modify: `tests/workflow/test_coach_step.py` — update `_make_ctx`
- Modify: `tests/workflow/test_collector.py` — update if needed
- Modify: `tests/workflow/test_workflow_api.py` — update if needed
- Modify: `configs/dsi_h200.yaml` — token limit increases
- Modify: `scripts/modal_production.py` — update HF weight references

**Step 1: Update all remaining `_make_ctx` helpers**

In every test file that creates a `WorkflowContext`, replace `ds_player=` with `ds_alice=`/`ds_bob=` and `player_ema=` with `alice_ema=`/`bob_ema=`. Search for:

```bash
grep -rn "ds_player\|player_ema" tests/
```

Update each occurrence.

**Step 2: Update dsi_h200.yaml token limits**

```yaml
training:
  model_name: "Qwen/Qwen3-4B-Instruct-2507"
  coach_model_name: "Qwen/Qwen3-14B"
  num_iterations: 3000
  checkpoint_dir: "checkpoints/ds"
  save_freq: 10
  eval_freq: 0
  attn_implementation: "sdpa"

infra:
  num_gpus_per_node: 4
  vllm_num_engines: 1
  vllm_enable_sleep: false
  max_model_len: 49152
  zero_stage: 2
  bf16: true
  learning_rate: 0.00001
  vllm_gpu_memory_utilization: 0.90
  coach_vllm_gpu_memory_utilization: 0.90
  coach_vllm_max_model_len: 49152
  adam_offload: true

player:
  rollouts_per_problem: 8
  alice_temperature: 0.8
  bob_temperature: 1.0
  max_new_tokens: 12288

coach:
  batch_size: 8
  warmup_iters: 5
  rampup_iters: 75
  repetition_window: 10
  coach_solve_max_new_tokens: 12288
```

**Step 3: Update modal_production.py**

Update `PLAYER_FINETUNED` references and the `--resume-hf` path handling to use `alice_hf`/`bob_hf` (backwards-compat with existing `player_hf` checkpoints is handled in `train.py`).

**Step 4: Run the full test suite**

Run: `python -m pytest tests/ -q --ignore=tests/infra/ --ignore=tests/test_config.py`
Expected: All pass (244+ tests)

**Step 5: Commit**

```bash
git add tests/ configs/dsi_h200.yaml scripts/modal_production.py
git commit -m "chore: update test helpers, H200 token limits, modal scripts for two-player"
```

---

### Task 7: Update memory and verify

**Files:**
- Modify: `/home/alex/.claude/projects/-home-alex-mech-interp-CRISP/memory/MEMORY.md`

**Step 1: Run the full test suite one final time**

Run: `python -m pytest tests/ -q --ignore=tests/infra/ --ignore=tests/test_config.py`
Expected: All pass

**Step 2: Update MEMORY.md**

Update the memory file to reflect the new two-player architecture:
- GPU layout: GPU 2 now hosts Alice + Bob + Ref
- WorkflowContext: `ds_alice`/`ds_bob` instead of `ds_player`
- Independent EMA trackers: `alice_ema`/`bob_ema`
- Sequential rollout generation with weight sync
- Persuader bonus computed in main_loop before per-player split
- Checkpoint dirs: alice/bob/coach instead of player/coach

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: two-player architecture complete — independent Alice/Bob training"
```
