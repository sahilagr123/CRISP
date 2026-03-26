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
    assert kwargs.get("js_beta") == ctx.config.grpo.coach_js_beta
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
