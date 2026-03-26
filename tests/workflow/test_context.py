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
