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
