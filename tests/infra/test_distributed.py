"""Tests for distributed training actors and strategy proxy."""
import sys
from unittest.mock import MagicMock, patch


def test_crisp_model_actor_init():
    """CRISPModelActor.init_model_from_pretrained creates strategy + model."""
    from crisp.infra.distributed import CRISPModelActor

    actor = CRISPModelActor.__new__(CRISPModelActor)

    mock_strategy = MagicMock()
    mock_engine = MagicMock()
    mock_strategy.prepare.return_value = mock_engine

    with patch("crisp.infra.distributed.DeepSpeedStrategy", return_value=mock_strategy), \
         patch("crisp.infra.actor_model.Actor") as MockActor:
        actor.init_model_from_pretrained(
            strategy_kwargs={"seed": 42, "bf16": True, "zero_stage": 2},
            pretrain="test-model",
            actor_kwargs={"bf16": True},
        )

    assert actor.strategy is mock_strategy
    mock_strategy.prepare.assert_called_once()
    MockActor.assert_called_once_with("test-model", bf16=True)


def test_crisp_model_actor_forward():
    """forward() delegates to strategy.forward()."""
    from crisp.infra.distributed import CRISPModelActor

    actor = CRISPModelActor.__new__(CRISPModelActor)
    actor.strategy = MagicMock()
    actor.strategy.forward.return_value = "logits"

    result = actor.forward("input_ids", attention_mask="mask")
    actor.strategy.forward.assert_called_once_with("input_ids", attention_mask="mask")
    assert result == "logits"


def test_crisp_model_actor_backward_and_step():
    """backward() and optimizer_step() delegate to strategy."""
    from crisp.infra.distributed import CRISPModelActor

    actor = CRISPModelActor.__new__(CRISPModelActor)
    actor.strategy = MagicMock()

    actor.backward("loss")
    actor.strategy.backward.assert_called_once_with("loss")

    actor.optimizer_step()
    actor.strategy.optimizer_step.assert_called_once()


def test_crisp_model_actor_offload_reload():
    """offload_states() and reload_states() delegate to strategy."""
    from crisp.infra.distributed import CRISPModelActor

    actor = CRISPModelActor.__new__(CRISPModelActor)
    actor.strategy = MagicMock()

    actor.offload_states()
    actor.strategy.offload_states.assert_called_once()

    actor.reload_states()
    actor.strategy.reload_states.assert_called_once()


def test_crisp_model_actor_sync_weights():
    """sync_weights() delegates to strategy."""
    from crisp.infra.distributed import CRISPModelActor

    actor = CRISPModelActor.__new__(CRISPModelActor)
    actor.strategy = MagicMock()
    engines = [MagicMock()]

    actor.sync_weights(engines, model_update_group="group")
    actor.strategy.sync_weights.assert_called_once_with(engines, model_update_group="group")


# --- DistributedStrategy proxy tests ---

def test_distributed_strategy_forward_rank0_only():
    """forward() calls rank-0 actor only and returns its result."""
    from crisp.infra.distributed import DistributedStrategy

    mock_group = MagicMock()
    rank0_ref = MagicMock()
    mock_group.async_run_method.return_value = [rank0_ref, MagicMock()]

    mock_ray = MagicMock()
    mock_ray.get.return_value = "logits"

    with patch("crisp.infra.distributed.ray", mock_ray):
        ds = DistributedStrategy(mock_group)
        result = ds.forward("input_ids", attention_mask="mask")

    mock_group.async_run_method.assert_called_once_with("forward", "input_ids", attention_mask="mask")
    mock_ray.get.assert_called_once_with(rank0_ref)
    assert result == "logits"


def test_distributed_strategy_backward_all_ranks():
    """backward() dispatches to ALL ranks and waits."""
    from crisp.infra.distributed import DistributedStrategy

    mock_group = MagicMock()
    refs = [MagicMock(), MagicMock()]
    mock_group.async_run_method.return_value = refs

    mock_ray = MagicMock()

    with patch("crisp.infra.distributed.ray", mock_ray):
        ds = DistributedStrategy(mock_group)
        ds.backward("loss")

    mock_group.async_run_method.assert_called_once_with("backward", "loss")
    mock_ray.get.assert_called_once_with(refs)


def test_distributed_strategy_optimizer_step_all_ranks():
    """optimizer_step() dispatches to ALL ranks."""
    from crisp.infra.distributed import DistributedStrategy

    mock_group = MagicMock()
    refs = [MagicMock(), MagicMock()]
    mock_group.async_run_method.return_value = refs

    mock_ray = MagicMock()

    with patch("crisp.infra.distributed.ray", mock_ray):
        ds = DistributedStrategy(mock_group)
        ds.optimizer_step()

    mock_group.async_run_method.assert_called_once_with("optimizer_step")
    mock_ray.get.assert_called_once_with(refs)


def test_distributed_strategy_offload_all_ranks():
    """offload_states() dispatches to ALL ranks."""
    from crisp.infra.distributed import DistributedStrategy

    mock_group = MagicMock()
    refs = [MagicMock()]
    mock_group.async_run_method.return_value = refs
    mock_ray = MagicMock()

    with patch("crisp.infra.distributed.ray", mock_ray):
        ds = DistributedStrategy(mock_group)
        ds.offload_states()

    mock_group.async_run_method.assert_called_once_with("offload_states")
    mock_ray.get.assert_called_once_with(refs)


def test_distributed_strategy_sync_weights_rank0():
    """sync_weights() dispatches to rank-0 actor only."""
    from crisp.infra.distributed import DistributedStrategy

    mock_group = MagicMock()
    rank0_actor = MagicMock()
    mock_group._actor_handlers = [rank0_actor, MagicMock()]

    mock_ray = MagicMock()

    with patch("crisp.infra.distributed.ray", mock_ray):
        ds = DistributedStrategy(mock_group)
        engines = [MagicMock()]
        ds.sync_weights(engines, model_update_group="grp")

    rank0_actor.sync_weights.remote.assert_called_once_with(engines, model_update_group="grp")
    mock_ray.get.assert_called_once()
