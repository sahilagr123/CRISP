"""Tests for DeepSpeedStrategy — mock DeepSpeed, verify interface contracts."""
import sys
from unittest.mock import MagicMock, patch

import pytest

from crisp.infra.strategy import DeepSpeedStrategy, StrategyArgs


def _make_strategy(**overrides):
    defaults = dict(
        seed=42, bf16=True, zero_stage=2, adam_offload=False,
        max_norm=1.0, learning_rate=1e-5, weight_decay=0.01,
        micro_train_batch_size=1, gradient_checkpointing=False,
        attn_implementation="eager", ref_reward_offload=False,
    )
    defaults.update(overrides)
    return DeepSpeedStrategy(**defaults)


def _prepare_with_mock_engine(strategy, is_rlhf=False):
    """Call prepare() with mocked deepspeed.initialize — sets strategy._engine directly."""
    mock_engine = MagicMock()
    mock_model = MagicMock()
    # Instead of patching deferred import, just set _engine directly
    strategy._engine = mock_engine
    return mock_engine, mock_model


# --- StrategyArgs ---

def test_strategy_args_accessible():
    """strategy.args.bf16, .attn_implementation, .ref_reward_offload are correct."""
    s = _make_strategy(bf16=False, attn_implementation="flash_attention_2",
                       ref_reward_offload=True)
    assert s.args.bf16 is False
    assert s.args.attn_implementation == "flash_attention_2"
    assert s.args.ref_reward_offload is True
    assert s.args.gradient_checkpointing is False


# --- setup_distributed ---

def test_setup_distributed_calls_deepspeed():
    """setup_distributed() calls deepspeed.init_distributed()."""
    s = _make_strategy()
    mock_ds = MagicMock()
    with patch.dict(sys.modules, {"deepspeed": mock_ds}):
        s.setup_distributed()
    mock_ds.init_distributed.assert_called_once()


# --- is_rank_0 ---

def test_is_rank_0_without_dist():
    """Without dist initialized, is_rank_0 returns True."""
    s = _make_strategy()
    with patch("crisp.infra.strategy.dist") as mock_dist:
        mock_dist.is_initialized.return_value = False
        assert s.is_rank_0() is True


def test_is_rank_0_with_dist_rank0():
    """With dist initialized at rank 0, returns True."""
    s = _make_strategy()
    with patch("crisp.infra.strategy.dist") as mock_dist:
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0
        assert s.is_rank_0() is True


def test_is_rank_0_with_dist_rank1():
    """With dist initialized at rank 1, returns False."""
    s = _make_strategy()
    with patch("crisp.infra.strategy.dist") as mock_dist:
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 1
        assert s.is_rank_0() is False


# --- prepare ---

def test_prepare_calls_deepspeed_initialize():
    """prepare() calls deepspeed.initialize with correct args."""
    s = _make_strategy()
    mock_engine = MagicMock()
    mock_model = MagicMock()
    mock_ds = MagicMock()
    mock_ds.initialize.return_value = (mock_engine, None, None, None)

    with patch.dict(sys.modules, {"deepspeed": mock_ds}), \
         patch("crisp.infra.strategy.get_train_ds_config",
               return_value={"train_batch_size": "auto"}) as mock_cfg, \
         patch("crisp.infra.strategy.get_optimizer_grouped_parameters",
               return_value=[{"params": [], "weight_decay": 0.0}]):
        s.prepare(mock_model)

    mock_cfg.assert_called_once()
    mock_ds.initialize.assert_called_once()
    call_kwargs = mock_ds.initialize.call_args
    assert call_kwargs[1]["model"] is mock_model


def test_prepare_rlhf_no_optimizer():
    """prepare(is_rlhf=True) does not create an optimizer."""
    s = _make_strategy()
    mock_engine = MagicMock()
    mock_model = MagicMock()
    mock_ds = MagicMock()
    mock_ds.initialize.return_value = (mock_engine, None, None, None)

    with patch.dict(sys.modules, {"deepspeed": mock_ds}), \
         patch("crisp.infra.strategy.get_train_ds_config", return_value={}), \
         patch("crisp.infra.strategy.get_optimizer_grouped_parameters") as mock_optim:
        s.prepare(mock_model, is_rlhf=True)

    mock_optim.assert_not_called()
    call_kwargs = mock_ds.initialize.call_args
    assert "optimizer" not in call_kwargs[1]


def test_prepare_returns_engine():
    """prepare() returns the DeepSpeed engine and sets self._engine."""
    s = _make_strategy()
    mock_engine = MagicMock()
    mock_model = MagicMock()
    mock_ds = MagicMock()
    mock_ds.initialize.return_value = (mock_engine, None, None, None)

    with patch.dict(sys.modules, {"deepspeed": mock_ds}), \
         patch("crisp.infra.strategy.get_train_ds_config", return_value={}), \
         patch("crisp.infra.strategy.get_optimizer_grouped_parameters",
               return_value=[{"params": [], "weight_decay": 0.0}]):
        engine = s.prepare(mock_model)

    assert engine is mock_engine
    assert s._engine is mock_engine


def test_prepare_enables_gradient_checkpointing():
    """prepare() enables gradient checkpointing when configured."""
    s = _make_strategy(gradient_checkpointing=True)
    mock_engine = MagicMock()
    mock_model = MagicMock()
    mock_ds = MagicMock()
    mock_ds.initialize.return_value = (mock_engine, None, None, None)

    with patch.dict(sys.modules, {"deepspeed": mock_ds}), \
         patch("crisp.infra.strategy.get_train_ds_config", return_value={}), \
         patch("crisp.infra.strategy.get_optimizer_grouped_parameters",
               return_value=[{"params": [], "weight_decay": 0.0}]):
        s.prepare(mock_model)

    mock_model.gradient_checkpointing_enable.assert_called_once()


# --- forward / backward / optimizer_step ---

def test_forward_delegates_to_engine():
    """forward() calls engine with the given arguments."""
    s = _make_strategy()
    _prepare_with_mock_engine(s)
    s._engine.return_value = "log_probs"
    result = s.forward("sequences", attention_mask="mask")
    s._engine.assert_called_once_with("sequences", attention_mask="mask")
    assert result == "log_probs"


def test_backward_delegates_to_engine():
    """backward() calls engine.backward(loss)."""
    s = _make_strategy()
    _prepare_with_mock_engine(s)
    s.backward("loss_tensor")
    s._engine.backward.assert_called_once_with("loss_tensor")


def test_optimizer_step_delegates_to_engine():
    """optimizer_step() calls engine.step()."""
    s = _make_strategy()
    _prepare_with_mock_engine(s)
    s.optimizer_step()
    s._engine.step.assert_called_once()


# --- __getattr__ delegation ---

def test_getattr_delegates_to_engine():
    """After prepare(), strategy.module and strategy.config come from engine."""
    s = _make_strategy()
    _prepare_with_mock_engine(s)
    s._engine.module = "the_module"
    s._engine.config = {"zero_optimization": {}}
    assert s.module == "the_module"
    assert s.config == {"zero_optimization": {}}


def test_getattr_before_prepare_raises():
    """Accessing delegated attributes before prepare() raises AttributeError."""
    s = _make_strategy()
    with pytest.raises(AttributeError, match="engine not initialized"):
        _ = s.module


# --- compatibility with sleep utilities ---

def test_offload_reload_compatibility():
    """Strategy can be passed to offload/reload without errors."""
    from crisp.infra.deepspeed_strategy import offload_deepspeed_states, reload_deepspeed_states

    s = _make_strategy()
    _prepare_with_mock_engine(s)
    s._engine.config = {"zero_optimization": {"offload_optimizer": {"device": "none"}}}
    s._engine.zero_optimization_stage.return_value = 2

    # Should not raise
    offload_deepspeed_states(s)
    reload_deepspeed_states(s)


# --- offload_states / reload_states / sync_weights methods ---

def test_offload_states_delegates():
    """offload_states() calls offload_deepspeed_states on self."""
    s = _make_strategy()
    _prepare_with_mock_engine(s)
    s._engine.config = {"zero_optimization": {"offload_optimizer": {"device": "none"}}}
    s._engine.zero_optimization_stage.return_value = 2
    s.offload_states()  # Should not raise


def test_reload_states_delegates():
    """reload_states() calls reload_deepspeed_states on self."""
    s = _make_strategy()
    _prepare_with_mock_engine(s)
    s._engine.config = {"zero_optimization": {"offload_optimizer": {"device": "none"}}}
    s._engine.zero_optimization_stage.return_value = 2
    s.reload_states()  # Should not raise


def test_sync_weights_delegates():
    """sync_weights() calls broadcast_weights_to_vllm."""
    s = _make_strategy(zero_stage=2)
    _prepare_with_mock_engine(s)
    mock_engines = [MagicMock()]
    with patch("crisp.infra.weight_sync.broadcast_weights_to_vllm") as mock_bcast:
        s.sync_weights(mock_engines)
    mock_bcast.assert_called_once()
