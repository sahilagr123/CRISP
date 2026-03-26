"""Tests for DeepSpeed config builders — no GPU required."""


def test_get_train_ds_config_zero2():
    """ZeRO-2 train config has correct structure."""
    from crisp.infra.deepspeed_strategy import get_train_ds_config
    cfg = get_train_ds_config(stage=2, bf16=True, max_norm=1.0)
    assert cfg["zero_optimization"]["stage"] == 2
    assert cfg["bf16"]["enabled"] is True
    assert cfg["gradient_clipping"] == 1.0
    assert "offload_param" in cfg["zero_optimization"]


def test_get_train_ds_config_zero3():
    """ZeRO-3 train config enables reduce_scatter."""
    from crisp.infra.deepspeed_strategy import get_train_ds_config
    cfg = get_train_ds_config(stage=3, bf16=True, max_norm=1.0)
    assert cfg["zero_optimization"]["stage"] == 3
    assert cfg["zero_optimization"]["reduce_scatter"] is True


def test_get_eval_ds_config():
    """Eval config uses stage 0 for non-ZeRO-3."""
    from crisp.infra.deepspeed_strategy import get_eval_ds_config
    cfg = get_eval_ds_config(offload=False, stage=0, bf16=True)
    assert cfg["zero_optimization"]["stage"] == 0
    assert cfg["bf16"]["enabled"] is True


def test_get_eval_ds_config_offload():
    """Eval config with offload uses CPU."""
    from crisp.infra.deepspeed_strategy import get_eval_ds_config
    cfg = get_eval_ds_config(offload=True, stage=0, bf16=True)
    assert cfg["zero_optimization"]["offload_param"]["device"] == "cpu"


def test_get_optimizer_grouped_parameters():
    """Weight decay excludes bias and norm layers."""
    from crisp.infra.deepspeed_strategy import get_optimizer_grouped_parameters
    from unittest.mock import MagicMock
    import torch

    model = MagicMock()
    param_with_decay = torch.nn.Parameter(torch.randn(10))
    param_bias = torch.nn.Parameter(torch.randn(10))
    param_norm = torch.nn.Parameter(torch.randn(10))
    model.named_parameters.return_value = [
        ("layer.weight", param_with_decay),
        ("layer.bias", param_bias),
        ("layer_norm.weight", param_norm),
    ]
    groups = get_optimizer_grouped_parameters(model, weight_decay=0.01)
    assert len(groups) == 2
    assert len(groups[0]["params"]) == 1
    assert groups[0]["weight_decay"] == 0.01
    assert len(groups[1]["params"]) == 2
    assert groups[1]["weight_decay"] == 0.0


def test_offload_reload_noop_when_adam_offload():
    """offload/reload are no-ops when adam_offload is already enabled."""
    from crisp.infra.deepspeed_strategy import offload_deepspeed_states, reload_deepspeed_states
    from unittest.mock import MagicMock

    model = MagicMock()
    model.zero_optimization_stage.return_value = 3
    model.config = {"zero_optimization": {"offload_optimizer": {"device": "cpu"}}}
    offload_deepspeed_states(model)
    reload_deepspeed_states(model)
    model.optimizer.offload_states.assert_not_called()
    model.reload_states.assert_not_called()
