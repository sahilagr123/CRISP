"""Tests for weight sync — no GPU required (mocked)."""
from unittest.mock import patch, MagicMock
import torch


def _make_mock_deepspeed():
    """Create a mock deepspeed module with zero.GatheredParameters."""
    ds = MagicMock()
    ds.zero.GatheredParameters.return_value.__enter__ = MagicMock()
    ds.zero.GatheredParameters.return_value.__exit__ = MagicMock(return_value=False)
    return ds


def test_broadcast_weights_calls_update_weight_for_each_param():
    """broadcast_weights_to_vllm calls update_weight for every named parameter."""
    from crisp.infra.weight_sync import broadcast_weights_to_vllm

    model = MagicMock()
    param1 = torch.nn.Parameter(torch.randn(10, 5))
    param2 = torch.nn.Parameter(torch.randn(3))
    model.module.named_parameters.return_value = [
        ("layer1.weight", param1),
        ("layer1.bias", param2),
    ]

    engine1 = MagicMock()
    engine1.update_weight.remote.return_value = "ref"
    engine2 = MagicMock()
    engine2.update_weight.remote.return_value = "ref"

    mock_ds = _make_mock_deepspeed()

    with patch("crisp.infra.weight_sync.deepspeed", mock_ds), \
         patch("crisp.infra.weight_sync.ray", MagicMock()), \
         patch("torch.distributed.get_rank", return_value=0), \
         patch("torch.cuda.empty_cache"), \
         patch("crisp.infra.weight_sync.torch_dist_barrier_and_cuda_sync"), \
         patch("torch.cuda.current_stream"):
        broadcast_weights_to_vllm(
            model=model,
            vllm_engines=[engine1, engine2],
            model_update_group=MagicMock(),
            zero_stage=2,
        )
        assert engine1.update_weight.remote.call_count == 2
        assert engine2.update_weight.remote.call_count == 2


def test_broadcast_weights_zero3_uses_gathered_parameters():
    """ZeRO-3 path wraps parameters in GatheredParameters context."""
    from crisp.infra.weight_sync import broadcast_weights_to_vllm

    model = MagicMock()
    param = torch.nn.Parameter(torch.randn(4))
    param.ds_shape = torch.Size([4])
    model.module.named_parameters.return_value = [("w", param)]

    engine = MagicMock()
    engine.update_weight.remote.return_value = "ref"

    mock_ds = _make_mock_deepspeed()

    with patch("crisp.infra.weight_sync.deepspeed", mock_ds), \
         patch("crisp.infra.weight_sync.ray", MagicMock()), \
         patch("torch.distributed.get_rank", return_value=0), \
         patch("torch.cuda.empty_cache"), \
         patch("crisp.infra.weight_sync.torch_dist_barrier_and_cuda_sync"), \
         patch("torch.cuda.current_stream"):
        broadcast_weights_to_vllm(
            model=model,
            vllm_engines=[engine],
            model_update_group=MagicMock(),
            zero_stage=3,
        )
        mock_ds.zero.GatheredParameters.assert_called()


def test_broadcast_weights_skips_on_non_rank0():
    """Non-rank-0 processes don't send weights."""
    from crisp.infra.weight_sync import broadcast_weights_to_vllm

    model = MagicMock()
    param = torch.nn.Parameter(torch.randn(4))
    model.module.named_parameters.return_value = [("w", param)]

    engine = MagicMock()

    mock_ds = _make_mock_deepspeed()

    with patch("crisp.infra.weight_sync.deepspeed", mock_ds), \
         patch("crisp.infra.weight_sync.ray", MagicMock()), \
         patch("torch.distributed.get_rank", return_value=1), \
         patch("torch.cuda.empty_cache"), \
         patch("crisp.infra.weight_sync.torch_dist_barrier_and_cuda_sync"), \
         patch("torch.cuda.current_stream"):
        broadcast_weights_to_vllm(
            model=model,
            vllm_engines=[engine],
            model_update_group=MagicMock(),
            zero_stage=2,
        )
        engine.update_weight.remote.assert_not_called()
