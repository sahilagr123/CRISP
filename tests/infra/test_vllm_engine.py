"""Tests for vLLM engine wrapper — no GPU required (mocked)."""
import sys
from unittest.mock import patch, MagicMock


def _ensure_mock_ray():
    """Insert a mock ``ray`` module into sys.modules if not installed."""
    if "ray" not in sys.modules:
        mock_ray = MagicMock()
        mock_ray.util = MagicMock()
        mock_ray.util.placement_group = MagicMock()
        mock_ray.util.scheduling_strategy = MagicMock()
        sys.modules.setdefault("ray", mock_ray)
        sys.modules.setdefault("ray.util", mock_ray.util)
        sys.modules.setdefault("ray.util.placement_group", mock_ray.util.placement_group)
        sys.modules.setdefault("ray.util.scheduling_strategy", mock_ray.util.scheduling_strategy)
    return sys.modules["ray"]


def test_batch_vllm_engine_call():
    """batch_vllm_engine_call fans out to all engines."""
    mock_ray = _ensure_mock_ray()

    from crisp.infra.vllm_engine import batch_vllm_engine_call

    engine1 = MagicMock()
    engine2 = MagicMock()
    engine1.sleep.remote.return_value = "ref1"
    engine2.sleep.remote.return_value = "ref2"

    mock_ray.get.return_value = ["ok", "ok"]

    with patch("torch.distributed.is_initialized", return_value=False):
        result = batch_vllm_engine_call([engine1, engine2], "sleep")
        engine1.sleep.remote.assert_called_once()
        engine2.sleep.remote.assert_called_once()
        mock_ray.get.assert_called_once_with(["ref1", "ref2"])
        assert result == ["ok", "ok"]


def test_batch_vllm_engine_call_rank0_only():
    """batch_vllm_engine_call skips non-rank-0 when rank_0_only=True."""
    _ensure_mock_ray()

    from crisp.infra.vllm_engine import batch_vllm_engine_call

    engine = MagicMock()

    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=1):
        result = batch_vllm_engine_call([engine], "sleep", rank_0_only=True)
        assert result is None
        engine.sleep.remote.assert_not_called()


def test_create_vllm_engines_returns_correct_count():
    """create_vllm_engines creates the requested number of engines."""
    mock_ray = _ensure_mock_ray()

    from crisp.infra.vllm_engine import create_vllm_engines

    mock_engine = MagicMock()
    mock_actor_cls = MagicMock()
    mock_actor_cls.options.return_value.remote.return_value = mock_engine

    # Setup placement group mock
    mock_pg = MagicMock()
    mock_pg.ready.return_value = True
    mock_ray.util.placement_group.placement_group.return_value = mock_pg

    engines = create_vllm_engines(
        num_engines=3,
        tensor_parallel_size=1,
        pretrain="test-model",
        seed=42,
        full_determinism=False,
        enable_prefix_caching=False,
        enforce_eager=True,
        max_model_len=4096,
        llm_actor_cls=mock_actor_cls,
    )
    assert len(engines) == 3
