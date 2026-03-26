"""Tests for vLLM WorkerWrap — no GPU required (mocked)."""
import sys
from unittest.mock import patch, MagicMock
import torch


def test_worker_wrap_init_process_group_nccl():
    """WorkerWrap initializes NCCL process group."""
    from crisp.infra.vllm_worker_wrap import WorkerWrap
    wrap = WorkerWrap()
    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=1), \
         patch("crisp.infra.utils.init_process_group") as mock_init:
        mock_init.return_value = MagicMock()
        wrap.init_process_group(
            master_address="127.0.0.1",
            master_port=12345,
            rank_offset=1,
            world_size=3,
            group_name="test_group",
            backend="nccl",
            use_ray=False,
        )
        mock_init.assert_called_once()
        assert wrap._model_update_group is not None
        assert wrap._model_update_with_ray is False


def test_worker_wrap_init_process_group_ray():
    """WorkerWrap can use Ray collective backend."""
    from crisp.infra.vllm_worker_wrap import WorkerWrap
    wrap = WorkerWrap()

    # Mock ray modules since ray may not be installed
    mock_ray = MagicMock()
    mock_ray_util = MagicMock()
    mock_ray_collective = MagicMock()
    mock_ray.util = mock_ray_util
    mock_ray.util.collective = mock_ray_collective

    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=1), \
         patch.dict(sys.modules, {
             "ray": mock_ray,
             "ray.util": mock_ray_util,
             "ray.util.collective": mock_ray_collective,
         }):
        wrap.init_process_group(
            master_address="127.0.0.1",
            master_port=12345,
            rank_offset=1,
            world_size=3,
            group_name="test_group",
            backend="nccl",
            use_ray=True,
        )
        mock_ray_collective.init_collective_group.assert_called_once()
        assert wrap._model_update_with_ray is True
        assert wrap._model_update_group == "test_group"
