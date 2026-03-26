"""Tests for crisp.infra.utils — no GPU required."""
from unittest.mock import patch, MagicMock
import os
import sys


def test_ray_noset_visible_devices_false():
    """Returns False when no NOSET env vars are set."""
    from crisp.infra.utils import ray_noset_visible_devices

    assert ray_noset_visible_devices(env_vars={}) is False


def test_ray_noset_visible_devices_cuda():
    """Returns True when CUDA NOSET env var is set."""
    from crisp.infra.utils import ray_noset_visible_devices

    env = {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
    assert ray_noset_visible_devices(env_vars=env) is True


def test_ray_noset_visible_devices_rocr():
    """Returns True when ROCm NOSET env var is set."""
    from crisp.infra.utils import ray_noset_visible_devices

    env = {"RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1"}
    assert ray_noset_visible_devices(env_vars=env) is True


def test_torch_dist_barrier_and_cuda_sync():
    """Calls barrier then synchronize."""
    from crisp.infra.utils import torch_dist_barrier_and_cuda_sync

    with patch("torch.distributed.barrier") as mock_barrier, \
         patch("torch.cuda.synchronize") as mock_sync:
        torch_dist_barrier_and_cuda_sync()
        mock_barrier.assert_called_once()
        mock_sync.assert_called_once()


def test_get_bundle_indices():
    """Extracts correct slice of bundle indices."""
    from crisp.infra.utils import get_bundle_indices

    # Mock placement group table: 4 bundles across 2 nodes
    mock_pg = MagicMock()
    pg_table = {
        "bundles_to_node_id": {
            "0": "node-a",
            "1": "node-a",
            "2": "node-b",
            "3": "node-b",
        }
    }

    # Create mock ray module since ray may not be installed
    mock_ray = MagicMock()
    mock_ray.util.placement_group_table.return_value = pg_table
    with patch.dict(sys.modules, {"ray": mock_ray, "ray.util": mock_ray.util}):
        indices = get_bundle_indices(mock_pg, 0, 2)
        assert len(indices) == 2
        indices2 = get_bundle_indices(mock_pg, 1, 2)
        assert len(indices2) == 2
        # All 4 indices should be covered
        assert set(indices + indices2) == {"0", "1", "2", "3"}


def test_infra_public_api():
    """crisp.infra exposes the public API."""
    import crisp.infra as infra

    # Config builders
    assert hasattr(infra, "get_train_ds_config")
    assert hasattr(infra, "get_eval_ds_config")

    # Weight sync
    assert hasattr(infra, "broadcast_weights_to_vllm")

    # Experience
    assert hasattr(infra, "generate_samples")
    assert hasattr(infra, "map_vllm_output_to_rollout")

    # Sleep mode
    assert hasattr(infra, "offload_deepspeed_states")
    assert hasattr(infra, "reload_deepspeed_states")
