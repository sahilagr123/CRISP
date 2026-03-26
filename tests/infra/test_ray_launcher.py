"""Tests for Ray launcher — no GPU required (mocked)."""
import sys
from unittest.mock import patch, MagicMock

# Install a mock 'ray' module so crisp.infra.ray_launcher can import it.
_mock_ray = MagicMock()
_mock_ray.get_gpu_ids = MagicMock(return_value=[0])
sys.modules.setdefault("ray", _mock_ray)
sys.modules.setdefault("ray.util", MagicMock())
sys.modules.setdefault("ray.util.placement_group", MagicMock())
sys.modules.setdefault("ray.util.scheduling_strategies", MagicMock())


def test_base_distributed_actor_sets_env():
    """BaseDistributedActor sets required env vars."""
    from crisp.infra.ray_launcher import BaseDistributedActor
    with patch.dict("os.environ", {}, clear=False), \
         patch("ray.get_gpu_ids", return_value=[0]), \
         patch.object(BaseDistributedActor, "_get_current_node_ip", return_value="10.0.0.1"):
        actor = BaseDistributedActor(
            world_size=2, rank=0, master_addr=None, master_port=12345
        )
        import os
        assert os.environ["WORLD_SIZE"] == "2"
        assert os.environ["RANK"] == "0"
        assert os.environ["MASTER_PORT"] == "12345"
        assert actor._master_addr == "10.0.0.1"


def test_base_distributed_actor_get_master_addr_port():
    """get_master_addr_port returns stored values."""
    from crisp.infra.ray_launcher import BaseDistributedActor
    with patch.dict("os.environ", {}, clear=False), \
         patch("ray.get_gpu_ids", return_value=[0]):
        actor = BaseDistributedActor(
            world_size=1, rank=0, master_addr="10.0.0.5", master_port=9999
        )
        addr, port = actor.get_master_addr_port()
        assert addr == "10.0.0.5"
        assert port == 9999


def test_base_model_actor_empty_cache():
    """empty_cache calls torch.cuda.empty_cache and synchronize."""
    from crisp.infra.ray_launcher import BaseModelActor
    with patch.dict("os.environ", {}, clear=False), \
         patch("ray.get_gpu_ids", return_value=[0]), \
         patch("torch.cuda.empty_cache") as mock_empty, \
         patch("torch.cuda.synchronize") as mock_sync:
        actor = BaseModelActor(world_size=1, rank=0, master_addr="x", master_port=1)
        actor.empty_cache()
        mock_empty.assert_called_once()
        mock_sync.assert_called_once()
