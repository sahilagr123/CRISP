"""Distributed training actors and strategy proxy for multi-GPU CRISP.

CRISPModelActor: Ray actor hosting a player/coach model on one GPU.
DistributedStrategy: Proxy wrapping RayActorGroup with same interface as DeepSpeedStrategy.
"""
from __future__ import annotations

from typing import Any, Optional

try:
    import ray
except ImportError:
    ray = None

from .ray_launcher import BaseModelActor, _ray_remote_decorator
from .strategy import DeepSpeedStrategy


@_ray_remote_decorator
class CRISPModelActor(BaseModelActor):
    """Ray actor hosting a player or coach model on a single GPU.

    Mirrors ReferenceModelActor but supports training (backward + optimizer_step).
    """

    def init_model_from_pretrained(
        self,
        strategy_kwargs: dict,
        pretrain: str,
        actor_kwargs: dict,
        is_rlhf: bool = False,
    ):
        """Create local DeepSpeedStrategy + Actor, call prepare()."""
        from .actor_model import Actor

        self.strategy = DeepSpeedStrategy(**strategy_kwargs)
        self._setup_distributed(self.strategy)
        model = Actor(pretrain, **actor_kwargs)
        self.engine = self.strategy.prepare(model, is_rlhf=is_rlhf)

    def forward(self, *args, **kwargs):
        return self.strategy.forward(*args, **kwargs)

    def backward(self, loss):
        self.strategy.backward(loss)

    def optimizer_step(self):
        self.strategy.optimizer_step()

    def offload_states(self, **kwargs):
        self.strategy.offload_states(**kwargs)

    def reload_states(self, **kwargs):
        self.strategy.reload_states(**kwargs)

    def sync_weights(self, vllm_engines, **kwargs):
        self.strategy.sync_weights(vllm_engines, **kwargs)

    def save_checkpoint(self, path, tag, client_state=None):
        self.strategy._engine.save_checkpoint(path, tag=tag, client_state=client_state)

    def load_checkpoint(self, path, tag=None):
        return self.strategy._engine.load_checkpoint(path, tag=tag)


class DistributedStrategy:
    """Proxy wrapping RayActorGroup with same interface as DeepSpeedStrategy.

    Dispatch rules:
    - INFERENCE (forward): rank-0 only, return result
    - TRAINING (backward, optimizer_step): ALL ranks (ZeRO shards)
    - LIFECYCLE (offload_states, reload_states): ALL ranks
    - WEIGHT SYNC (sync_weights): rank-0 only (broadcasts via NCCL)
    """

    def __init__(self, actor_group):
        self._group = actor_group

    def forward(self, *args, **kwargs):
        """Inference: rank-0 only."""
        refs = self._group.async_run_method("forward", *args, **kwargs)
        return ray.get(refs[0])

    def backward(self, loss):
        """Training: ALL ranks (ZeRO needs gradients on every shard)."""
        refs = self._group.async_run_method("backward", loss)
        ray.get(refs)

    def optimizer_step(self):
        """Training: ALL ranks."""
        refs = self._group.async_run_method("optimizer_step")
        ray.get(refs)

    def offload_states(self, **kwargs):
        """Lifecycle: ALL ranks (each offloads its own shard)."""
        refs = self._group.async_run_method("offload_states", **kwargs)
        ray.get(refs)

    def reload_states(self, **kwargs):
        """Lifecycle: ALL ranks."""
        refs = self._group.async_run_method("reload_states", **kwargs)
        ray.get(refs)

    def sync_weights(self, vllm_engines, **kwargs):
        """Weight sync: rank-0 broadcasts to vLLM engines."""
        ref = self._group._actor_handlers[0].sync_weights.remote(vllm_engines, **kwargs)
        ray.get(ref)

    def save_checkpoint(self, path, tag, client_state=None):
        """Checkpoint: all ranks save (DeepSpeed handles multi-rank internally)."""
        refs = self._group.async_run_method("save_checkpoint", path, tag, client_state=client_state)
        ray.get(refs)

    def load_checkpoint(self, path, tag=None):
        """Checkpoint: all ranks load."""
        refs = self._group.async_run_method("load_checkpoint", path, tag)
        return ray.get(refs[0])
