"""Ray actor base classes and actor group orchestrator.

Vendored from MARTI (https://github.com/TsinghuaC3I/MARTI).
Provides BaseDistributedActor (env setup), BaseModelActor (DeepSpeed init),
RayActorGroup (placement groups + batched execution), and
ReferenceModelActor (frozen ref policy for JS-divergence).

Removed: RewardModelActor, CriticModelActor (not needed for GRPO).
"""
import logging
import os
import socket
from typing import Any, Dict, Optional, Type

try:
    import ray
    from ray.util.placement_group import PlacementGroup, placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
except ImportError:  # pragma: no cover – ray is optional at import time
    ray = None  # type: ignore[assignment]
    PlacementGroup = Any  # type: ignore[assignment,misc]

import torch
from tqdm import tqdm

from .utils import ray_noset_visible_devices


class BaseDistributedActor:
    """Sets up distributed environment for a Ray actor."""

    def __init__(self, world_size, rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = (
            str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"
        )

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


class BaseModelActor(BaseDistributedActor):
    """Base for model-hosting Ray actors with DeepSpeed."""

    def _setup_distributed(self, strategy):
        self.strategy = strategy
        strategy.setup_distributed()

    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError()

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def execute_batch(self, method_name: str, all_data, start_idx, end_idx):
        """Process a slice of batched data by calling method per-sample."""
        kwargs = {key: value[start_idx:end_idx] for key, value in all_data.items()}
        first_param = next(iter(kwargs.values()))
        list_length = len(first_param)

        for param_name, param_value in kwargs.items():
            if len(param_value) != list_length:
                raise ValueError(
                    f"Parameter {param_name} has length {len(param_value)}, expected {list_length}"
                )

        func = getattr(self, method_name)
        if not callable(func):
            raise ValueError(f"Function {method_name} is not callable")

        results = []
        for i in tqdm(range(list_length), desc=f"{method_name}",
                      disable=not self.strategy.is_rank_0()):
            sample_kwargs = {
                param_name: param_value[i]
                for param_name, param_value in kwargs.items()
            }
            result = func(**sample_kwargs)
            results.append(result)

        return results


def _ray_remote_decorator(cls):
    """Apply @ray.remote(num_gpus=1) only when ray is available."""
    if ray is not None:
        return ray.remote(num_gpus=1)(cls)
    return cls  # pragma: no cover


@_ray_remote_decorator
class ReferenceModelActor(BaseModelActor):
    """Frozen reference policy for computing reference log-probs.

    WARNING: This actor must NEVER be included in any optimizer group.
    Accidental updates corrupt the JS-divergence signal silently.
    """

    def init_model_from_pretrained(self, strategy, pretrain):
        from .actor_model import Actor
        from .deepspeed_strategy import get_eval_ds_config

        self._setup_distributed(strategy)
        model = Actor(
            pretrain,
            bf16=strategy.args.bf16,
            attn_implementation=strategy.args.attn_implementation,
            ds_config=get_eval_ds_config(
                offload=getattr(strategy.args, "ref_reward_offload", False)
            ),
        )

        if getattr(strategy.args, "ref_reward_offload", False):
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            log_probs = self.model(
                sequences.to(device),
                action_mask.to(device) if action_mask is not None else None,
                attention_mask.to(device),
            )
        return log_probs.to("cpu")


class RayActorGroup:
    """Orchestrates a group of Ray actors across GPUs with placement groups."""

    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[BaseModelActor],
        pg: "PlacementGroup" = None,
        num_gpus_per_actor=1,
        duplicate_actors: int = 1,
        resources: Dict[str, float] = None,
        num_resources_per_node: int = None,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type
        self.duplicate_actors = duplicate_actors
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node
        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(world_size)]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node
            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())

        if pg:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0
                ),
            ).remote(world_size, 0, None, None)
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(world_size, 0, None, None)

        self._actor_handlers = [master_actor]

        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                if pg:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=rank,
                        ),
                    ).remote(world_size, rank, master_addr, master_port)
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                    ).remote(world_size, rank, master_addr, master_port)
                self._actor_handlers.append(worker_actor)

    def async_init_model_from_pretrained(self, *args, **kwargs):
        return [
            actor.init_model_from_pretrained.remote(*args, **kwargs)
            for actor in self._actor_handlers
        ]

    def async_save_model(self):
        return [actor.save_model.remote() for actor in self._actor_handlers]

    def async_run_method(self, method_name, *args, **kwargs):
        refs = []
        for actor in self._actor_handlers:
            method = getattr(actor, method_name)
            refs.append(method.remote(*args, **kwargs))
        return refs

    def async_run_method_batch(self, method_name, **kwargs):
        """Distribute batched work across actors with round-robin scheduling."""
        for key, value in kwargs.items():
            if not hasattr(value, "__len__"):
                raise ValueError(f"Parameter {key} must be iterable")

        first_param = next(iter(kwargs.values()))
        total_length = len(first_param)

        for key, value in kwargs.items():
            if len(value) != total_length:
                raise ValueError(
                    f"All parameters must have the same length. "
                    f"{key} has length {len(value)}, expected {total_length}"
                )

        num_actors = len(self._actor_handlers)
        effective_actors = num_actors // self.duplicate_actors
        chunk_size = total_length // effective_actors
        assert total_length >= effective_actors, (
            f"Total length {total_length} must be >= effective actors {effective_actors}"
        )
        if total_length % effective_actors != 0:
            chunk_size += 1

        all_data_ref = ray.put(kwargs)

        refs = []
        for chunk_idx in range(effective_actors):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_length)

            for j in range(self.duplicate_actors):
                actor_idx = chunk_idx * self.duplicate_actors + j
                actor = self._actor_handlers[actor_idx]
                refs.append(
                    actor.execute_batch.remote(method_name, all_data_ref, start_idx, end_idx)
                )

        return refs
