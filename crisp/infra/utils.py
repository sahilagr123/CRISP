"""Distributed utilities for CRISP infrastructure.

Vendored from MARTI (https://github.com/TsinghuaC3I/MARTI) with
PPO-specific code removed. Provides process group initialization,
GPU detection, and synchronization helpers.
"""
import os
from datetime import timedelta
from typing import Any, Optional, Union

import torch
import torch.distributed


def torch_dist_barrier_and_cuda_sync():
    """Synchronize distributed training and CUDA operations."""
    torch.distributed.barrier()
    torch.cuda.synchronize()


def ray_noset_visible_devices(env_vars=os.environ):
    """Check if Ray is configured to not set *_VISIBLE_DEVICES."""
    NOSET_ENV_VARS = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(v) for v in NOSET_ENV_VARS)


def get_bundle_indices(placement_group, index, length):
    """Get bundle indices for a placement group segment.

    Groups bundles by node to ensure colocated bundles are contiguous,
    then returns the slice for the given index.
    Workaround for https://github.com/ray-project/ray/issues/51117
    """
    import ray

    pg_infos = ray.util.placement_group_table(placement_group)
    node_id_to_bundles = {}
    for bundle, node_id in pg_infos["bundles_to_node_id"].items():
        node_id_to_bundles.setdefault(node_id, []).append(bundle)

    sorted_bundle_indices = sum(node_id_to_bundles.values(), [])
    return sorted_bundle_indices[index * length : (index + 1) * length]


def get_physical_gpu_id():
    """Return the UUID string of the current CUDA device."""
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """Create a StatelessProcessGroup with PyNccl for train<->vLLM communication."""
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


def init_process_group(
    backend: Union[str, object] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store=None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    """Create a torch process group (allows multiple main groups).

    Adapted from PyTorch's init_process_group to support creating
    multiple process groups with different group names.
    """
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0
        assert rank >= 0
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        store = PrefixStore(group_name, store)

    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg
