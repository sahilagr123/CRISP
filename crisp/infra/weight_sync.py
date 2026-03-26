"""Weight synchronization from DeepSpeed training to vLLM inference.

Extracted from MARTI's ActorPPOTrainer._broadcast_to_vllm().
The key invariant: after broadcast_weights_to_vllm() completes,
vLLM engines serve the EXACT same weights as the DeepSpeed model.
"""
from typing import List

import torch

try:
    import deepspeed
except ImportError:
    deepspeed = None

try:
    import ray
except ImportError:
    ray = None

from .utils import torch_dist_barrier_and_cuda_sync


def _direct_sync_weights(model, vllm_engines: List, zero_stage: int) -> None:
    """Sync weights directly to vLLM without a process group (single-GPU path).

    Saves the full state dict to a temp file, then tells vLLM workers to load
    from that file. This avoids tensor serialization through vLLM V1's msgspec
    IPC (which can't handle torch tensors).

    Handles LoRA models by merging adapters before syncing and unmerging after.
    Uses Actor.model (the HF CausalLM) directly to get clean parameter names
    that match vLLM's expected format.
    """
    import logging
    import os
    import tempfile

    _logger = logging.getLogger(__name__)
    module = model.module  # Actor wrapping HF/Peft model

    # Get the inner HF model (skip Actor wrapper to get clean param names)
    inner_model = module.model

    # Handle LoRA: merge adapter into base weights for sync
    is_peft = False
    try:
        from peft import PeftModel
        is_peft = isinstance(inner_model, PeftModel)
    except ImportError:
        pass

    if is_peft:
        inner_model.merge_adapter()
        source_model = inner_model.get_base_model()
    else:
        source_model = inner_model

    # Collect state dict on CPU with HF-format names
    state_dict = {}
    for name, param in source_model.named_parameters():
        if zero_stage == 3 and deepspeed is not None:
            with deepspeed.zero.GatheredParameters([param]):
                state_dict[name] = param.data.cpu()
        else:
            state_dict[name] = param.data.cpu()

    if is_peft:
        inner_model.unmerge_adapter()

    num_params = len(state_dict)
    _logger.info("Syncing %d parameters to vLLM (file-based)", num_params)

    # Save to temp file (accessible by EngineCore subprocess on same machine)
    fd, path = tempfile.mkstemp(suffix=".pt", prefix="crisp_wsync_")
    os.close(fd)
    try:
        torch.save(state_dict, path)
        _logger.info("Saved %d params to %s, loading into vLLM...", num_params, path)

        refs = [engine.load_weights_from_file.remote(path) for engine in vllm_engines]
        ray.get(refs)
    finally:
        os.unlink(path)

    del state_dict
    torch.cuda.empty_cache()
    _logger.info("Weight sync complete")


def broadcast_weights_to_vllm(
    model,
    vllm_engines: List,
    model_update_group,
    zero_stage: int,
    use_ray: bool = False,
    enable_prefix_caching: bool = False,
):
    """Broadcast all model parameters from DeepSpeed rank 0 to vLLM engines."""
    if model_update_group is None:
        # Single-GPU: use direct weight transfer instead of broadcast.
        _direct_sync_weights(model, vllm_engines, zero_stage)
        return

    cache_reset_refs = []
    if enable_prefix_caching and torch.distributed.get_rank() == 0:
        for engine in vllm_engines:
            cache_reset_refs.append(engine.reset_prefix_cache.remote())

    torch.cuda.empty_cache()
    module = model.module
    count, num_params = 0, len(list(module.named_parameters()))

    for name, param in module.named_parameters():
        count += 1
        with deepspeed.zero.GatheredParameters([param], enabled=zero_stage == 3):
            if torch.distributed.get_rank() == 0:
                shape = param.shape if zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(
                        name, dtype=param.dtype, shape=shape,
                        empty_cache=(count == num_params),
                    )
                    for engine in vllm_engines
                ]
                if use_ray:
                    import ray.util.collective as collective
                    collective.broadcast(param.data, 0, group_name=model_update_group)
                else:
                    model_update_group.broadcast(
                        param.data, src=0, stream=torch.cuda.current_stream()
                    )
                ray.get(refs)

    if cache_reset_refs:
        ray.get(cache_reset_refs)
    torch.cuda.empty_cache()
    torch_dist_barrier_and_cuda_sync()
