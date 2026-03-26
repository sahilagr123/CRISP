"""DeepSpeed config builders and sleep-mode utilities for CRISP training."""
from __future__ import annotations

from typing import Any, List, Optional

NO_DECAY_NAME_LIST = [
    "bias",
    "layer_norm.weight",
    "layernorm.weight",
    "norm.weight",
    "ln_f.weight",
]


def get_train_ds_config(
    offload: bool = False,
    adam_offload: bool = False,
    stage: int = 2,
    bf16: bool = True,
    max_norm: float = 1.0,
    zpg: int = 8,
    grad_accum_dtype: Optional[str] = None,
    overlap_comm: bool = False,
) -> dict[str, Any]:
    """Build a DeepSpeed JSON config dict for training.

    Parameters
    ----------
    offload : bool
        Whether to offload parameters to CPU.
    adam_offload : bool
        Whether to offload the Adam optimizer state to CPU.
    stage : int
        ZeRO optimization stage (0, 1, 2, or 3).
    bf16 : bool
        Enable bfloat16 mixed precision.
    max_norm : float
        Gradient clipping max norm.
    zpg : int
        ZeRO partition group size.
    grad_accum_dtype : str or None
        Data type for gradient accumulation (e.g. "fp32").
    overlap_comm : bool
        Overlap communication with computation.
    """
    zero_opt: dict[str, Any] = {
        "stage": stage,
        "offload_param": {
            "device": "cpu" if offload else "none",
        },
        "offload_optimizer": {
            "device": "cpu" if adam_offload else "none",
        },
        "overlap_comm": overlap_comm,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "zero_quantized_weights": False,
        "zero_hpz_partition_size": zpg,
        "zero_quantized_gradients": False,
    }

    if stage == 3:
        zero_opt["reduce_scatter"] = True

    if grad_accum_dtype is not None:
        zero_opt["accumulate_grads_in_fp32"] = grad_accum_dtype == "fp32"

    cfg: dict[str, Any] = {
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 10,
        "zero_optimization": zero_opt,
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }

    if bf16:
        cfg["bf16"] = {"enabled": True}
    else:
        cfg["fp16"] = {"enabled": True}

    return cfg


def get_eval_ds_config(
    offload: bool = False,
    stage: int = 0,
    bf16: bool = True,
) -> dict[str, Any]:
    """Build a DeepSpeed JSON config dict for evaluation (no optimizer).

    Parameters
    ----------
    offload : bool
        Whether to offload parameters to CPU.
    stage : int
        ZeRO optimization stage.
    bf16 : bool
        Enable bfloat16 mixed precision.
    """
    zero_opt: dict[str, Any] = {
        "stage": stage,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
    }

    if offload:
        zero_opt["offload_param"] = {"device": "cpu"}

    cfg: dict[str, Any] = {
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 10,
        "zero_optimization": zero_opt,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }

    if bf16:
        cfg["bf16"] = {"enabled": True}
    else:
        cfg["fp16"] = {"enabled": True}

    return cfg


def get_optimizer_grouped_parameters(
    model: Any,
    weight_decay: float,
    no_decay_name_list: Optional[List[str]] = None,
) -> list[dict[str, Any]]:
    """Group model parameters into decay and no-decay buckets.

    Parameters that match any name in *no_decay_name_list* (by substring)
    receive ``weight_decay=0.0``; the rest receive *weight_decay*.

    Parameters
    ----------
    model
        A PyTorch model (or any object with ``named_parameters()``).
    weight_decay : float
        Weight decay coefficient for the decay group.
    no_decay_name_list : list[str] or None
        Substrings that identify parameters exempt from weight decay.
        Defaults to :data:`NO_DECAY_NAME_LIST`.
    """
    if no_decay_name_list is None:
        no_decay_name_list = NO_DECAY_NAME_LIST

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_name_list):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def z3_params_to_fetch(param_list: list) -> list:
    """Return the subset of *param_list* that need ZeRO-3 gather.

    Only parameters whose ``ds_status`` is ``NOT_AVAILABLE`` are returned.
    This import is deferred so the module loads even without deepspeed.
    """
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    return [
        p
        for p in param_list
        if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def _is_adam_offload_enabled(model: Any) -> bool:
    """Check whether the model's config already offloads the optimizer to CPU."""
    try:
        cfg = model.config
        device = cfg["zero_optimization"]["offload_optimizer"]["device"]
        return device == "cpu"
    except (KeyError, TypeError, AttributeError):
        return False


def offload_deepspeed_states(
    model: Any,
    pin_memory: bool = True,
    non_blocking: bool = True,
) -> None:
    """Offload model parameters and optimizer states to CPU.

    Supports ZeRO stages 2 and 3. No-op when adam_offload is already enabled.
    """
    if _is_adam_offload_enabled(model):
        return

    import torch

    try:
        stage = model.zero_optimization_stage()
    except (AttributeError, TypeError):
        return

    if stage == 3 and hasattr(model.optimizer, "offload_states"):
        model.optimizer.offload_states(
            include="all",
            device="cpu",
            pin_memory=pin_memory,
            non_blocking=non_blocking,
        )
    elif stage <= 2:
        # ZeRO-2/1/0: manually move model params and optimizer states to CPU
        for param in model.module.parameters():
            param.data = param.data.cpu()
            if param.grad is not None:
                param.grad = param.grad.cpu()
        # Move optimizer state tensors (skip DummyOptim used by frozen models)
        if hasattr(model, 'optimizer') and hasattr(model.optimizer, 'state'):
            for state in model.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def reload_deepspeed_states(
    model: Any,
    non_blocking: bool = True,
) -> None:
    """Reload model parameters and optimizer states to GPU.

    Supports ZeRO stages 2 and 3. No-op when adam_offload is already enabled.
    """
    if _is_adam_offload_enabled(model):
        return

    import torch

    try:
        stage = model.zero_optimization_stage()
    except (AttributeError, TypeError):
        return

    if stage == 3 and hasattr(model, "reload_states"):
        model.reload_states(non_blocking=non_blocking)
    elif stage <= 2:
        device = torch.device("cuda")
        for param in model.module.parameters():
            param.data = param.data.to(device, non_blocking=non_blocking)
            if param.grad is not None:
                param.grad = param.grad.to(device, non_blocking=non_blocking)
        if hasattr(model, 'optimizer') and hasattr(model.optimizer, 'state'):
            for state in model.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device, non_blocking=non_blocking)
