"""CRISP infrastructure layer.

Vendored from MARTI (https://github.com/TsinghuaC3I/MARTI) with
PPO-specific code removed. Provides Ray/vLLM/DeepSpeed wrappers
for distributed training and inference.
"""
from .deepspeed_strategy import (
    get_train_ds_config,
    get_eval_ds_config,
    get_optimizer_grouped_parameters,
    offload_deepspeed_states,
    reload_deepspeed_states,
)
from .strategy import DeepSpeedStrategy
from .lora_utils import has_lora, save_lora_adapters, merge_and_save
from .weight_sync import broadcast_weights_to_vllm

# Lazy imports — distributed requires ray which may not be available
def __getattr__(name):
    if name in ("CRISPModelActor", "DistributedStrategy"):
        from .distributed import CRISPModelActor, DistributedStrategy
        return {"CRISPModelActor": CRISPModelActor, "DistributedStrategy": DistributedStrategy}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
from .experience import generate_samples, map_vllm_output_to_rollout
