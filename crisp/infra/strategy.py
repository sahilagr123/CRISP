"""DeepSpeed training strategy for CRISP.

Provides the unified interface consumed by:
- train_step.py: forward(), backward(), optimizer_step()
- weight_sync.py: module, config (via engine delegation)
- deepspeed_strategy.py sleep utils: config, zero_optimization_stage(), optimizer
- ray_launcher.py: setup_distributed(), is_rank_0(), prepare(), args
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist

from .deepspeed_strategy import (
    get_eval_ds_config,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
)


@dataclass
class StrategyArgs:
    """Arguments accessible as strategy.args.*"""
    bf16: bool = True
    attn_implementation: str = "eager"
    ref_reward_offload: bool = False
    gradient_checkpointing: bool = True


class DeepSpeedStrategy:
    """Wraps DeepSpeed engine initialization and training operations.

    After prepare() is called, this object delegates attribute access
    for engine-level properties (config, module, optimizer, etc.) to
    the underlying DeepSpeed engine, so it can be passed directly to
    offload_deepspeed_states() and broadcast_weights_to_vllm().
    """

    def __init__(
        self,
        seed: int = 42,
        bf16: bool = True,
        zero_stage: int = 2,
        adam_offload: bool = False,
        max_norm: float = 1.0,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        micro_train_batch_size: int = 1,
        gradient_checkpointing: bool = True,
        attn_implementation: str = "eager",
        ref_reward_offload: bool = False,
    ):
        self.args = StrategyArgs(
            bf16=bf16,
            attn_implementation=attn_implementation,
            ref_reward_offload=ref_reward_offload,
            gradient_checkpointing=gradient_checkpointing,
        )
        self._seed = seed
        self._bf16 = bf16
        self._zero_stage = zero_stage
        self._adam_offload = adam_offload
        self._max_norm = max_norm
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._micro_train_batch_size = micro_train_batch_size
        self._engine: Optional[Any] = None

    def setup_distributed(self, dist_backend: str = "nccl") -> None:
        """Initialize torch.distributed process group via DeepSpeed.

        Args:
            dist_backend: Backend for torch.distributed. Use 'gloo' when
                training models on different GPUs in a single process
                (4-GPU mode) since NCCL binds to one device.
        """
        import os

        import deepspeed

        # Set env vars for single-node so DeepSpeed skips MPI discovery
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        deepspeed.init_distributed(dist_backend=dist_backend)
        torch.manual_seed(self._seed)

    def is_rank_0(self) -> bool:
        """Check if current process is rank 0."""
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True

    def prepare(self, model: Any, is_rlhf: bool = False) -> Any:
        """Wrap model with DeepSpeed engine.

        Returns the DeepSpeed engine. Also stores it internally so that
        forward/backward/optimizer_step and __getattr__ delegation work.

        Parameters
        ----------
        model : nn.Module
            The model to wrap.
        is_rlhf : bool
            If True, initialize without optimizer (for frozen ref model).
        """
        import deepspeed

        if is_rlhf:
            # Frozen ref model: no optimizer needed, use stage 0
            ds_config = get_eval_ds_config(
                stage=0,
                bf16=self._bf16,
            )
            engine, *_ = deepspeed.initialize(model=model, config=ds_config)
        else:
            ds_config = get_train_ds_config(
                adam_offload=self._adam_offload,
                stage=self._zero_stage,
                bf16=self._bf16,
                max_norm=self._max_norm,
            )
            optim_groups = get_optimizer_grouped_parameters(
                model, self._weight_decay,
            )
            if self._adam_offload:
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                optimizer = DeepSpeedCPUAdam(optim_groups, lr=self._learning_rate)
            else:
                optimizer = torch.optim.AdamW(optim_groups, lr=self._learning_rate)
            engine, *_ = deepspeed.initialize(
                model=model, optimizer=optimizer, config=ds_config,
            )

        if (
            self.args.gradient_checkpointing
            and hasattr(model, "gradient_checkpointing_enable")
        ):
            model.gradient_checkpointing_enable()

        self._engine = engine
        return engine

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the wrapped model."""
        return self._engine(*args, **kwargs)

    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass through DeepSpeed engine."""
        self._engine.backward(loss)

    def optimizer_step(self) -> None:
        """Optimizer step through DeepSpeed engine."""
        self._engine.step()

    def offload_states(self, **kwargs) -> None:
        """Offload optimizer/parameter states to CPU (sleep mode)."""
        from .deepspeed_strategy import offload_deepspeed_states
        offload_deepspeed_states(self, **kwargs)

    def reload_states(self, **kwargs) -> None:
        """Reload optimizer/parameter states from CPU (wake mode)."""
        from .deepspeed_strategy import reload_deepspeed_states
        reload_deepspeed_states(self, **kwargs)

    def sync_weights(self, vllm_engines, model_update_group=None, **kwargs) -> None:
        """Broadcast weights from this model to vLLM engines."""
        from .weight_sync import broadcast_weights_to_vllm
        broadcast_weights_to_vllm(
            self, vllm_engines, model_update_group,
            zero_stage=self._zero_stage, **kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying DeepSpeed engine.

        This makes strategy.config, strategy.module, strategy.optimizer,
        strategy.zero_optimization_stage(), strategy.reload_states() etc.
        all work transparently for sleep-mode and weight-sync utilities.
        """
        if name.startswith("_") or name == "args":
            raise AttributeError(name)
        engine = object.__getattribute__(self, "_engine")
        if engine is not None:
            return getattr(engine, name)
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}' "
            f"(engine not initialized — call prepare() first)"
        )
