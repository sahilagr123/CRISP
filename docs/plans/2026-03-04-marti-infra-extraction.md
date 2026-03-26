# MARTI Infra Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Vendor MARTI's Ray/vLLM/DeepSpeed infrastructure into `crisp/infra/` and wire it to CRISP's existing GRPO training pipeline.

**Architecture:** Copy 8 MARTI source files into `crisp/infra/`, strip all PPO/critic/reward-model code, rename imports from `marti.*` to `crisp.infra.*`, and add a `weight_sync.py` module extracted from MARTI's PPO actor. Build `generate_samples()` in `experience.py` that maps vLLM output directly to CRISP's `Rollout` type. Add `InfraConfig` to `crisp/config.py` for infra hyperparameters.

**Tech Stack:** Ray (>=2.9), vLLM (>=0.8.2), DeepSpeed (>=0.16.4), PyTorch, Transformers, PEFT

---

### Task 1: Add InfraConfig to crisp/config.py

**Files:**
- Modify: `crisp/config.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

In `tests/test_config.py`, add:

```python
def test_infra_config_defaults():
    """InfraConfig provides sensible defaults for single-GPU dev."""
    from crisp.config import InfraConfig

    cfg = InfraConfig()
    assert cfg.zero_stage == 2
    assert cfg.bf16 is True
    assert cfg.num_gpus_per_node == 1
    assert cfg.num_nodes == 1
    assert cfg.vllm_tensor_parallel_size == 1
    assert cfg.vllm_num_engines == 1
    assert cfg.vllm_gpu_memory_utilization == 0.85
    assert cfg.vllm_enable_sleep is True
    assert cfg.adam_offload is False
    assert cfg.gradient_checkpointing is True
    assert cfg.max_model_len == 10240
    assert cfg.micro_train_batch_size == 1
    assert cfg.lora_rank == 0  # 0 means full fine-tuning
    assert cfg.lora_alpha == 16
    assert cfg.seed == 42


def test_infra_config_in_crisp_config():
    """CRISPConfig includes infra sub-config."""
    from crisp.config import CRISPConfig

    cfg = CRISPConfig()
    assert cfg.infra is not None
    assert cfg.infra.zero_stage == 2


def test_infra_config_custom():
    """InfraConfig accepts overrides."""
    from crisp.config import InfraConfig

    cfg = InfraConfig(num_nodes=4, num_gpus_per_node=8, zero_stage=3, lora_rank=64)
    assert cfg.num_nodes == 4
    assert cfg.num_gpus_per_node == 8
    assert cfg.zero_stage == 3
    assert cfg.lora_rank == 64
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_infra_config_defaults -v`
Expected: FAIL with `ImportError: cannot import name 'InfraConfig'`

**Step 3: Write minimal implementation**

Add to `crisp/config.py`:

```python
@dataclass
class InfraConfig:
    """Infrastructure hyperparameters for Ray/vLLM/DeepSpeed."""
    # Distributed
    num_nodes: int = 1
    num_gpus_per_node: int = 1
    seed: int = 42

    # DeepSpeed
    zero_stage: int = 2
    bf16: bool = True
    adam_offload: bool = False
    gradient_checkpointing: bool = True
    micro_train_batch_size: int = 1
    max_norm: float = 1.0
    learning_rate: float = 1e-5
    weight_decay: float = 0.01

    # vLLM
    vllm_num_engines: int = 1
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.85
    vllm_enable_sleep: bool = True
    max_model_len: int = 10240  # L_hard from design doc
    enable_prefix_caching: bool = False

    # LoRA
    lora_rank: int = 0  # 0 = full fine-tuning
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = []
```

Update `CRISPConfig` to include:

```python
@dataclass
class CRISPConfig:
    """Top-level configuration."""
    player: PlayerConfig = None
    coach: CoachConfig = None
    advantage: AdvantageConfig = None
    grpo: GRPOConfig = None
    infra: InfraConfig = None

    def __post_init__(self):
        if self.player is None:
            self.player = PlayerConfig()
        if self.coach is None:
            self.coach = CoachConfig()
        if self.advantage is None:
            self.advantage = AdvantageConfig()
        if self.grpo is None:
            self.grpo = GRPOConfig()
        if self.infra is None:
            self.infra = InfraConfig()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: ALL PASS (old tests + new tests)

**Step 5: Commit**

```bash
git add crisp/config.py tests/test_config.py
git commit -m "feat: add InfraConfig for Ray/vLLM/DeepSpeed hyperparameters"
```

---

### Task 2: Vendor utils.py (distributed utilities)

**Files:**
- Create: `crisp/infra/utils.py`
- Test: `tests/infra/test_utils.py`
- Create: `tests/infra/__init__.py`

**Step 1: Write the failing test**

Create `tests/infra/__init__.py` (empty) and `tests/infra/test_utils.py`:

```python
"""Tests for crisp.infra.utils — no GPU required."""
from unittest.mock import patch, MagicMock
import os


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
    with patch("ray.util.placement_group_table", return_value=pg_table):
        indices = get_bundle_indices(mock_pg, 0, 2)
        assert len(indices) == 2
        indices2 = get_bundle_indices(mock_pg, 1, 2)
        assert len(indices2) == 2
        # All 4 indices should be covered
        assert set(indices + indices2) == {"0", "1", "2", "3"}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/infra/test_utils.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `crisp/infra/utils.py`:

```python
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
    """Create a StatelessProcessGroup with PyNccl for train↔vLLM communication."""
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/infra/test_utils.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crisp/infra/utils.py tests/infra/__init__.py tests/infra/test_utils.py
git commit -m "feat: vendor distributed utilities into crisp/infra/utils.py"
```

---

### Task 3: Vendor deepspeed_strategy.py

**Files:**
- Create: `crisp/infra/deepspeed_strategy.py`
- Test: `tests/infra/test_deepspeed_config.py`

**Step 1: Write the failing test**

Create `tests/infra/test_deepspeed_config.py`:

```python
"""Tests for DeepSpeed config builders — no GPU required."""
from crisp.config import InfraConfig


def test_get_train_ds_config_zero2():
    """ZeRO-2 train config has correct structure."""
    from crisp.infra.deepspeed_strategy import get_train_ds_config

    cfg = get_train_ds_config(stage=2, bf16=True, max_norm=1.0)
    assert cfg["zero_optimization"]["stage"] == 2
    assert cfg["bf16"]["enabled"] is True
    assert cfg["gradient_clipping"] == 1.0
    assert "offload_param" in cfg["zero_optimization"]


def test_get_train_ds_config_zero3():
    """ZeRO-3 train config enables reduce_scatter."""
    from crisp.infra.deepspeed_strategy import get_train_ds_config

    cfg = get_train_ds_config(stage=3, bf16=True, max_norm=1.0)
    assert cfg["zero_optimization"]["stage"] == 3
    assert cfg["zero_optimization"]["reduce_scatter"] is True


def test_get_eval_ds_config():
    """Eval config uses stage 0 for non-ZeRO-3."""
    from crisp.infra.deepspeed_strategy import get_eval_ds_config

    cfg = get_eval_ds_config(offload=False, stage=0, bf16=True)
    assert cfg["zero_optimization"]["stage"] == 0
    assert cfg["bf16"]["enabled"] is True


def test_get_eval_ds_config_offload():
    """Eval config with offload uses CPU."""
    from crisp.infra.deepspeed_strategy import get_eval_ds_config

    cfg = get_eval_ds_config(offload=True, stage=0, bf16=True)
    assert cfg["zero_optimization"]["offload_param"]["device"] == "cpu"


def test_get_optimizer_grouped_parameters():
    """Weight decay excludes bias and norm layers."""
    from crisp.infra.deepspeed_strategy import get_optimizer_grouped_parameters
    from unittest.mock import MagicMock
    import torch

    model = MagicMock()
    # Simulate named_parameters
    param_with_decay = torch.nn.Parameter(torch.randn(10))
    param_bias = torch.nn.Parameter(torch.randn(10))
    param_norm = torch.nn.Parameter(torch.randn(10))

    model.named_parameters.return_value = [
        ("layer.weight", param_with_decay),
        ("layer.bias", param_bias),
        ("layer_norm.weight", param_norm),
    ]

    groups = get_optimizer_grouped_parameters(model, weight_decay=0.01)
    assert len(groups) == 2
    # First group: params with decay (only layer.weight)
    assert len(groups[0]["params"]) == 1
    assert groups[0]["weight_decay"] == 0.01
    # Second group: no decay (bias + norm)
    assert len(groups[1]["params"]) == 2
    assert groups[1]["weight_decay"] == 0.0


def test_offload_reload_noop_when_adam_offload():
    """offload/reload are no-ops when adam_offload is already enabled."""
    from crisp.infra.deepspeed_strategy import offload_deepspeed_states, reload_deepspeed_states
    from unittest.mock import MagicMock

    model = MagicMock()
    model.zero_optimization_stage.return_value = 3
    model.config = {"zero_optimization": {"offload_optimizer": {"device": "cpu"}}}

    # Should return without calling anything on the model
    offload_deepspeed_states(model)
    reload_deepspeed_states(model)
    model.optimizer.offload_states.assert_not_called()
    model.reload_states.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/infra/test_deepspeed_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `crisp/infra/deepspeed_strategy.py`:

```python
"""DeepSpeed configuration and strategy utilities.

Vendored from MARTI (https://github.com/TsinghuaC3I/MARTI).
Contains config builders and optimizer state offload/reload for sleep mode.
The full DeepspeedStrategy class is intentionally NOT vendored — CRISP
uses these building blocks directly in its trainer.
"""
import torch
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def get_train_ds_config(
    offload=False,
    adam_offload=False,
    stage=2,
    bf16=True,
    max_norm=1.0,
    zpg=8,
    grad_accum_dtype=None,
    overlap_comm=False,
):
    """Build DeepSpeed JSON config for training."""
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {
            "device": "cpu" if adam_offload else "none",
            "pin_memory": True,
        },
        "sub_group_size": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "reduce_bucket_size": "auto",
        "zero_hpz_partition_size": zpg,
        "zero_quantized_weights": False,
        "zero_quantized_gradients": False,
    }
    if overlap_comm:
        zero_opt_dict["overlap_comm"] = True
        zero_opt_dict["contiguous_gradients"] = True
    if stage == 3:
        zero_opt_dict["reduce_scatter"] = True

    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {"enabled": bf16},
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "data_types": {"grad_accum_dtype": grad_accum_dtype},
    }


def get_eval_ds_config(offload=False, stage=0, bf16=True):
    """Build DeepSpeed JSON config for evaluation (no optimizer)."""
    zero_opt_dict = {
        "stage": stage,
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "offload_param": {
            "device": "cpu" if offload else "none",
            "pin_memory": True,
        },
    }
    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {"enabled": bf16},
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    no_decay_name_list=None,
):
    """Group parameters into decay/no-decay for AdamW."""
    if no_decay_name_list is None:
        no_decay_name_list = [
            "bias", "layer_norm.weight", "layernorm.weight",
            "norm.weight", "ln_f.weight",
        ]
    return [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_name_list) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay_name_list) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]


def z3_params_to_fetch(param_list):
    """Return ZeRO-3 params that need to be gathered before access."""
    return [
        p for p in param_list
        if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def offload_deepspeed_states(model, pin_memory=True, non_blocking=True):
    """Offload optimizer states to CPU (sleep mode).

    Frees GPU memory so vLLM can use it for inference.
    No-op if adam_offload is already enabled.
    """
    adam_offload = model.config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    if adam_offload:
        return

    import deepspeed
    from packaging import version
    from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum, OffloadStateTypeEnum

    zero_stage = model.zero_optimization_stage()
    if zero_stage != 3 and version.parse(deepspeed.__version__) <= version.parse("0.17.5"):
        raise NotImplementedError(
            "Only ZeRO stage 3 is supported for offload with DeepSpeed <= 0.17.5"
        )

    offload_state_types = [
        OffloadStateTypeEnum.optim_states,
        OffloadStateTypeEnum.contiguous_grad_buffer,
        OffloadStateTypeEnum.hp_params,
    ]
    if version.parse(deepspeed.__version__) >= version.parse("0.16.5"):
        offload_state_types.append(OffloadStateTypeEnum.lp_grads)

    model.optimizer.offload_states(
        include=offload_state_types,
        device=OffloadDeviceEnum.cpu,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
    )
    model.empty_partition_cache()
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    torch.cuda.synchronize()


def reload_deepspeed_states(model, non_blocking=True):
    """Reload optimizer states from CPU back to GPU.

    Called before training phase resumes after vLLM inference.
    No-op if adam_offload is already enabled.
    """
    adam_offload = model.config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    if adam_offload:
        return

    import deepspeed
    from packaging import version

    zero_stage = model.zero_optimization_stage()
    if zero_stage != 3 and version.parse(deepspeed.__version__) <= version.parse("0.17.5"):
        raise NotImplementedError(
            "Only ZeRO stage 3 is supported for reload with DeepSpeed <= 0.17.5"
        )

    model.reload_states(non_blocking=non_blocking)
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    torch.cuda.synchronize()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/infra/test_deepspeed_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crisp/infra/deepspeed_strategy.py tests/infra/test_deepspeed_config.py
git commit -m "feat: vendor DeepSpeed config builders and sleep mode utilities"
```

---

### Task 4: Vendor actor_model.py

**Files:**
- Create: `crisp/infra/actor_model.py`
- Test: `tests/infra/test_actor_model.py`

**Step 1: Write the failing test**

Create `tests/infra/test_actor_model.py`:

```python
"""Tests for Actor model wrapper — no GPU required (tests use mocks)."""
from unittest.mock import patch, MagicMock
import torch


def test_actor_from_existing_model():
    """Actor can wrap an existing nn.Module."""
    from crisp.infra.actor_model import Actor

    mock_model = MagicMock()
    actor = Actor(mock_model)
    assert actor.model is mock_model


def test_actor_gradient_checkpointing():
    """Actor delegates gradient checkpointing to inner model."""
    from crisp.infra.actor_model import Actor

    mock_model = MagicMock()
    actor = Actor(mock_model)
    actor.gradient_checkpointing_enable()
    mock_model.gradient_checkpointing_enable.assert_called_once()
    actor.gradient_checkpointing_disable()
    mock_model.gradient_checkpointing_disable.assert_called_once()


def test_actor_has_use_cache_false():
    """Actor loaded from pretrained disables use_cache."""
    from crisp.infra.actor_model import Actor

    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained:
        mock_model = MagicMock()
        mock_model.config.to_dict.return_value = {}
        mock_from_pretrained.return_value = mock_model

        actor = Actor("fake-model-path", bf16=True)
        assert actor.model.config.use_cache is False


def test_log_probs_from_logits():
    """log_probs_from_logits computes correct values."""
    from crisp.infra.actor_model import log_probs_from_logits

    # Simple 2-token vocab, batch=1, seq=3
    logits = torch.tensor([[[2.0, 1.0], [1.0, 2.0], [0.5, 0.5]]])  # [1, 3, 2]
    labels = torch.tensor([[1, 0, 1]])  # [1, 3]

    log_probs = log_probs_from_logits(logits, labels)
    assert log_probs.shape == (1, 3)
    # Token 1 at position 0: log_softmax([2,1])[1] = 1 - log(e^2 + e^1)
    expected_0 = torch.log_softmax(torch.tensor([2.0, 1.0]), dim=0)[1]
    assert abs(log_probs[0, 0].item() - expected_0.item()) < 1e-5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/infra/test_actor_model.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `crisp/infra/actor_model.py`:

```python
"""Actor model wrapper for HuggingFace CausalLM with LoRA support.

Vendored from MARTI (https://github.com/TsinghuaC3I/MARTI).
Provides model loading, forward pass (log-probs), and gradient checkpointing.
Ring attention removed (not needed for CRISP).
"""
from typing import Optional

import deepspeed
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute per-token log probabilities from logits.

    Args:
        logits: [B, T, V] model output logits
        labels: [B, T] token IDs to evaluate
        temperature: softmax temperature

    Returns:
        [B, T] log probabilities for each label token
    """
    if temperature != 1.0:
        logits = logits / temperature
    log_probs = logits.log_softmax(dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


class Actor(nn.Module):
    """HuggingFace CausalLM wrapper with LoRA and DeepSpeed support.

    Can be initialized from a pretrained model path (str) or an existing nn.Module.
    """

    def __init__(
        self,
        pretrain_or_model,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        attn_implementation="flash_attention_2",
        temperature=1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature

        if isinstance(pretrain_or_model, str):
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "4-bit quantization requires bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            self.model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                from peft import LoraConfig, TaskType, get_peft_model
                from peft.tuners.lora import LoraLayer

                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE support
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                self.model.config.output_router_logits = True
                for m in self.model.modules():
                    if "SparseMoeBlock" in m.__class__.__name__:
                        deepspeed.utils.set_z3_leaf_modules(self.model, [m.__class__])
                        break

            self.model.config.use_cache = False
        else:
            self.model = pretrain_or_model

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
    ) -> torch.Tensor:
        """Compute action log-probs for given sequences.

        Args:
            sequences: [B, S] input token IDs
            action_mask: [B, A] mask for action tokens (response portion)
            attention_mask: [B, S] attention mask

        Returns:
            [B, A] action log-probabilities (if action_mask provided)
            or output dict (if return_output=True and no action_mask)
        """
        rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)
        output["logits"] = output["logits"].to(torch.float32)

        if action_mask is None and return_output:
            return output

        log_probs = log_probs_from_logits(output["logits"], rolled_sequences, temperature=self.temperature)
        log_probs = log_probs[:, :-1]

        if action_mask is None:
            return (log_probs, output) if return_output else log_probs

        action_log_probs = log_probs[:, -action_mask.shape[1]:] * action_mask.float()
        return (action_log_probs, output) if return_output else action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/infra/test_actor_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crisp/infra/actor_model.py tests/infra/test_actor_model.py
git commit -m "feat: vendor Actor model wrapper with LoRA support"
```

---

### Task 5: Vendor vllm_worker_wrap.py

**Files:**
- Create: `crisp/infra/vllm_worker_wrap.py`
- Test: `tests/infra/test_vllm_worker_wrap.py`

**Step 1: Write the failing test**

Create `tests/infra/test_vllm_worker_wrap.py`:

```python
"""Tests for vLLM WorkerWrap — no GPU required (mocked)."""
from unittest.mock import patch, MagicMock, PropertyMock
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
    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=1), \
         patch("ray.util.collective.init_collective_group") as mock_ray:

        wrap.init_process_group(
            master_address="127.0.0.1",
            master_port=12345,
            rank_offset=1,
            world_size=3,
            group_name="test_group",
            backend="nccl",
            use_ray=True,
        )
        mock_ray.assert_called_once()
        assert wrap._model_update_with_ray is True
        assert wrap._model_update_group == "test_group"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/infra/test_vllm_worker_wrap.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `crisp/infra/vllm_worker_wrap.py`:

```python
"""vLLM worker extension for weight synchronization.

Vendored from MARTI (https://github.com/TsinghuaC3I/MARTI).
Injected into vLLM workers via worker_extension_cls to enable
broadcasting updated model weights from DeepSpeed training to
vLLM inference engines.
"""


class WorkerWrap:
    """Extension injected into each vLLM worker process.

    Provides init_process_group() to establish a torch distributed
    group connecting vLLM workers to the DeepSpeed rank-0 process,
    and update_weight()/update_weight_cuda_ipc() to receive
    broadcast weight tensors.
    """

    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl", use_ray=False
    ):
        """Init torch process group for model weights update."""
        import torch
        from crisp.infra.utils import init_process_group

        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_with_ray = use_ray

        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(
                world_size=world_size, rank=rank, backend=backend, group_name=group_name
            )
            self._model_update_group = group_name
        else:
            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        """Receive broadcast weight from DeepSpeed rank 0 via NCCL."""
        import torch

        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, (
            f"dtype mismatch: src {dtype}, dst {self.model_config.dtype}"
        )
        weight = torch.empty(shape, dtype=dtype, device="cuda")

        if self._model_update_with_ray:
            import ray.util.collective as collective

            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            self._model_update_group.broadcast(weight, src=0, stream=torch.cuda.current_stream())

        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
        """Receive weight via CUDA IPC handles (zero-copy, same-node only)."""
        import torch
        from crisp.infra.utils import get_physical_gpu_id

        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, (
            f"dtype mismatch: src {dtype}, dst {self.model_config.dtype}"
        )

        handle = ipc_handles[get_physical_gpu_id()]
        device_id = self.device.index
        func, args = handle
        list_args = list(args)
        list_args[6] = device_id
        weight = func(*list_args)
        self.model_runner.model.load_weights(weights=[(name, weight)])
        torch.cuda.synchronize()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/infra/test_vllm_worker_wrap.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crisp/infra/vllm_worker_wrap.py tests/infra/test_vllm_worker_wrap.py
git commit -m "feat: vendor vLLM WorkerWrap for weight synchronization"
```

---

### Task 6: Vendor vllm_engine.py

**Files:**
- Create: `crisp/infra/vllm_engine.py`
- Test: `tests/infra/test_vllm_engine.py`

**Step 1: Write the failing test**

Create `tests/infra/test_vllm_engine.py`:

```python
"""Tests for vLLM engine wrapper — no GPU required (mocked)."""
from unittest.mock import patch, MagicMock


def test_batch_vllm_engine_call():
    """batch_vllm_engine_call fans out to all engines."""
    from crisp.infra.vllm_engine import batch_vllm_engine_call

    engine1 = MagicMock()
    engine2 = MagicMock()
    engine1.sleep.remote.return_value = "ref1"
    engine2.sleep.remote.return_value = "ref2"

    with patch("ray.get", return_value=["ok", "ok"]) as mock_get, \
         patch("torch.distributed.is_initialized", return_value=False):
        result = batch_vllm_engine_call([engine1, engine2], "sleep")
        engine1.sleep.remote.assert_called_once()
        engine2.sleep.remote.assert_called_once()
        mock_get.assert_called_once_with(["ref1", "ref2"])
        assert result == ["ok", "ok"]


def test_batch_vllm_engine_call_rank0_only():
    """batch_vllm_engine_call skips non-rank-0 when rank_0_only=True."""
    from crisp.infra.vllm_engine import batch_vllm_engine_call

    engine = MagicMock()

    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=1):
        result = batch_vllm_engine_call([engine], "sleep", rank_0_only=True)
        assert result is None
        engine.sleep.remote.assert_not_called()


def test_create_vllm_engines_returns_correct_count():
    """create_vllm_engines creates the requested number of engines."""
    from crisp.infra.vllm_engine import create_vllm_engines

    mock_engine = MagicMock()
    mock_actor_cls = MagicMock()
    mock_actor_cls.options.return_value.remote.return_value = mock_engine

    with patch("ray.get"), \
         patch("ray.util.placement_group.placement_group") as mock_pg:
        mock_pg.return_value.ready.return_value = True

        engines = create_vllm_engines(
            num_engines=3,
            tensor_parallel_size=1,
            pretrain="test-model",
            seed=42,
            full_determinism=False,
            enable_prefix_caching=False,
            enforce_eager=True,
            max_model_len=4096,
            llm_actor_cls=mock_actor_cls,
        )
        assert len(engines) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/infra/test_vllm_engine.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `crisp/infra/vllm_engine.py`:

```python
"""vLLM engine Ray actor and factory.

Vendored from MARTI (https://github.com/TsinghuaC3I/MARTI).
Provides LLMRayActor (wraps vllm.LLM in a Ray actor) and
create_vllm_engines() factory for creating multiple inference engines
with proper placement group scheduling.
"""
import os
import queue
from typing import Any, List

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .utils import get_bundle_indices, ray_noset_visible_devices


class BaseLLMRayActor:
    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        noset_visible_devices = ray_noset_visible_devices()
        if kwargs.get("distributed_executor_backend") == "ray":
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        num_gpus = kwargs.pop("num_gpus")
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))

        self.requests = {}
        self.response_queues = queue.Queue()

        full_determinism = kwargs.pop("full_determinism", False)
        if full_determinism:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        self.kwargs = kwargs

        import vllm
        from packaging import version

        if version.parse(vllm.__version__) >= version.parse("0.9.0"):
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


@ray.remote
class LLMRayActor(BaseLLMRayActor):
    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        import vllm

        self.llm = vllm.LLM(*args, **self.kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc("update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache))

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm.sleep(level=level)

    def wake_up(self):
        self.llm.wake_up()

    def add_requests(self, sampling_params, prompt_token_ids):
        """Tokenize and generate responses."""
        from vllm.inputs import TokensPrompt

        requests = [TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids]
        responses = self.llm.generate(prompts=requests, sampling_params=sampling_params)
        self.response_queues.put(responses)

    def get_responses(self):
        """Return the next batch of generation results."""
        return self.response_queues.get()


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    full_determinism: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
    llm_actor_cls=LLMRayActor,
):
    """Create multiple vLLM inference engine actors.

    Args:
        num_engines: Number of vLLM engine instances to create.
        tensor_parallel_size: TP degree per engine.
        pretrain: HuggingFace model name or path.
        seed: Random seed (incremented per engine).
        full_determinism: Enable deterministic generation.
        enable_prefix_caching: Enable vLLM prefix caching.
        enforce_eager: Disable CUDA graphs.
        max_model_len: Maximum sequence length.
        shared_pg: Existing placement group (for hybrid engine mode).
        gpu_memory_utilization: vLLM GPU memory fraction.
        vllm_enable_sleep: Put engines to sleep after creation.
        llm_actor_cls: Ray actor class (for testing overrides).

    Returns:
        List of Ray actor handles.
    """
    vllm_engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        num_gpus = 0.2

    if not use_hybrid_engine:
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        ray.get(shared_pg.ready())

    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = get_bundle_indices(shared_pg, i, tensor_parallel_size)

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_indices[0] if bundle_indices else i,
        )

        vllm_engines.append(
            llm_actor_cls.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=pretrain,
                enforce_eager=enforce_eager,
                worker_extension_cls="crisp.infra.vllm_worker_wrap.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                dtype="bfloat16",
                trust_remote_code=True,
                full_determinism=full_determinism,
                gpu_memory_utilization=gpu_memory_utilization,
                bundle_indices=bundle_indices,
                num_gpus=0.2 if use_hybrid_engine else 1,
                enable_sleep_mode=vllm_enable_sleep,
            )
        )

    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep")

    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """Fan out a method call to all vLLM engines."""
    import torch

    if torch.distributed.is_initialized():
        if rank_0_only and torch.distributed.get_rank() != 0:
            return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return ray.get(refs)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/infra/test_vllm_engine.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crisp/infra/vllm_engine.py tests/infra/test_vllm_engine.py
git commit -m "feat: vendor vLLM engine actor and factory"
```

---

### Task 7: Vendor ray_launcher.py

**Files:**
- Create: `crisp/infra/ray_launcher.py`
- Test: `tests/infra/test_ray_launcher.py`

**Step 1: Write the failing test**

Create `tests/infra/test_ray_launcher.py`:

```python
"""Tests for Ray launcher — no GPU required (mocked)."""
from unittest.mock import patch, MagicMock


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/infra/test_ray_launcher.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `crisp/infra/ray_launcher.py`:

```python
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
from typing import Dict, Optional, Type

import ray
import torch
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
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


@ray.remote(num_gpus=1)
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
        pg: PlacementGroup = None,
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/infra/test_ray_launcher.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crisp/infra/ray_launcher.py tests/infra/test_ray_launcher.py
git commit -m "feat: vendor Ray launcher with actor group orchestrator"
```

---

### Task 8: Create weight_sync.py (extracted from MARTI PPO actor)

**Files:**
- Create: `crisp/infra/weight_sync.py`
- Test: `tests/infra/test_weight_sync.py`

**Step 1: Write the failing test**

Create `tests/infra/test_weight_sync.py`:

```python
"""Tests for weight sync — no GPU required (mocked)."""
from unittest.mock import patch, MagicMock, call
import torch


def test_broadcast_weights_calls_update_weight_for_each_param():
    """broadcast_weights_to_vllm calls update_weight for every named parameter."""
    from crisp.infra.weight_sync import broadcast_weights_to_vllm

    # Mock model with 2 parameters
    model = MagicMock()
    param1 = torch.nn.Parameter(torch.randn(10, 5))
    param2 = torch.nn.Parameter(torch.randn(3))
    model.module.named_parameters.return_value = [
        ("layer1.weight", param1),
        ("layer1.bias", param2),
    ]

    # Mock vLLM engines
    engine1 = MagicMock()
    engine1.update_weight.remote.return_value = "ref"
    engine2 = MagicMock()
    engine2.update_weight.remote.return_value = "ref"

    with patch("torch.distributed.get_rank", return_value=0), \
         patch("ray.get"), \
         patch("torch.cuda.empty_cache"), \
         patch("crisp.infra.weight_sync.torch_dist_barrier_and_cuda_sync"), \
         patch("torch.cuda.current_stream"):

        broadcast_weights_to_vllm(
            model=model,
            vllm_engines=[engine1, engine2],
            model_update_group=MagicMock(),
            zero_stage=2,
        )

        # Each engine should be called for each parameter
        assert engine1.update_weight.remote.call_count == 2
        assert engine2.update_weight.remote.call_count == 2


def test_broadcast_weights_zero3_uses_gathered_parameters():
    """ZeRO-3 path wraps parameters in GatheredParameters context."""
    from crisp.infra.weight_sync import broadcast_weights_to_vllm

    model = MagicMock()
    param = torch.nn.Parameter(torch.randn(4))
    param.ds_shape = torch.Size([4])
    model.module.named_parameters.return_value = [("w", param)]

    engine = MagicMock()
    engine.update_weight.remote.return_value = "ref"

    with patch("torch.distributed.get_rank", return_value=0), \
         patch("ray.get"), \
         patch("torch.cuda.empty_cache"), \
         patch("crisp.infra.weight_sync.torch_dist_barrier_and_cuda_sync"), \
         patch("torch.cuda.current_stream"), \
         patch("deepspeed.zero.GatheredParameters") as mock_gathered:
        mock_gathered.return_value.__enter__ = MagicMock()
        mock_gathered.return_value.__exit__ = MagicMock(return_value=False)

        broadcast_weights_to_vllm(
            model=model,
            vllm_engines=[engine],
            model_update_group=MagicMock(),
            zero_stage=3,
        )

        # GatheredParameters must be used for ZeRO-3
        mock_gathered.assert_called()


def test_broadcast_weights_skips_on_non_rank0():
    """Non-rank-0 processes only participate in barrier, don't send weights."""
    from crisp.infra.weight_sync import broadcast_weights_to_vllm

    model = MagicMock()
    param = torch.nn.Parameter(torch.randn(4))
    model.module.named_parameters.return_value = [("w", param)]

    engine = MagicMock()

    with patch("torch.distributed.get_rank", return_value=1), \
         patch("ray.get"), \
         patch("torch.cuda.empty_cache"), \
         patch("crisp.infra.weight_sync.torch_dist_barrier_and_cuda_sync"), \
         patch("torch.cuda.current_stream"):

        broadcast_weights_to_vllm(
            model=model,
            vllm_engines=[engine],
            model_update_group=MagicMock(),
            zero_stage=2,
        )

        # Non-rank-0 should NOT call engine.update_weight
        engine.update_weight.remote.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/infra/test_weight_sync.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `crisp/infra/weight_sync.py`:

```python
"""Weight synchronization from DeepSpeed training to vLLM inference.

Extracted from MARTI's ActorPPOTrainer._broadcast_to_vllm().
The key invariant: after broadcast_weights_to_vllm() completes,
vLLM engines serve the EXACT same weights as the DeepSpeed model.
Failure here means rollouts come from a stale policy, and training
silently degrades with no error signal.
"""
from typing import List

import deepspeed
import ray
import torch

from .utils import torch_dist_barrier_and_cuda_sync


def broadcast_weights_to_vllm(
    model,
    vllm_engines: List,
    model_update_group,
    zero_stage: int,
    use_ray: bool = False,
    enable_prefix_caching: bool = False,
):
    """Broadcast all model parameters from DeepSpeed rank 0 to vLLM engines.

    Args:
        model: DeepSpeed model (must have .module attribute).
        vllm_engines: List of LLMRayActor handles.
        model_update_group: Torch process group or Ray collective group name
            connecting DS rank 0 with all vLLM workers.
        zero_stage: DeepSpeed ZeRO stage (2 or 3).
            ZeRO-3 requires GatheredParameters to materialize sharded params.
        use_ray: If True, use Ray collective broadcast instead of NCCL.
        enable_prefix_caching: If True, reset vLLM prefix cache before sync.
    """
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
                        name,
                        dtype=param.dtype,
                        shape=shape,
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/infra/test_weight_sync.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crisp/infra/weight_sync.py tests/infra/test_weight_sync.py
git commit -m "feat: extract weight sync from MARTI PPO actor into standalone module"
```

---

### Task 9: Create experience.py (vLLM output → CRISP types)

**Files:**
- Create: `crisp/infra/experience.py`
- Test: `tests/infra/test_experience.py`

**Step 1: Write the failing test**

Create `tests/infra/test_experience.py`:

```python
"""Tests for experience.py — vLLM output to CRISP Rollout mapping."""
from unittest.mock import MagicMock, patch
import torch


def _make_mock_vllm_output(prompt_token_ids, output_token_ids, logprobs=None):
    """Helper to create a mock vLLM CompletionOutput."""
    output = MagicMock()
    output.prompt_token_ids = prompt_token_ids
    mock_completion = MagicMock()
    mock_completion.token_ids = output_token_ids
    if logprobs is not None:
        mock_logprobs = []
        for tid, lp in zip(output_token_ids, logprobs):
            mock_entry = MagicMock()
            mock_entry.logprob = lp
            mock_logprobs.append({tid: mock_entry})
        mock_completion.logprobs = mock_logprobs
    else:
        mock_completion.logprobs = None
    output.outputs = [mock_completion]
    return output


def test_map_vllm_output_to_rollout():
    """Single vLLM output maps to a Rollout with correct fields."""
    from crisp.infra.experience import map_vllm_output_to_rollout

    output = _make_mock_vllm_output(
        prompt_token_ids=[1, 2, 3],
        output_token_ids=[10, 20, 30],
    )

    rollout = map_vllm_output_to_rollout(
        output, problem_idx=0, player_id=1
    )

    assert rollout.problem_idx == 0
    assert rollout.player_id == 1
    assert rollout.tokens == [1, 2, 3, 10, 20, 30]
    assert len(rollout.log_probs) == 6  # padded with zeros for prompt
    assert rollout.text == ""  # text set later by caller
    assert rollout.reward == 0.0


def test_map_vllm_output_with_logprobs():
    """When vLLM returns logprobs, they're captured in the Rollout."""
    from crisp.infra.experience import map_vllm_output_to_rollout

    output = _make_mock_vllm_output(
        prompt_token_ids=[1, 2],
        output_token_ids=[10, 20],
        logprobs=[-0.5, -1.2],
    )

    rollout = map_vllm_output_to_rollout(output, problem_idx=0, player_id=0)
    # prompt log_probs are 0, response log_probs from vLLM
    assert rollout.log_probs[:2] == [0.0, 0.0]
    assert abs(rollout.log_probs[2] - (-0.5)) < 1e-6
    assert abs(rollout.log_probs[3] - (-1.2)) < 1e-6


def test_generate_samples_distributes_across_engines():
    """generate_samples distributes prompts across vLLM engines evenly."""
    from crisp.infra.experience import generate_samples

    engine1 = MagicMock()
    engine2 = MagicMock()

    output1 = _make_mock_vllm_output([1, 2], [10, 20])
    output2 = _make_mock_vllm_output([3, 4], [30, 40])

    engine1.add_requests.remote.return_value = "ref1"
    engine2.add_requests.remote.return_value = "ref2"
    engine1.get_responses.remote.return_value = "resp_ref1"
    engine2.get_responses.remote.return_value = "resp_ref2"

    with patch("ray.get") as mock_ray_get:
        # First ray.get: wait for add_requests
        # Second ray.get: get responses
        mock_ray_get.side_effect = [
            None,  # ray.get(add_requests refs)
            [[output1], [output2]],  # ray.get(get_responses refs)
        ]

        rollouts = generate_samples(
            vllm_engines=[engine1, engine2],
            prompt_token_ids=[[1, 2], [3, 4]],
            problem_indices=[0, 1],
            player_id=0,
            max_new_tokens=1024,
        )

        assert len(rollouts) == 2
        assert rollouts[0].problem_idx == 0
        assert rollouts[1].problem_idx == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/infra/test_experience.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `crisp/infra/experience.py`:

```python
"""Maps vLLM generation output to CRISP Rollout objects.

This module is the ONLY place where vLLM output types convert to CRISP types.
All downstream code (rewards, discussion, training) works exclusively with
Rollout, DiscussionResult, and TrainingBatch from crisp.types.
"""
from typing import List, Optional

import ray

from crisp.types import Rollout


def map_vllm_output_to_rollout(
    vllm_output,
    problem_idx: int,
    player_id: int,
) -> Rollout:
    """Convert a single vLLM CompletionOutput to a CRISP Rollout.

    Args:
        vllm_output: vLLM RequestOutput object.
        problem_idx: Index of the problem this rollout is for.
        player_id: 0 = Alice, 1 = Bob.

    Returns:
        Rollout with tokens and log_probs populated.
        text, answer, correct, reward are left at defaults — set by caller.
    """
    prompt_ids = list(vllm_output.prompt_token_ids)
    response_ids = list(vllm_output.outputs[0].token_ids)
    tokens = prompt_ids + response_ids

    # Build log_probs: zeros for prompt, vLLM logprobs for response
    prompt_log_probs = [0.0] * len(prompt_ids)
    if vllm_output.outputs[0].logprobs is not None:
        response_log_probs = []
        for i, logprob_dict in enumerate(vllm_output.outputs[0].logprobs):
            token_id = response_ids[i]
            response_log_probs.append(logprob_dict[token_id].logprob)
    else:
        response_log_probs = [0.0] * len(response_ids)

    log_probs = prompt_log_probs + response_log_probs

    return Rollout(
        problem_idx=problem_idx,
        player_id=player_id,
        tokens=tokens,
        text="",  # Decoded by caller (needs tokenizer)
        log_probs=log_probs,
    )


def generate_samples(
    vllm_engines: List,
    prompt_token_ids: List[List[int]],
    problem_indices: List[int],
    player_id: int,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
) -> List[Rollout]:
    """Generate rollouts using vLLM engines and return CRISP Rollout objects.

    Distributes prompts across engines evenly, collects results,
    and maps each to a Rollout.

    Args:
        vllm_engines: List of LLMRayActor handles.
        prompt_token_ids: List of tokenized prompts.
        problem_indices: Problem index for each prompt (parallel to prompt_token_ids).
        player_id: 0 = Alice, 1 = Bob.
        max_new_tokens: Maximum response length.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling (-1 = disabled).

    Returns:
        List of Rollout objects (one per prompt).
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        min_tokens=1,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
        logprobs=1,
    )

    num_engines = len(vllm_engines)
    batch_size = (len(prompt_token_ids) + num_engines - 1) // num_engines

    # Distribute prompts across engines
    refs = []
    for i, engine in enumerate(vllm_engines):
        batch = prompt_token_ids[i * batch_size : (i + 1) * batch_size]
        if batch:
            refs.append(
                engine.add_requests.remote(
                    sampling_params=sampling_params,
                    prompt_token_ids=batch,
                )
            )
    ray.get(refs)

    # Collect responses
    response_refs = [engine.get_responses.remote() for engine in vllm_engines]
    all_outputs = sum(ray.get(response_refs), [])

    # Map to Rollouts
    rollouts = []
    for i, output in enumerate(all_outputs):
        rollout = map_vllm_output_to_rollout(
            output,
            problem_idx=problem_indices[i],
            player_id=player_id,
        )
        rollouts.append(rollout)

    return rollouts
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/infra/test_experience.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crisp/infra/experience.py tests/infra/test_experience.py
git commit -m "feat: add experience.py mapping vLLM output to CRISP Rollout"
```

---

### Task 10: Update crisp/infra/__init__.py with public API

**Files:**
- Modify: `crisp/infra/__init__.py`

**Step 1: Write the failing test**

Add to `tests/infra/test_utils.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/infra/test_utils.py::test_infra_public_api -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `crisp/infra/__init__.py`:

```python
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
from .weight_sync import broadcast_weights_to_vllm
from .experience import generate_samples, map_vllm_output_to_rollout
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/infra/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add crisp/infra/__init__.py tests/infra/test_utils.py
git commit -m "feat: expose crisp.infra public API"
```

---

### Task 11: Run full test suite and verify no regressions

**Files:** None (verification only)

**Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: ALL 137+ tests PASS (original + new infra tests)

**Step 2: Check imports don't break without GPU dependencies**

Run: `python -c "from crisp.config import InfraConfig; print('InfraConfig OK')"`
Expected: prints "InfraConfig OK"

**Step 3: Tag the release**

```bash
git tag v0.2.0-infra-vendored
```

---

### Task 12: Update pyproject.toml infra dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update dependency versions**

Update the `[project.optional-dependencies]` infra section to match MARTI's tested versions:

```toml
infra = [
    "ray>=2.9",
    "vllm>=0.8.2",
    "deepspeed>=0.16.4",
    "peft>=0.7",
    "packaging",
]
```

**Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "chore: update infra dependency version bounds to match MARTI"
```

---

## Summary

| Task | What it builds | Files |
|------|---------------|-------|
| 1 | InfraConfig in config.py | config.py, test_config.py |
| 2 | utils.py (distributed utils) | infra/utils.py, test_utils.py |
| 3 | deepspeed_strategy.py | infra/deepspeed_strategy.py, test_deepspeed_config.py |
| 4 | actor_model.py (HF wrapper) | infra/actor_model.py, test_actor_model.py |
| 5 | vllm_worker_wrap.py | infra/vllm_worker_wrap.py, test_vllm_worker_wrap.py |
| 6 | vllm_engine.py | infra/vllm_engine.py, test_vllm_engine.py |
| 7 | ray_launcher.py | infra/ray_launcher.py, test_ray_launcher.py |
| 8 | weight_sync.py | infra/weight_sync.py, test_weight_sync.py |
| 9 | experience.py (vLLM→Rollout) | infra/experience.py, test_experience.py |
| 10 | __init__.py public API | infra/__init__.py |
| 11 | Full test suite verification | — |
| 12 | pyproject.toml dependencies | pyproject.toml |
