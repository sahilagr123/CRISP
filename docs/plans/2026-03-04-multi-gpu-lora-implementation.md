# Multi-GPU Ray Distribution + LoRA Merge/Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire existing vendored Ray infra into `init_infra()` for multi-GPU training, and add LoRA adapter save + merge/export utilities.

**Architecture:** `DeepSpeedStrategy` gains `offload_states()`, `reload_states()`, `sync_weights()` methods. A new `DistributedStrategy` proxy wraps `RayActorGroup` with the same interface. `init_infra()` branches on `num_gpus_per_node`. Separate `lora_utils.py` handles adapter save (non-destructive) and merge-from-disk (safe).

**Tech Stack:** Python 3.10+, Ray, DeepSpeed, vLLM, PEFT, pytest

---

## Priority Map

| Priority | Task | Blocks |
|----------|------|--------|
| P0 | Task 1: Strategy method extraction | Tasks 2-4 |
| P0 | Task 2: CRISPModelActor | Task 3-4 |
| P0 | Task 3: DistributedStrategy proxy | Task 4 |
| P0 | Task 4: Wire init_infra multi-GPU branch | Complete multi-GPU |
| P1 | Task 5: LoRA utilities | Task 6 |
| P1 | Task 6: CLI integration for LoRA | Complete LoRA |
| P2 | Task 7: Full test run + exports | Verification |

---

### Task 1: Strategy Method Extraction

**Files:**
- Modify: `crisp/infra/strategy.py`
- Modify: `crisp/workflow/main_loop.py`
- Modify: `crisp/workflow/train_step.py`
- Test: `tests/infra/test_strategy.py` (update existing)
- Test: `tests/workflow/test_main_loop.py` (update existing)
- Test: `tests/workflow/test_train_step.py` (update existing)

Currently `main_loop.py` calls `offload_deepspeed_states(ctx.ds_player)` and `train_step.py` calls `broadcast_weights_to_vllm(ctx.ds_player, ...)` — both are free functions that access engine internals. In multi-GPU mode, engines live on remote actors. Solution: move these into strategy methods so `DistributedStrategy` can override them.

**Step 1: Write the failing tests**

Add to `tests/infra/test_strategy.py`:

```python
def test_offload_states_delegates(mock_deepspeed):
    """offload_states() calls offload_deepspeed_states on self."""
    from crisp.infra.strategy import DeepSpeedStrategy
    s = DeepSpeedStrategy()
    s._engine = MagicMock()
    # offload_deepspeed_states checks _is_adam_offload_enabled → needs config
    s._engine.config = {"zero_optimization": {"offload_optimizer": {"device": "none"}}}
    s.offload_states()  # Should not raise


def test_reload_states_delegates(mock_deepspeed):
    """reload_states() calls reload_deepspeed_states on self."""
    from crisp.infra.strategy import DeepSpeedStrategy
    s = DeepSpeedStrategy()
    s._engine = MagicMock()
    s._engine.config = {"zero_optimization": {"offload_optimizer": {"device": "none"}}}
    s.reload_states()  # Should not raise


def test_sync_weights_delegates(mock_deepspeed):
    """sync_weights() calls broadcast_weights_to_vllm."""
    from crisp.infra.strategy import DeepSpeedStrategy
    s = DeepSpeedStrategy(zero_stage=2)
    s._engine = MagicMock()
    mock_engines = [MagicMock()]
    with patch("crisp.infra.strategy.broadcast_weights_to_vllm") as mock_bcast:
        s.sync_weights(mock_engines)
    mock_bcast.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/infra/test_strategy.py -v -k "offload_states or reload_states or sync_weights"`
Expected: FAIL — `AttributeError: 'DeepSpeedStrategy' object has no attribute 'offload_states'`

**Step 3: Write the implementation**

Add to `crisp/infra/strategy.py`, after `optimizer_step()`:

```python
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
```

Then update callers in `crisp/workflow/main_loop.py`:

Replace:
```python
from crisp.infra.deepspeed_strategy import offload_deepspeed_states, reload_deepspeed_states
```
with nothing (remove import).

Replace all occurrences:
- `offload_deepspeed_states(ctx.ds_player)` → `ctx.ds_player.offload_states()`
- `offload_deepspeed_states(ctx.ds_coach)` → `ctx.ds_coach.offload_states()`
- `reload_deepspeed_states(ctx.ds_player)` → `ctx.ds_player.reload_states()`
- `reload_deepspeed_states(ctx.ds_coach)` → `ctx.ds_coach.reload_states()`

In `crisp/workflow/train_step.py`, replace:
```python
from crisp.infra.weight_sync import broadcast_weights_to_vllm
```
with nothing (remove import).

Replace:
```python
    broadcast_weights_to_vllm(
        ctx.ds_player, ctx.player_vllm,
        model_update_group=None,
        zero_stage=ctx.config.infra.zero_stage,
    )
```
with:
```python
    ctx.ds_player.sync_weights(ctx.player_vllm)
```

Same for coach:
```python
    broadcast_weights_to_vllm(
        ctx.ds_coach, ctx.coach_vllm,
        model_update_group=None,
        zero_stage=ctx.config.infra.zero_stage,
    )
```
→
```python
    ctx.ds_coach.sync_weights(ctx.coach_vllm)
```

**Step 4: Update existing tests**

In `tests/workflow/test_main_loop.py`, existing patches for `offload_deepspeed_states` and `reload_deepspeed_states` need updating. The mocks on `ctx.ds_player` and `ctx.ds_coach` (MagicMock objects) already auto-create `.offload_states()` and `.reload_states()` methods, so those tests should pass without changes. But verify the patch targets — remove any patches of `crisp.workflow.main_loop.offload_deepspeed_states` or `crisp.workflow.main_loop.reload_deepspeed_states` since those imports no longer exist.

In `tests/workflow/test_train_step.py`, remove any patches of `crisp.workflow.train_step.broadcast_weights_to_vllm` — the mock ctx already handles `.sync_weights()` via MagicMock.

**Step 5: Run tests**

Run: `pytest tests/infra/test_strategy.py tests/workflow/test_main_loop.py tests/workflow/test_train_step.py -v --tb=short`
Expected: PASS

Run: `pytest tests/ -x --ignore=tests/infra/test_strategy.py -q`
Expected: PASS (all existing tests)

**Step 6: Commit**

```bash
git add crisp/infra/strategy.py crisp/workflow/main_loop.py crisp/workflow/train_step.py tests/
git commit -m "refactor: extract offload/reload/sync_weights methods onto DeepSpeedStrategy"
```

---

### Task 2: CRISPModelActor

**Files:**
- Create: `crisp/infra/distributed.py`
- Test: `tests/infra/test_distributed.py`

A Ray actor class for player/coach training models, mirroring `ReferenceModelActor` from `ray_launcher.py`.

**Step 1: Write the failing tests**

```python
"""Tests for distributed training actors and strategy proxy."""
import sys
from unittest.mock import MagicMock, patch


def test_crisp_model_actor_init():
    """CRISPModelActor.init_model_from_pretrained creates strategy + model."""
    from crisp.infra.distributed import CRISPModelActor

    # CRISPModelActor inherits from BaseModelActor which needs
    # (world_size, rank, master_addr, master_port)
    # For unit tests, we instantiate directly without Ray
    actor = CRISPModelActor.__new__(CRISPModelActor)

    mock_strategy = MagicMock()
    mock_engine = MagicMock()
    mock_strategy.prepare.return_value = mock_engine

    with patch("crisp.infra.distributed.DeepSpeedStrategy", return_value=mock_strategy), \
         patch("crisp.infra.distributed.Actor") as MockActor:
        actor.init_model_from_pretrained(
            strategy_kwargs={"seed": 42, "bf16": True, "zero_stage": 2},
            pretrain="test-model",
            actor_kwargs={"bf16": True},
        )

    assert actor.strategy is mock_strategy
    mock_strategy.prepare.assert_called_once()
    MockActor.assert_called_once_with("test-model", bf16=True)


def test_crisp_model_actor_forward():
    """forward() delegates to strategy.forward()."""
    from crisp.infra.distributed import CRISPModelActor

    actor = CRISPModelActor.__new__(CRISPModelActor)
    actor.strategy = MagicMock()
    actor.strategy.forward.return_value = "logits"

    result = actor.forward("input_ids", attention_mask="mask")
    actor.strategy.forward.assert_called_once_with("input_ids", attention_mask="mask")
    assert result == "logits"


def test_crisp_model_actor_backward_and_step():
    """backward() and optimizer_step() delegate to strategy."""
    from crisp.infra.distributed import CRISPModelActor

    actor = CRISPModelActor.__new__(CRISPModelActor)
    actor.strategy = MagicMock()

    actor.backward("loss")
    actor.strategy.backward.assert_called_once_with("loss")

    actor.optimizer_step()
    actor.strategy.optimizer_step.assert_called_once()


def test_crisp_model_actor_offload_reload():
    """offload_states() and reload_states() delegate to strategy."""
    from crisp.infra.distributed import CRISPModelActor

    actor = CRISPModelActor.__new__(CRISPModelActor)
    actor.strategy = MagicMock()

    actor.offload_states()
    actor.strategy.offload_states.assert_called_once()

    actor.reload_states()
    actor.strategy.reload_states.assert_called_once()


def test_crisp_model_actor_sync_weights():
    """sync_weights() delegates to strategy."""
    from crisp.infra.distributed import CRISPModelActor

    actor = CRISPModelActor.__new__(CRISPModelActor)
    actor.strategy = MagicMock()
    engines = [MagicMock()]

    actor.sync_weights(engines, model_update_group="group")
    actor.strategy.sync_weights.assert_called_once_with(engines, model_update_group="group")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/infra/test_distributed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'crisp.infra.distributed'`

**Step 3: Write the implementation**

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/infra/test_distributed.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add crisp/infra/distributed.py tests/infra/test_distributed.py
git commit -m "feat: add CRISPModelActor for distributed training"
```

---

### Task 3: DistributedStrategy Proxy

**Files:**
- Modify: `crisp/infra/distributed.py`
- Test: `tests/infra/test_distributed.py` (add tests)

The proxy wraps a `RayActorGroup` and exposes the same interface as `DeepSpeedStrategy`. Dispatch rules:
- **Inference** (forward): rank-0 only, return result
- **Training** (backward, optimizer_step): ALL ranks
- **Lifecycle** (offload_states, reload_states): ALL ranks
- **Weight sync** (sync_weights): rank-0 only (it broadcasts via NCCL)

**Step 1: Write the failing tests**

Add to `tests/infra/test_distributed.py`:

```python
def test_distributed_strategy_forward_rank0_only():
    """forward() calls rank-0 actor only and returns its result."""
    from crisp.infra.distributed import DistributedStrategy

    mock_group = MagicMock()
    mock_ref = MagicMock()
    mock_group.async_run_method.return_value = [mock_ref, MagicMock()]

    mock_ray = MagicMock()
    mock_ray.get.return_value = "logits"

    with patch.dict(sys.modules, {"ray": mock_ray}):
        ds = DistributedStrategy(mock_group)
        result = ds.forward("input_ids", attention_mask="mask")

    mock_group.async_run_method.assert_called_once_with("forward", "input_ids", attention_mask="mask")
    # ray.get called with rank-0 ref only
    mock_ray.get.assert_called_once_with(mock_ref)
    assert result == "logits"


def test_distributed_strategy_backward_all_ranks():
    """backward() dispatches to ALL ranks and waits."""
    from crisp.infra.distributed import DistributedStrategy

    mock_group = MagicMock()
    refs = [MagicMock(), MagicMock()]
    mock_group.async_run_method.return_value = refs

    mock_ray = MagicMock()

    with patch.dict(sys.modules, {"ray": mock_ray}):
        ds = DistributedStrategy(mock_group)
        ds.backward("loss")

    mock_group.async_run_method.assert_called_once_with("backward", "loss")
    mock_ray.get.assert_called_once_with(refs)  # ALL refs, not just [0]


def test_distributed_strategy_optimizer_step_all_ranks():
    """optimizer_step() dispatches to ALL ranks."""
    from crisp.infra.distributed import DistributedStrategy

    mock_group = MagicMock()
    refs = [MagicMock(), MagicMock()]
    mock_group.async_run_method.return_value = refs

    mock_ray = MagicMock()

    with patch.dict(sys.modules, {"ray": mock_ray}):
        ds = DistributedStrategy(mock_group)
        ds.optimizer_step()

    mock_group.async_run_method.assert_called_once_with("optimizer_step")
    mock_ray.get.assert_called_once_with(refs)


def test_distributed_strategy_offload_all_ranks():
    """offload_states() dispatches to ALL ranks."""
    from crisp.infra.distributed import DistributedStrategy

    mock_group = MagicMock()
    refs = [MagicMock()]
    mock_group.async_run_method.return_value = refs
    mock_ray = MagicMock()

    with patch.dict(sys.modules, {"ray": mock_ray}):
        ds = DistributedStrategy(mock_group)
        ds.offload_states()

    mock_group.async_run_method.assert_called_once_with("offload_states")
    mock_ray.get.assert_called_once_with(refs)


def test_distributed_strategy_sync_weights_rank0():
    """sync_weights() dispatches to rank-0 actor only."""
    from crisp.infra.distributed import DistributedStrategy

    mock_group = MagicMock()
    rank0_actor = MagicMock()
    mock_group._actor_handlers = [rank0_actor, MagicMock()]

    mock_ray = MagicMock()

    with patch.dict(sys.modules, {"ray": mock_ray}):
        ds = DistributedStrategy(mock_group)
        engines = [MagicMock()]
        ds.sync_weights(engines, model_update_group="grp")

    rank0_actor.sync_weights.remote.assert_called_once_with(engines, model_update_group="grp")
    mock_ray.get.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/infra/test_distributed.py -v -k "distributed_strategy"`
Expected: FAIL — `ImportError: cannot import name 'DistributedStrategy'`

**Step 3: Write the implementation**

Add to `crisp/infra/distributed.py`:

```python
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
        """Checkpoint: rank-0 saves (DeepSpeed handles multi-rank internally)."""
        refs = self._group.async_run_method("save_checkpoint", path, tag, client_state=client_state)
        ray.get(refs)

    def load_checkpoint(self, path, tag=None):
        """Checkpoint: all ranks load."""
        refs = self._group.async_run_method("load_checkpoint", path, tag)
        return ray.get(refs[0])  # Return rank-0's client_state
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/infra/test_distributed.py -v`
Expected: PASS (10 tests)

**Step 5: Commit**

```bash
git add crisp/infra/distributed.py tests/infra/test_distributed.py
git commit -m "feat: add DistributedStrategy proxy for multi-GPU training"
```

---

### Task 4: Wire init_infra Multi-GPU Branch

**Files:**
- Modify: `crisp/train.py:47-164` (init_infra)
- Test: `tests/test_train.py` (add multi-GPU test)

`init_infra()` branches on `config.infra.num_gpus_per_node`: 1 = current local path (unchanged), >1 = create `RayActorGroup` instances wrapping `CRISPModelActor` and `ReferenceModelActor`, wrap in `DistributedStrategy`.

**Step 1: Write the failing test**

Add to `tests/test_train.py`:

```python
def test_init_infra_multi_gpu():
    """init_infra creates RayActorGroup + DistributedStrategy when num_gpus > 1."""
    import sys
    from crisp.train import init_infra
    from crisp.workflow.context import WorkflowContext

    config = CRISPConfig()
    config.infra.num_gpus_per_node = 2

    mock_ray = MagicMock()
    mock_group = MagicMock()
    mock_group.async_init_model_from_pretrained.return_value = [MagicMock()]

    with patch.dict(sys.modules, {"ray": mock_ray}), \
         patch("crisp.infra.vllm_engine.create_vllm_engines",
               return_value=[MagicMock()]), \
         patch("crisp.infra.distributed.RayActorGroup",
               return_value=mock_group) as MockGroup, \
         patch("crisp.infra.distributed.DistributedStrategy") as MockDS:

        ctx = init_infra(config)

    # RayActorGroup created 3 times (player, coach, ref)
    assert MockGroup.call_count == 3
    # DistributedStrategy wraps each group
    assert MockDS.call_count == 3
    assert isinstance(ctx, WorkflowContext)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py::test_init_infra_multi_gpu -v`
Expected: FAIL

**Step 3: Implement**

In `crisp/train.py`, refactor `init_infra()`. After the vLLM creation (line 82), add a branch:

```python
    if icfg.num_gpus_per_node > 1:
        # --- Multi-GPU path ---
        from crisp.infra.distributed import CRISPModelActor, DistributedStrategy
        from crisp.infra.ray_launcher import RayActorGroup, ReferenceModelActor

        strategy_kwargs = dict(
            seed=icfg.seed, bf16=icfg.bf16, zero_stage=icfg.zero_stage,
            adam_offload=icfg.adam_offload, max_norm=icfg.max_norm,
            learning_rate=icfg.learning_rate, weight_decay=icfg.weight_decay,
            micro_train_batch_size=icfg.micro_train_batch_size,
            gradient_checkpointing=icfg.gradient_checkpointing,
            attn_implementation=tcfg.attn_implementation,
            ref_reward_offload=tcfg.ref_reward_offload,
        )

        # Player
        player_group = RayActorGroup(
            num_nodes=icfg.num_nodes, num_gpus_per_node=icfg.num_gpus_per_node,
            ray_actor_type=CRISPModelActor,
        )
        ray.get(player_group.async_init_model_from_pretrained(
            strategy_kwargs=strategy_kwargs, pretrain=tcfg.model_name,
            actor_kwargs=actor_kwargs,
        ))
        ds_player = DistributedStrategy(player_group)

        # Coach
        coach_group = RayActorGroup(
            num_nodes=icfg.num_nodes, num_gpus_per_node=icfg.num_gpus_per_node,
            ray_actor_type=CRISPModelActor,
        )
        ray.get(coach_group.async_init_model_from_pretrained(
            strategy_kwargs=strategy_kwargs, pretrain=tcfg.model_name,
            actor_kwargs=actor_kwargs,
        ))
        ds_coach = DistributedStrategy(coach_group)

        # Reference model (frozen)
        ref_group = RayActorGroup(
            num_nodes=icfg.num_nodes, num_gpus_per_node=icfg.num_gpus_per_node,
            ray_actor_type=CRISPModelActor,
        )
        ray.get(ref_group.async_init_model_from_pretrained(
            strategy_kwargs=strategy_kwargs, pretrain=tcfg.model_name,
            actor_kwargs=actor_kwargs, is_rlhf=True,
        ))
        ref_model = DistributedStrategy(ref_group)

        logger.info("Multi-GPU: player, coach, ref initialized on %d GPUs",
                     icfg.num_gpus_per_node)
    else:
        # --- Single-GPU path (unchanged) ---
        ds_player = _make_strategy()
        ds_player.setup_distributed()
        player_model = Actor(tcfg.model_name, **actor_kwargs)
        ds_player.prepare(player_model)
        logger.info("Player model initialized")

        ds_coach = _make_strategy()
        coach_model = Actor(tcfg.model_name, **actor_kwargs)
        ds_coach.prepare(coach_model)
        logger.info("Coach model initialized")

        ref_strategy = _make_strategy()
        ref_model_actor = Actor(tcfg.model_name, **actor_kwargs)
        ref_strategy.prepare(ref_model_actor, is_rlhf=True)
        ref_model = ref_strategy
        logger.info("Reference model initialized (frozen)")
```

The rest (EMA, rep_buffer, tokenizer, WorkflowContext return) stays the same.

**Step 4: Run tests**

Run: `pytest tests/test_train.py -v --tb=short`
Expected: PASS (all existing + new test)

Run: `pytest tests/ -x --ignore=tests/infra/test_strategy.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add crisp/train.py tests/test_train.py
git commit -m "feat: wire multi-GPU Ray distribution into init_infra"
```

---

### Task 5: LoRA Utilities

**Files:**
- Create: `crisp/infra/lora_utils.py`
- Test: `tests/infra/test_lora_utils.py`

Three functions: `has_lora(strategy)`, `save_lora_adapters(strategy, path)`, `merge_and_save(adapter_path, output_path, base_model_name, tokenizer_name)`.

**Step 1: Write the failing tests**

```python
"""Tests for LoRA adapter save and merge utilities."""
import sys
from unittest.mock import MagicMock, patch, call


def test_has_lora_true():
    """has_lora returns True for a PeftModel."""
    from crisp.infra.lora_utils import has_lora

    mock_peft = MagicMock()
    mock_peft_model_cls = MagicMock()
    strategy = MagicMock()
    strategy.module = MagicMock(spec=[])  # plain mock

    with patch.dict(sys.modules, {"peft": mock_peft}):
        mock_peft.PeftModel = mock_peft_model_cls
        # Make isinstance check return True
        mock_peft_model_cls.__instancecheck__ = lambda self, obj: True
        assert has_lora(strategy) is True


def test_has_lora_false():
    """has_lora returns False for a non-PEFT model."""
    from crisp.infra.lora_utils import has_lora

    mock_peft = MagicMock()
    mock_peft_model_cls = MagicMock()
    strategy = MagicMock()

    with patch.dict(sys.modules, {"peft": mock_peft}):
        mock_peft.PeftModel = mock_peft_model_cls
        mock_peft_model_cls.__instancecheck__ = lambda self, obj: False
        assert has_lora(strategy) is False


def test_has_lora_no_peft():
    """has_lora returns False when peft is not installed."""
    from crisp.infra.lora_utils import has_lora

    strategy = MagicMock()
    with patch.dict(sys.modules, {"peft": None}):
        assert has_lora(strategy) is False


def test_save_lora_adapters():
    """save_lora_adapters calls model.save_pretrained."""
    from crisp.infra.lora_utils import save_lora_adapters

    strategy = MagicMock()
    mock_model = MagicMock()
    strategy.module = mock_model

    with patch("crisp.infra.lora_utils.os.makedirs"):
        save_lora_adapters(strategy, "/tmp/adapters")

    mock_model.save_pretrained.assert_called_once_with("/tmp/adapters")


def test_merge_and_save_loads_from_disk():
    """merge_and_save loads adapters from disk, merges, saves."""
    from crisp.infra.lora_utils import merge_and_save

    mock_peft = MagicMock()
    mock_transformers = MagicMock()
    mock_base_model = MagicMock()
    mock_peft_model = MagicMock()
    mock_merged = MagicMock()

    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_base_model
    mock_peft.PeftModel.from_pretrained.return_value = mock_peft_model
    mock_peft_model.merge_and_unload.return_value = mock_merged

    with patch.dict(sys.modules, {"peft": mock_peft, "transformers": mock_transformers}):
        merge_and_save("/tmp/adapters", "/tmp/merged", "base-model")

    mock_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()
    mock_peft.PeftModel.from_pretrained.assert_called_once_with(mock_base_model, "/tmp/adapters")
    mock_peft_model.merge_and_unload.assert_called_once()
    mock_merged.save_pretrained.assert_called_once_with("/tmp/merged")


def test_merge_and_save_with_tokenizer():
    """merge_and_save saves tokenizer when tokenizer_name provided."""
    from crisp.infra.lora_utils import merge_and_save

    mock_peft = MagicMock()
    mock_transformers = MagicMock()
    mock_peft_model = MagicMock()
    mock_merged = MagicMock()
    mock_tokenizer = MagicMock()

    mock_peft.PeftModel.from_pretrained.return_value = mock_peft_model
    mock_peft_model.merge_and_unload.return_value = mock_merged
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

    with patch.dict(sys.modules, {"peft": mock_peft, "transformers": mock_transformers}):
        merge_and_save("/tmp/adapters", "/tmp/merged", "base-model", tokenizer_name="base-model")

    mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
        "base-model", trust_remote_code=True,
    )
    mock_tokenizer.save_pretrained.assert_called_once_with("/tmp/merged")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/infra/test_lora_utils.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
"""LoRA adapter save and merge utilities for CRISP.

save_lora_adapters: Non-destructive save of adapter weights from live model.
merge_and_save: Load adapters from disk, merge into base model, save full model.
                Never touches the live training model.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import torch


def has_lora(strategy: Any) -> bool:
    """Check if the model has LoRA adapters (is a PeftModel)."""
    try:
        from peft import PeftModel
        return isinstance(strategy.module, PeftModel)
    except (ImportError, AttributeError):
        return False


def save_lora_adapters(strategy: Any, path: str) -> None:
    """Save LoRA adapter weights only. Non-destructive.

    Saves adapter_model.safetensors + adapter_config.json to path.
    """
    os.makedirs(path, exist_ok=True)
    model = strategy.module
    model.save_pretrained(path)


def merge_and_save(
    adapter_path: str,
    output_path: str,
    base_model_name: str,
    tokenizer_name: Optional[str] = None,
) -> None:
    """Load adapters from disk, merge into base model, save full model.

    NEVER operates on the live training model. Safe to call during or
    after training. Can run in a separate process.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_path)

    if tokenizer_name:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        tok.save_pretrained(output_path)
```

**Step 4: Run tests**

Run: `pytest tests/infra/test_lora_utils.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add crisp/infra/lora_utils.py tests/infra/test_lora_utils.py
git commit -m "feat: add LoRA adapter save and merge utilities"
```

---

### Task 6: CLI Integration for LoRA

**Files:**
- Modify: `crisp/train.py` (parse_args, run)
- Test: `tests/test_train.py` (add LoRA CLI tests)

Add `--save-lora` and `--merge-lora` CLI flags. At end of training, save adapters and optionally merge.

**Step 1: Write the failing tests**

Add to `tests/test_train.py`:

```python
def test_parse_args_lora():
    """parse_args handles --save-lora and --merge-lora."""
    args = parse_args(["--config", "f.yaml", "--save-lora", "/tmp/adapters",
                       "--merge-lora", "/tmp/merged"])
    assert args.save_lora == "/tmp/adapters"
    assert args.merge_lora == "/tmp/merged"

    # Defaults to None
    args = parse_args(["--config", "f.yaml"])
    assert args.save_lora is None
    assert args.merge_lora is None


def test_run_saves_lora_at_end():
    """run() calls save_lora_adapters and merge_and_save when flags set."""
    from crisp.train import run

    config = CRISPConfig()
    config.training.num_iterations = 1
    config.training.save_freq = 0

    mock_result = MagicMock()
    mock_result.player_loss = 0.5
    mock_result.coach_loss = None
    mock_result.player_accuracy = 0.7
    mock_result.num_problems = 4
    mock_result.num_discussions = 1

    mock_ctx = MagicMock()
    mock_ctx.iteration = 0

    with patch("crisp.workflow.main_loop.step", return_value=mock_result), \
         patch("crisp.train.init_infra", return_value=mock_ctx), \
         patch("crisp.train.save_lora_adapters") as mock_save, \
         patch("crisp.train.merge_and_save") as mock_merge:
        run(config, save_lora_path="/tmp/adapters", merge_lora_path="/tmp/merged")

    # Player and coach adapters saved
    assert mock_save.call_count == 2
    # Merge called
    mock_merge.assert_called_once()
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `parse_args` doesn't accept `--save-lora`

**Step 3: Implement**

In `parse_args`, add:
```python
    parser.add_argument("--save-lora", type=str, default=None,
                        help="Save LoRA adapters to this path at end of training")
    parser.add_argument("--merge-lora", type=str, default=None,
                        help="Merge LoRA into base model and save to this path")
```

Update `run()` signature:
```python
def run(config: Any, resume_path: Optional[str] = None,
        save_lora_path: Optional[str] = None,
        merge_lora_path: Optional[str] = None) -> None:
```

At end of `run()`, before the final log message, add:
```python
    # LoRA save/merge
    if save_lora_path:
        from crisp.infra.lora_utils import save_lora_adapters, merge_and_save
        save_lora_adapters(ctx.ds_player, os.path.join(save_lora_path, "player"))
        save_lora_adapters(ctx.ds_coach, os.path.join(save_lora_path, "coach"))
        logger.info("LoRA adapters saved to %s", save_lora_path)

        if merge_lora_path:
            merge_and_save(
                os.path.join(save_lora_path, "player"),
                os.path.join(merge_lora_path, "player"),
                tcfg.model_name, tokenizer_name=tcfg.model_name,
            )
            merge_and_save(
                os.path.join(save_lora_path, "coach"),
                os.path.join(merge_lora_path, "coach"),
                tcfg.model_name, tokenizer_name=tcfg.model_name,
            )
            logger.info("Merged models saved to %s", merge_lora_path)
```

Update `main()` to pass new args:
```python
    run(config, resume_path=args.resume,
        save_lora_path=args.save_lora, merge_lora_path=args.merge_lora)
```

**Step 4: Run tests**

Run: `pytest tests/test_train.py -v --tb=short`
Expected: PASS

**Step 5: Commit**

```bash
git add crisp/train.py tests/test_train.py
git commit -m "feat: add --save-lora and --merge-lora CLI flags"
```

---

### Task 7: Exports + Full Test Run

**Files:**
- Modify: `crisp/infra/__init__.py`

**Step 1: Update exports**

Add to `crisp/infra/__init__.py`:
```python
from .distributed import CRISPModelActor, DistributedStrategy
from .lora_utils import has_lora, save_lora_adapters, merge_and_save
```

**Step 2: Run full test suite**

```bash
pytest tests/ -v --tb=short --ignore=tests/infra/test_strategy.py
```

Expected: All tests pass (existing 231 + ~20 new).

**Step 3: Commit**

```bash
git add crisp/infra/__init__.py
git commit -m "feat: export distributed training and LoRA utilities"
```

---

## Verification

After all tasks:

```bash
pytest tests/ -v --tb=short   # All tests pass
```

## What This Unlocks

After these 7 tasks:
- `crisp-train --config configs/multi_gpu.yaml` with `num_gpus_per_node: 4` distributes training across GPUs
- `crisp-train --config configs/example.yaml --save-lora checkpoints/lora` saves adapter weights
- `crisp-train --config configs/example.yaml --save-lora checkpoints/lora --merge-lora checkpoints/merged` saves adapters + merged full model
- Workflow code (`main_loop.py`, `train_step.py`) is unchanged — same interface for single-GPU and multi-GPU
