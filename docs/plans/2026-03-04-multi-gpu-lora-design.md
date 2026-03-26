# Multi-GPU Ray Distribution + LoRA Merge/Export Design

## Goal

Wire the existing vendored Ray infrastructure (RayActorGroup, ReferenceModelActor, etc.) into `init_infra()` for multi-GPU training, and add LoRA adapter save + merge/export utilities.

## Architecture

Two features, both behind the existing `WorkflowContext` interface so workflow code stays unchanged:

1. **Multi-GPU**: `init_infra()` branches on `num_gpus_per_node`. Single-GPU keeps current local path. Multi-GPU creates `RayActorGroup` instances wrapping new `CRISPModelActor` actors, fronted by a `DistributedStrategy` proxy that exposes the same `.forward()/.backward()/.optimizer_step()` interface.

2. **LoRA**: New `lora_utils.py` module with `save_lora_adapters()` (non-destructive, saves adapter weights from live model) and `merge_and_save()` (loads adapters from disk, merges into base model, saves full model — never touches the live training model).

## Part 1: Multi-GPU Ray Distribution

### New file: `crisp/infra/distributed.py`

#### CRISPModelActor

```python
@ray.remote(num_gpus=1)
class CRISPModelActor(BaseModelActor):
    """Ray actor hosting a player or coach model on a single GPU."""

    def init_model_from_pretrained(self, strategy_kwargs, pretrain, actor_kwargs, is_rlhf=False):
        """Create local Strategy + Actor, call prepare()."""
        self.strategy = DeepSpeedStrategy(**strategy_kwargs)
        self._setup_distributed(self.strategy)
        model = Actor(pretrain, **actor_kwargs)
        self.engine = self.strategy.prepare(model, is_rlhf=is_rlhf)

    def forward(self, input_ids, attention_mask=None):
        return self.strategy.forward(input_ids, attention_mask=attention_mask)

    def backward(self, loss):
        self.strategy.backward(loss)

    def optimizer_step(self):
        self.strategy.optimizer_step()

    def save_checkpoint(self, path, tag, client_state=None):
        self.strategy._engine.save_checkpoint(path, tag=tag, client_state=client_state)

    def load_checkpoint(self, path, tag=None):
        return self.strategy._engine.load_checkpoint(path, tag=tag)
```

Mirrors the existing `ReferenceModelActor` pattern from `ray_launcher.py`.

#### DistributedStrategy (Proxy)

```python
class DistributedStrategy:
    """Proxy that wraps a RayActorGroup, exposing the same interface as DeepSpeedStrategy.

    Method dispatch rules:
    - INFERENCE methods (forward, log_probs) → rank-0 only, return result
    - TRAINING methods (backward, optimizer_step) → ALL ranks (ZeRO needs gradients on every shard)
    - PROPERTY access (_engine, module, config) → rank-0 only
    """

    def __init__(self, actor_group):
        self._group = actor_group

    def forward(self, *args, **kwargs):
        """Inference: rank-0 only."""
        refs = self._group.async_run_method("forward", *args, **kwargs)
        return ray.get(refs[0])  # Only need rank-0 result

    def backward(self, loss):
        """Training: ALL ranks (ZeRO shards need gradients)."""
        refs = self._group.async_run_method("backward", loss)
        ray.get(refs)  # Wait for all ranks

    def optimizer_step(self):
        """Training: ALL ranks."""
        refs = self._group.async_run_method("optimizer_step")
        ray.get(refs)

    # For broadcast_weights_to_vllm and offload/reload compatibility
    def __getattr__(self, name):
        """Delegate to rank-0 actor for property access."""
        ...
```

### init_infra() changes

```python
def init_infra(config):
    icfg = config.infra

    if icfg.num_gpus_per_node == 1:
        # Current local path — unchanged
        ...
    else:
        # Multi-GPU path
        player_group = RayActorGroup(
            num_nodes=icfg.num_nodes,
            num_gpus_per_node=icfg.num_gpus_per_node,
            ray_actor_type=CRISPModelActor,
        )
        # Init models on all actors
        ray.get(player_group.async_init_model_from_pretrained(
            strategy_kwargs={...}, pretrain=tcfg.model_name, actor_kwargs={...},
        ))
        ds_player = DistributedStrategy(player_group)

        # Same for coach, ref
        ...

    # WorkflowContext shape is identical either way
    return WorkflowContext(ds_player=ds_player, ...)
```

### Weight sync in multi-GPU mode

`broadcast_weights_to_vllm()` already handles this — it reads `model.module.named_parameters()` from rank-0 and broadcasts to vLLM engines. The `DistributedStrategy.__getattr__` delegation makes `strategy.module` work via rank-0 actor access.

For `offload_deepspeed_states` / `reload_deepspeed_states`: these need to run on ALL ranks (each rank offloads its own shard). The proxy dispatches these to all actors.

## Part 2: LoRA Merge/Export

### New file: `crisp/infra/lora_utils.py`

```python
def has_lora(strategy) -> bool:
    """Check if the model has LoRA adapters."""
    # Check if strategy.module is a PeftModel

def save_lora_adapters(strategy, path: str) -> None:
    """Save LoRA adapter weights only. Non-destructive.

    For ZeRO-3: gathers sharded parameters before saving.
    Saves adapter_model.bin + adapter_config.json to path.
    """
    model = strategy.module  # or _engine.module
    # If ZeRO-3: use GatheredParameters context
    model.save_pretrained(path)  # PEFT saves adapters only

def merge_and_save(
    adapter_path: str,
    output_path: str,
    base_model_name: str,
    tokenizer_name: str = None,
) -> None:
    """Load adapters from disk, merge into base model, save full model.

    NEVER operates on the live training model. Safe to call during or
    after training. Can run in a separate process.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_path)

    if tokenizer_name:
        tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        tok.save_pretrained(output_path)
```

### CLI integration

```python
# train.py parse_args:
parser.add_argument("--save-lora", type=str, default=None, help="Save LoRA adapters to path")
parser.add_argument("--merge-lora", type=str, default=None, help="Merge LoRA and save full model to path")

# End of run():
if save_lora_path:
    save_lora_adapters(ctx.ds_player, save_lora_path)
    save_lora_adapters(ctx.ds_coach, save_lora_path + "/coach")
if merge_lora_path and save_lora_path:
    merge_and_save(save_lora_path, merge_lora_path, tcfg.model_name, tcfg.model_name)
```

## Testing Strategy

All tests mocked (no real GPUs):

**Multi-GPU (`tests/infra/test_distributed.py`):**
- `test_distributed_strategy_forward_rank0_only` — verify forward calls rank-0 actor
- `test_distributed_strategy_backward_all_ranks` — verify backward calls ALL actors
- `test_distributed_strategy_optimizer_step_all_ranks` — verify step calls ALL actors
- `test_init_infra_multi_gpu_creates_ray_groups` — mock Ray, verify RayActorGroup created
- `test_init_infra_single_gpu_unchanged` — verify local path still works

**LoRA (`tests/infra/test_lora_utils.py`):**
- `test_has_lora_true/false` — PeftModel detection
- `test_save_lora_adapters` — mock save_pretrained, verify called
- `test_merge_and_save_loads_from_disk` — mock PeftModel.from_pretrained + merge_and_unload
- `test_merge_and_save_with_tokenizer` — verify tokenizer saved alongside

## Files Changed

| File | Change |
|------|--------|
| `crisp/infra/distributed.py` | NEW: CRISPModelActor + DistributedStrategy |
| `crisp/infra/lora_utils.py` | NEW: save_lora_adapters + merge_and_save |
| `crisp/train.py` | MODIFY: init_infra multi-GPU branch, CLI args, end-of-training save |
| `crisp/infra/__init__.py` | MODIFY: export new classes |
| `tests/infra/test_distributed.py` | NEW: multi-GPU tests |
| `tests/infra/test_lora_utils.py` | NEW: LoRA tests |
