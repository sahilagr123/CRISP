# Two-Player Architecture Design

**Date**: 2026-03-12
**Status**: Approved

## Overview

Restore the SystemWalkthrough's two-player design. The original spec called for Alice and Bob as independent agents with independent training, independent advantage normalization (Dr. MAS principle), and per-player JS-divergence against a shared frozen reference. This was never implemented — from the first commit, a single `ds_player` trained on pooled rollouts from both players.

**Hardware**: 4×H200 (141GB each).

## GPU Layout

```
GPU 0: Player vLLM (shared, sequential weight sync per player)
GPU 1: Coach vLLM (dedicated)
GPU 2: Alice DeepSpeed + Bob DeepSpeed + Ref model (parked on CPU)
        ~27GB peak each = ~54GB total, 87GB headroom on 141GB
GPU 3: Coach DeepSpeed (dedicated)
```

Player vLLM is shared — same base architecture, just different trained weights. Weight sync (~1s for 4B model) happens before each player's rollout generation.

## WorkflowContext Changes

Replace single `ds_player` / `player_ema` with per-player instances:

```python
@dataclass
class WorkflowContext:
    player_vllm: List[Any]           # shared vLLM (same base model)
    coach_vllm: Optional[List[Any]]
    ref_model: Any                   # shared frozen ref (one 4B model)
    ds_alice: Any                    # Alice DeepSpeed strategy
    ds_bob: Any                      # Bob DeepSpeed strategy
    ds_coach: Any
    config: CRISPConfig
    alice_ema: EMATracker            # Alice's independent Pool 2 stats
    bob_ema: EMATracker              # Bob's independent Pool 2 stats
    coach_ema: EMATracker
    rep_buffer: RepetitionBuffer
    iteration: int = 0
    pad_token_id: int = 0
    tokenizer: Any = None
    coach_tokenizer: Any = None
    accuracy_history: List[float] = field(default_factory=list)
```

## init_infra Changes (4-GPU Path)

Two player models on GPU 2:

```python
# GPU 2: Alice
ds_alice = _make_strategy()
ds_alice.setup_distributed(dist_backend="gloo")
alice_model = Actor(tcfg.model_name, **actor_kwargs)
ds_alice.prepare(alice_model)
alice_model.gradient_checkpointing_enable()

# GPU 2: Bob (same GPU, independent weights + optimizer)
ds_bob = _make_strategy()
bob_model = Actor(tcfg.model_name, **actor_kwargs)
ds_bob.prepare(bob_model)
bob_model.gradient_checkpointing_enable()

# GPU 2: Ref (parked on CPU, shared by both players)
# ... unchanged from current ...
```

Two EMA trackers:

```python
alice_ema = EMATracker(mu=acfg.ema_init_mu, sigma_sq=acfg.ema_init_sigma_sq, eta=acfg.ema_eta)
bob_ema = EMATracker(mu=acfg.ema_init_mu, sigma_sq=acfg.ema_init_sigma_sq, eta=acfg.ema_eta)
```

## Generation Phase — Sequential Per-Player

Rollouts and discussion become sequential with weight syncs:

```python
# Step 1: Coach generates problems (unchanged)
problems = coach_step.generate_problems(ctx, accuracy_history=ctx.accuracy_history)

# Step 2-4: Sequential per-player rollouts
ctx.ds_alice.sync_weights(ctx.player_vllm)
alice_rollouts = rollout_step.generate_rollouts(ctx, solvable_problems, player_id=0)

ctx.ds_bob.sync_weights(ctx.player_vllm)
bob_rollouts = rollout_step.generate_rollouts(ctx, solvable_problems, player_id=1)

rollouts = {0: alice_rollouts, 1: bob_rollouts}

# Steps 5-6: Discussion — also sequential per player
discussion_results, majority_answers = discussion_step.run_discussion(ctx, rollouts, solvable_problems)
# Internally: sync Alice → Alice discusses → sync Bob → Bob discusses
```

Discussion generation splits per player because each player's discussion response must be generated under that player's weights (importance ratio requires matching old_log_probs).

## Training Phase — Independent Per-Player

### Persuader bonus: called once in main_loop before split

```python
apply_persuader_bonus(rollouts, discussion_results, majority_answers, solvable_problems,
                      gamma=cfg.player.persuader_bonus)
```

This needs both players' data to detect who the persuader was. Called once, modifies rollouts in-place, then per-player training uses the already-modified rollouts.

### Per-player training

```python
alice_loss = train_step.train_player(
    ctx, player_id=0,
    rollouts=rollouts[0],
    discussion_results=discussion_results[0],
    majority_answers=majority_answers,
    problems=solvable_problems,
    ds_model=ctx.ds_alice,
    ema_tracker=ctx.alice_ema,
    sync_weights=False,
)
torch.cuda.empty_cache()

bob_loss = train_step.train_player(
    ctx, player_id=1,
    rollouts=rollouts[1],
    discussion_results=discussion_results[1],
    majority_answers=majority_answers,
    problems=solvable_problems,
    ds_model=ctx.ds_bob,
    ema_tracker=ctx.bob_ema,
    sync_weights=False,
)
```

Key properties:
- Each player's rollouts filtered independently by `filter_dynamic_sampling`
- Each player's advantages computed with their own EMA tracker (Dr. MAS)
- JS-divergence against shared frozen ref model (diversity from independent training signals)
- Ref model forward shared (CPU→GPU→CPU, same as current)
- Weight sync deferred to next iteration's generation phase

### train_player signature change

```python
def train_player(
    ctx: Any,
    player_id: int,
    rollouts: List[Rollout],              # single player's rollouts (was Dict)
    discussion_results: List[DiscussionResult],  # single player's (was Dict)
    majority_answers: Dict[Tuple[int, int], str],
    problems: List[Problem],
    ds_model: Any,                        # player-specific DeepSpeed strategy
    ema_tracker: EMATracker,              # player-specific EMA
    sync_weights: bool = True,
) -> float:
```

Internally: no more `for pid in rollouts` loops. Works on flat lists directly.

## Coach Training — Unchanged

- Coach rewards aggregate both players' rollouts for p_hat (correct per walkthrough)
- Coach JS-div: keep self-referencing with `js_beta=0.001` (production fix, prevents mode collapse)
- Coach DCPO: keep as-is (harmless with self-referencing)

## Checkpointing

Three model subdirectories:

```
checkpoints/ds/
├── alice/     # was player/
├── bob/       # new
└── coach/     # unchanged
```

HF weight export:

```
hf_weights/
├── alice_hf/  # was player_hf/
├── bob_hf/    # new
├── coach_hf/  # unchanged
└── iteration.txt
```

Resume logic loads alice + bob separately. Iteration counter from alice's client_state (arbitrary choice, both are at same iteration).

## H200 Token Limit Increases

With 141GB per GPU, increase token budgets in `dsi_h200.yaml`:

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `player.max_new_tokens` | 8192 | 12288 | ~16GB forward peak per player with GC, two fit in 141GB |
| `coach.coach_solve_max_new_tokens` | 8192 | 12288 | More thinking room, fewer false "unsolvable" |
| `infra.max_model_len` | 32768 | 49152 | Accommodate longer discussion prompts |
| `infra.coach_vllm_max_model_len` | 32768 | 49152 | Room for longer coach self-solve chains |

## Config Changes

No new config fields. H200 token limits handled via YAML overrides.

## Testing Impact

- `test_train_step.py`: Update to use `ds_alice`/`ds_bob` instead of `ds_player`, test per-player training independence
- `test_main_loop.py`: Update step() tests for sequential rollout generation and per-player weight sync
- `test_context.py`: Update WorkflowContext construction
- All other tests: unaffected (domain logic is player-agnostic)

## Summary of Changes by File

| File | Change |
|------|--------|
| `crisp/workflow/context.py` | `ds_player` → `ds_alice`/`ds_bob`, `player_ema` → `alice_ema`/`bob_ema` |
| `crisp/workflow/main_loop.py` | Sequential rollouts with weight sync, persuader bonus moved out of train_player, per-player training calls |
| `crisp/workflow/train_step.py` | `train_player` takes single player's data + explicit ds_model/ema_tracker args |
| `crisp/workflow/discussion_step.py` | Split discussion generation per player with weight sync |
| `crisp/train.py` | `init_infra` creates two player models + two EMAs; checkpoint save/load handles alice/bob/coach; HF weight save/load updated |
| `configs/dsi_h200.yaml` | Token limit increases |
| `scripts/modal_production.py` | Update resume paths for alice/bob |
| Tests | Update mock contexts and assertions |
