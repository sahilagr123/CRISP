# CRISP Design Document

**Date**: 2026-03-03
**Status**: Approved

## Overview

CRISP is a multi-agent reinforcement learning training system where a **coach** generates math problems and two **players** (Alice and Bob) solve them, with an asymmetric **discussion** mechanism triggered by disagreement. All three agents are trained via GRPO (Group Relative Policy Optimization).

**Base framework**: Fork of [MARTI](https://github.com/TsinghuaC3I/MARTI) (Approach A: heavy reuse). Keep MARTI's Ray/vLLM/DeepSpeed infrastructure. Replace trainer, workflow, and reward modules.

**Models**: Qwen3-4B-Instruct (players), Qwen3-14B (coach).

**Hardware**: Modal GPUs for production. Local 4080 Super (16GB) for development/testing.

## Architecture

### Module Structure

```
crisp/
├── config.py                    # Dataclasses for all hyperparams
├── verifier/
│   ├── sympy_verify.py          # 3-strategy verifier (string → numeric → symbolic)
│   │                            # .check(answer, ground_truth) → bool
│   │                            # .equivalent(answer_a, answer_b) → bool (symmetric)
│   └── answer_extraction.py     # Shared \boxed{} parsing (used by all modules)
├── rewards/
│   ├── player_rewards.py        # compute_solve_reward + apply_persuader_bonus
│   ├── coach_rewards.py         # r_uncertainty + r_discussion - r_repetition
│   │                            # Owns intra-batch repetition computation
│   ├── advantages.py            # Two-pool normalization (imports ema_tracker)
│   ├── ema_tracker.py           # EMA mean/variance tracker (η=0.2, init μ=0.5, σ²=0.25)
│   └── repetition_buffer.py    # Sliding window FIFO for cross-batch cosine similarity
├── discussion/
│   ├── trigger.py               # Majority vote + disagreement detection
│   ├── representative.py        # Rollout selection (highest log-prob correct / longest)
│   └── post_discussion.py       # EVALUATION/FINAL ANSWER segment boundary detection
│                                # Delegates \boxed{} extraction to answer_extraction
├── training/
│   ├── grpo_loss.py             # GRPO with DCPO clipping + JS-divergence
│   ├── batch_builder.py         # Dynamic sampling filter + batch assembly
│   └── overlong_shaping.py      # Length penalty for overlong chains
├── evaluation/
│   ├── benchmarks.py            # Stub
│   └── bayes_at_n.py            # Stub
├── workflow/
│   ├── coach_step.py            # Problem generation + self-solve
│   ├── rollout_step.py          # Player rollout generation
│   ├── discussion_step.py       # Trigger + discussion execution
│   └── main_loop.py             # Steps 1-13 orchestration (critical path)
└── infra/                       # Kept/adapted from MARTI
    ├── ray_actors.py
    ├── vllm_engine.py
    └── deepspeed_trainer.py
```

### Key Data Types

```python
@dataclass
class Problem:
    text: str                    # Problem statement
    ground_truth: str            # Coach's \boxed{} answer
    coach_embedding: np.ndarray  # 384-dim MiniLM (for repetition penalty)
    coach_sequence: TokenSequence

@dataclass
class Rollout:
    problem_idx: int
    player_id: int               # 0 = Alice, 1 = Bob
    tokens: List[int]
    text: str                    # Decoded text (for logging + extraction)
    log_probs: List[float]
    answer: Optional[str]        # Extracted \boxed{} content
    correct: Optional[bool]
    reward: float

@dataclass
class DiscussionResult:
    problem_idx: int
    player_id: int
    tokens: List[int]
    text: str
    log_probs: List[float]
    evaluation_text: str         # EVALUATION segment (may be empty)
    final_answer: Optional[str]
    correct: Optional[bool]
    reward: float

@dataclass
class TrainingBatch:
    sequences: List[TokenSequence]
    advantages: List[float]
    ref_log_probs: List[List[float]]  # Reference model log-probs for JS-div
    is_post_discussion: List[bool]
```

## Main Loop Data Flow (Steps 1-13)

```python
def main_loop_iteration(state: TrainingState) -> TrainingState:
    # Step 1: Coach generates problems
    problems = coach_step.generate_batch(coach_model, batch_size=8)

    # Step 2: Players generate rollouts (8 per problem per player)
    rollouts = {pid: rollout_step.generate(model, problems, 8) for pid, model in ...}

    # Step 3: Verify (decode text, extract \boxed{}, check against ground truth)
    # Rollout.text populated during generation; answer_extraction works on text

    # Step 4: Pre-discussion rewards (1.0 / 0.0 / -0.5)

    # Step 5: Majority vote + disagreement detection
    # Uses sympy_verify.equivalent() (symmetric) to compare player majorities

    # Step 6: Discussion (ground truth used for rollout SELECTION only, never in prompts)

    # Step 7: Persuader bonus + post-discussion rewards
    # Persuader = correct majority AND peer actually flipped to correct
    # apply_persuader_bonus asserts single invocation

    # Step 8: Build training batches (dynamic sampling filters in batch_builder)

    # Step 9: Compute advantages (pre-filtered rollouts → Pool 1; discussion → Pool 2 EMA)

    # Step 9.5: Compute reference model log-probs for JS-divergence
    ref_log_probs = vllm_engine.compute_log_probs(reference_model, batch.sequences)

    # Step 10: GRPO gradient update per player

    # Step 10.5: Sync updated weights to vLLM engines
    vllm_engine.sync_weights(player_models[player_id])

    # Step 11: Coach rewards (r_uncertainty + r_discussion - r_repetition)
    # Depends on Step 7 having run (needs discussion outcomes)

    # Step 12: Coach GRPO update (no JS-div, no DCPO)

    # Step 12.5: Sync coach weights to vLLM
    vllm_engine.sync_weights(coach_model)

    # Step 13: Push embeddings to repetition buffer
```

## GRPO Loss with DCPO Clipping

```python
def compute_grpo_loss(
    current_log_probs,  # [B, T]
    old_log_probs,      # [B, T] (rollout policy)
    ref_log_probs,      # [B, T] (frozen reference)
    advantages,         # [B] per-sequence
    attention_mask,     # [B, T]
    dcpo_alpha=3.0, clip_base=0.2, js_beta=0.001
):
    rho = (current_log_probs - old_log_probs).exp()

    # DCPO: prior_prob from REFERENCE policy (not rollout policy)
    prior_prob = ref_log_probs.exp()
    eps = clip_base * (1.0 + dcpo_alpha * (1.0 - prior_prob))

    adv = advantages.unsqueeze(1).expand_as(rho)
    surr1 = rho * adv
    surr2 = rho.clamp(1.0 - eps, 1.0 + eps) * adv
    policy_loss = -torch.min(surr1, surr2)

    # JS-divergence (single-sample estimator — standard approximation)
    p, q = current_log_probs.exp(), ref_log_probs.exp()
    m = 0.5 * (p + q) + 1e-8  # Numerical stability
    js_div = (0.5 * (p * (current_log_probs - m.log()) +
                      q * (ref_log_probs - m.log()))).clamp(min=0.0)

    total_loss = (policy_loss + js_beta * js_div) * attention_mask
    return total_loss.sum() / attention_mask.sum()
```

**Length limits**:
- Pre-discussion: L_max=8192, buffer=2048 (L_hard=10240)
- Post-discussion: L_max=4096, buffer=1024 (L_hard=5120)

## Reward System

### Player rewards
- `r_solve = 1.0` (correct), `0.0` (wrong), `-0.5` (no `\boxed{}`)
- Persuader bonus: `+0.3` to ALL correct rollouts of the persuading player on discussed problems
- Persuader defined as: player with correct majority answer AND peer flipped from wrong to correct

### Coach rewards
```
r_coach(x) = max(0, r_uncertainty + r_discussion - r_repetition)
```
- `r_uncertainty = 1 - 2|p̂ - 0.5|` where p̂ = avg solve rate across both players
- `r_discussion = 0.3` (resolved correctly), `0.15` (occurred but unresolved), `0` (no discussion)
- `r_repetition = intra_batch_penalty + cross_batch_penalty`
  - Intra: `λ * count(sim > τ) / batch_size`
  - Cross: `λ * count(sim > τ) / |H|` (only after buffer full, W=10 batches)
  - τ_sim=0.85, λ=1.0, embeddings from frozen MiniLM-L6-v2 (384-dim)

### Advantage computation
- Pool 1 (pre-discussion, ~50-64 seqs): per-batch mean/std normalization
- Pool 2 (post-discussion, ~3-4 seqs): EMA-smoothed (η=0.2, init μ=0.5, σ²=0.25)
- Dynamic sampling: filter problems where all 8 rollouts have identical rewards (before Pool 1 stats)
- Per-player independent statistics (no cross-player normalization)
- Log warning if Pool 2 empty for 5+ consecutive batches

## Testing Strategy

### Tier 1: Pure-function unit tests (no GPU, CI on every commit)

| Module | Tests |
|--------|-------|
| answer_extraction | Nested braces, LaTeX in box, no box, multiple boxes, empty box |
| sympy_verify | Fractions↔decimals, symbolic equivalence, numeric tolerance, .equivalent() symmetry |
| player_rewards | All-correct, no-box penalty, persuader bonus only on correct, idempotency guard |
| coach_rewards | p̂=0.5 max, p̂=0/1 zero, discussion resolved/unresolved, repetition warm-up |
| advantages | All-same rewards, single rollout, EMA convergence multi-batch |
| ema_tracker | Init values, empty updates, convergence speed |
| repetition_buffer | Warm-up, overflow, identical/orthogonal embeddings, normalization |
| trigger | All-same, tie-break, unanimous, all-different |
| representative | Both correct, both wrong, one each, log-prob vs longest |
| post_discussion | Missing delimiter, multiple delimiters, empty evaluation |
| grpo_loss | Zero advantages, DCPO bounds at extremes, JS numerical stability |
| batch_builder | All-correct filtered, all-wrong filtered, mixed retained, post-discussion kept |
| overlong_shaping | Under L_max, in buffer, at L_hard |
| **Full reward flow** | **Synthetic 8-rollout scenario with hardcoded expected values end-to-end** |

### Tier 2: Integration tests (mock models, pre-Modal launch)
- Full pipeline with canned data: coach → rollouts → verify → rewards → advantages → batch
- Discussion triggers correctly on disagreeing mocks
- Persuader bonus propagates only when peer flips
- **Conservation invariant**: batch size = (non-filtered × 8) + discussion count per player
- No NaN advantages in any sequence
- Weight sync calls happen in correct order

### Tier 3: Smoke tests (real models, GPU required)
- 1 iteration with batch_size=2, rollouts_per_problem=2 on Qwen3-0.6B
- Gradients flow, no NaN/Inf, loss decreases
- vLLM generation produces valid tokens

## Key Design Decisions

1. **Pure-function reward/advantage modules** — testable with synthetic data, no GPU dependency
2. **Verifier is standalone** — reused across coach self-solve, player rollouts, post-discussion
3. **MARTI infra layer is a thin wrapper** — Ray actors, vLLM engine, DeepSpeed trainer kept as-is
4. **Single-sample JS-divergence estimator** — standard approximation per DPH-RL
5. **DCPO uses reference policy probabilities** for clip bounds (not rollout policy)
6. **Repetition buffer only owns cross-batch history** — intra-batch computed in coach_rewards
7. **apply_persuader_bonus asserts single invocation** — prevents silent reward corruption
8. **Dynamic sampling filtering happens in batch_builder** before advantage computation
