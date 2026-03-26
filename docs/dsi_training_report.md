# DSI Cluster Training Report

**CRISP: Multi-Agent RL for Math Reasoning**
Qwen3-4B-Instruct (Alice + Bob) + Qwen3-14B (Coach) on 4×H200

---

## Early Runs — March 12-13 (jobs 757877–762439)

### Jobs 757877, 759258, 760526 — Failed to start
Infrastructure issues during initial cluster setup. Zero iterations completed.

### Job 759526 — March 12 (10 iterations)
First run that actually trained. Reached iter 10 before the job ended. Bob already showed instability: **bob_loss=7.86** at iter 10. This early warning of loss spikes was a precursor to the -9.3 spike we'd see later.

### Job 760702 — March 12 (11 iterations)
Similar early run, 11 iterations. More stable than 759526 (bob_loss=-0.67 at iter 10) but still short.

### Job 762439 — March 13 (44 iterations — longest early run)
The longest run before we began systematic debugging. Reached **44 iterations** over ~11.5 hours. Accuracy settled at 57.8% by iter 44. This run predated most of our fixes — no SIGTERM handling, no save_freq tuning, original hyperparameters. The weights from this run were not evaluated.

---

## Systematic Debugging Phase — March 14-23

### Run A — March 14 (job 764340, 23 iterations)

**Config:** js_beta=0.001, MAX_ADV=50, lr=1e-5, save_freq=20

**What happened:** Training collapsed catastrophically at iteration 6. During warmup (iter 1-5), the coach generated trivially easy AMC 10 problems — 100% accuracy, zero reward variance, zero GRPO signal. The model learned nothing. At iteration 6, the coach jumped to AMC 12/AIME difficulty. Accuracy plummeted from 94% to 38% in one step. With MAX_ADV=50 and js_beta=0.001, the resulting gradient updates were massive and irreversible. By iteration 10, Alice was producing word salad. By iteration 20, complete gibberish.

The job ran for the full 12 hours but the SIGTERM handler never fired — Slurm killed it before the handler could save. Only the periodic save at iter 20 survived, containing a destroyed model.

**Sample output (iter 20):** *"rded Flash have confidence interact, program next if] .. view peers trusted accuracy pri"*

**Fixes applied:**
- js_beta: 0.001 → 0.01 (stronger reference anchoring)
- MAX_ADV: 50 → 5 (tighter advantage clamping)
- Added `#SBATCH --signal=B:TERM@600` for graceful shutdown
- Added bash SIGTERM trap to forward signal to Python child process

### Run B — March 15 (job 765426, 31 iterations)

**Config:** js_beta=0.01, MAX_ADV=5, lr=1e-5, save_freq=20

**What happened:** No collapse! The difficulty transition at iter 6 was smooth (accuracy 85% → 77%). Losses stayed small. Accuracy stabilized at 35-50%. The model produced coherent math reasoning throughout all 31 iterations. SIGTERM handler did not fire (the `--signal` fix wasn't pulled before this run). Saved at iter 20.

But when we loaded the saved weights for evaluation: **complete gibberish**. Pass@1 = 0.000.

This launched an extensive debugging investigation across 9 diagnostic scripts (`check_output.py`, `check_weights.py`, `check_config.py`, `check_index.py`, `check_tokenizer.py`, `check_base_vs_saved.py`, `check_safetensors_swap.py`, `check_manual_save.py`, `check_weights_full.py`). We systematically isolated every component:

- Swapped config.json between base and saved → still broken
- Swapped tokenizer between base and saved → weights were the problem
- Compared all 398 tensors → max diff was 1 bf16 ULP (0.000244)
- Manually rebuilt safetensors from raw tensor data → still broken
- Compared safetensors index files → identical

We discovered two separate issues:

1. **`use_cache=False` in saved config.json.** The Actor sets `model.config.use_cache = False` for training efficiency. `save_pretrained()` persisted this to config.json. When vLLM loaded the saved model, `use_cache=False` broke autoregressive generation, producing word salad.

2. **Training updates too small for bf16 precision.** With js_beta=0.01 and MAX_ADV=5, the regularization was so conservative that weight updates over 20 iterations were smaller than the bf16 unit of least precision (0.000244). The saved weights were indistinguishable from the base model — all 398 tensors differed by at most 1 bf16 ULP, which was just optimizer round-trip noise, not actual learning.

**Fixes applied:**
- Set `use_cache=True` before `save_pretrained()`, restore after
- Relaxed regularization: js_beta 0.01 → 0.005, MAX_ADV 5 → 10
- Removed save_freq=20 override (yaml default is 10)

### Run C — March 17 (job 766651, 24 iterations)

**Config:** js_beta=0.005, MAX_ADV=10, lr=1e-5, Dr. GRPO (mean-only advantages), asymmetric clip (clip_low=0.2, clip_high=0.28)

**What happened:** Three new problems emerged:

1. **Bob loss spike: -9.3 at iter 7.** The Dr. GRPO mean-only normalization (advantages = r - mean, no std division) made advantages scale-dependent. Combined with asymmetric DCPO clipping, a single batch produced a catastrophic loss value that damaged Bob's weights.

2. **Alice learned to literally output `\boxed{answer}`.** The system prompt said "You MUST end your response with \boxed{answer}" — Alice took this literally, outputting the word "answer" in the box instead of a number. This received reward=0.0 (wrong answer, not missing-box penalty of -0.5), providing insufficient negative signal.

3. **Coach collapsed at iter 23.** The coach uses self-referencing (its own log probs as the JS-divergence reference), with a weak js_beta=0.005. Without a frozen reference model to anchor against, the coach drifted freely across iterations. By iter 23, it was outputting its own system prompt as the problem text.

**Eval:** 11.1% pass@1 (baseline: 49.7%). Model degraded significantly.

**Fixes applied:**
- Restored std normalization with MIN_SIGMA=0.5 floor (prevents loss spikes while dampening volatile advantages)
- Added separate `coach_js_beta=0.1` (20x stronger than player's 0.005)
- Changed prompt: "end with \boxed{answer}" → "end with \boxed{} containing your numerical answer"

### Infrastructure failures — March 19-20 (jobs 770871, 771626)

**Job 770871:** Ray worker crashed with "Failed to register worker to Raylet: IOError: End of file" due to stale `/tmp/ray` from a previous job. Fix: added `rm -rf /tmp/ray ... || true` cleanup.

**Job 771626:** The `rm -rf` cleanup command (without `|| true`) returned non-zero on permission-denied files in `/tmp`. With `set -euo pipefail`, this killed the entire script with no error message in the log. Fix: added `|| true`.

### Run D — March 20 (job 771629, 26 iterations)

**Config:** js_beta=0.005, coach_js_beta=0.1, MAX_ADV=10, lr=1e-5, std normalization with floor, use_cache fix, SIGTERM trap, prompt fix

**What happened:** All stability fixes worked. No loss spikes (bob_loss peaked at 0.73, far from the -9.3 of Run C). Coach held together through all 26 iterations — no system prompt leaking. SIGTERM handler fired correctly and saved at iter 25. No `\boxed{answer}` literal outputs.

However, accuracy still degraded steadily. By iter 20+, Alice's outputs showed word salad: *"philosophical consistency based on no-case presence of realization and near-total inaccuracy."*

**Eval:** 2.2% pass@1 — even worse than Run C's 11.1%, and far below the 49.7% baseline.

**Root cause identified:** Literature review of DAPO, M-GRPO, GTPO, and SDRL revealed that every successful GRPO math training paper uses **lr=1e-6**. Our lr=1e-5 was 10x too high. At this learning rate, each gradient step overshoots and erodes pre-trained capabilities — especially with our small batch size (64 rollouts vs DAPO's 8192).

| Paper | Learning Rate | Batch Size |
|-------|-------------|-----------|
| DAPO | 1e-6 | 512×16 = 8192 |
| M-GRPO | 1e-6 | — |
| GTPO | 1e-6 | — |
| SDRL (multi-agent) | 1e-6 | 256×8 = 2048 |
| CRISP | 1e-5 | 8×8 = 64 |

**Fix applied:** lr: 1e-5 → 1e-6

---

## Complete Run Log

| Job | Date | Iters | Outcome |
|-----|------|-------|---------|
| 757877 | Mar 12 | 0 | Failed to start (infra) |
| 759258 | Mar 12 | 0 | Failed to start (infra) |
| 759526 | Mar 12 | 10 | Early run; bob_loss=7.86 at iter 10 |
| 760526 | Mar 12 | 0 | Failed to start (infra) |
| 760702 | Mar 12 | 11 | Early run; more stable |
| 762439 | Mar 13 | 44 | Longest early run; accuracy 57.8% at iter 44 |
| 764340 | Mar 14 | 23 | **Run A:** Catastrophic collapse at iter 6 |
| 765426 | Mar 15 | 31 | **Run B:** Stable training, broken save (use_cache bug) |
| 766651 | Mar 17 | 24 | **Run C:** Bob spike, coach collapse, \boxed{answer} literal |
| 770871 | Mar 19 | 0 | Ray startup failure (stale /tmp) |
| 771626 | Mar 20 | 0 | Silent death (rm -rf + set -e) |
| 771629 | Mar 20 | 26 | **Run D:** Stable but lr too high; 2.2% pass@1 |

---

## Summary of All Changes

| Parameter | Run A | Run B | Run C | Run D | Next |
|-----------|-------|-------|-------|-------|------|
| lr | 1e-5 | 1e-5 | 1e-5 | 1e-5 | **1e-6** |
| js_beta | 0.001 | 0.01 | 0.005 | 0.005 | 0.005 |
| coach_js_beta | 0.001 | 0.01 | 0.005 | **0.1** | 0.1 |
| MAX_ADV | 50 | 5 | 10 | 10 | 10 |
| Advantage norm | mean/std | mean/std | mean-only | **std w/ floor** | std w/ floor |
| Clipping | sym 0.2 | sym 0.2 | **asym 0.2/0.28** | asym 0.2/0.28 | asym 0.2/0.28 |
| save_freq | 20 | 20 | **10** | 10 | 10 |
| use_cache fix | no | no | no | **yes** | yes |
| SIGTERM trap | no | no | no | **yes** | yes |
| Prompt fix | no | no | no | no | **yes** |
| Player model | Instruct | Instruct | Instruct | Instruct | **Base (planned)** |

---

## Key Lessons

1. **Learning rate is king.** 10x too high → model destruction. Every GRPO paper uses 1e-6.
2. **Start from base, not instruct.** Instruct models already have 50% math accuracy — hard to improve, easy to destroy. All DAPO/M-GRPO/SDRL papers train from base models.
3. **Save what you mean.** Training-only config (use_cache=False) silently corrupts saved models for inference.
4. **Coach self-referencing is fragile.** Without a frozen reference or strong js_beta, the coach drifts freely.
5. **Prompts are code.** "End with \boxed{answer}" is ambiguous to a language model under RL pressure.
6. **`set -e` and `rm -rf` don't mix.** Permission-denied on shared /tmp kills scripts silently.
7. **SIGTERM needs explicit forwarding.** Slurm signals the shell, not the child process. Bash must trap and forward.
8. **bf16 precision sets a floor.** Training updates smaller than 1 ULP (0.000244) are invisible after save/load.
9. **Small batch sizes need small learning rates.** With 64 rollouts (vs DAPO's 8192), advantage estimates are noisy — compensate with slower steps.
