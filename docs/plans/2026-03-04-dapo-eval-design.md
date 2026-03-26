# DAPO-17k Evaluation + Separate Model Configs Design

**Date**: 2026-03-04
**Status**: Approved

## Overview

Three changes: (1) separate player/coach model names in config, (2) DAPO-17k dataset loader for evaluation, (3) wire `run_evaluation` to actually run benchmarks.

## Config Changes

`TrainingConfig` defaults change:
- `model_name`: `"Qwen/Qwen3-4B-Instruct"` (players + reference model)
- `coach_model_name`: `"Qwen/Qwen3-14B"` (coach). `None` falls back to `model_name`.
- `eval_dataset`: `"dapo"` (dataset identifier)
- `eval_n_problems`: `100` (problems sampled per eval run)

`init_infra` uses `coach_model_name` for coach vLLM engines and DeepSpeed strategy. Player vLLM, player DeepSpeed, and reference model all use `model_name`.

## DAPO-17k Loader

New `crisp/evaluation/dapo.py`:
- `load_dapo_problems(max_problems=None) -> List[Problem]`
- Uses `datasets.load_dataset("open-r1/DAPO-Math-17k-Processed", "en")["train"]`
- Maps `row["prompt"]` → `Problem.text`, `row["solution"]` → `Problem.ground_truth`
- DAPO answers are always integers — existing `check()` handles this via string/numeric match
- `datasets` added to `[project.optional-dependencies] eval`

## Evaluation Wiring

`run_evaluation(ctx)` in `train.py`:
1. Load DAPO problems (cached in module-level variable after first load)
2. Sample `eval_n_problems` deterministically (seeded by `ctx.iteration`)
3. Call `evaluate_on_problems(problems, ctx.player_vllm, ctx.tokenizer, n_samples=ctx.config.training.eval_n_samples)`
4. Compute `bayesian_pass_at_n(num_correct, num_total, n=1)`
5. Log accuracy and pass@1
