"""Evaluate a trained CRISP player model on math benchmarks.

Supports DAPO-17k, AIME 2024, and AIME 2025 evaluation datasets.

Usage:
    # Evaluate on DAPO-17k (default):
    python scripts/eval_model.py --model ./models/player_hf --n-problems 500

    # Evaluate on AIME 2024 (30 problems):
    python scripts/eval_model.py --model ./models/player_hf --dataset aime24

    # Evaluate on AIME 2025 (30 problems):
    python scripts/eval_model.py --model ./models/player_hf --dataset aime25

    # Evaluate base model as a baseline:
    python scripts/eval_model.py --model Qwen/Qwen3-4B --dataset aime24 --n-samples 16
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from typing import List, Optional

logger = logging.getLogger(__name__)


def _load_problems(dataset: str):
    """Load problems for the requested dataset."""
    if dataset == "aime24":
        from crisp.evaluation.aime import load_aime24_problems
        return load_aime24_problems()
    elif dataset == "aime25":
        from crisp.evaluation.aime import load_aime25_problems
        return load_aime25_problems()
    elif dataset == "dapo":
        from crisp.evaluation.dapo import load_dapo_problems
        return load_dapo_problems()
    else:
        raise ValueError(f"Unknown dataset: {dataset!r} (expected aime24/aime25/dapo)")


def evaluate(
    model_path: str,
    dataset: str = "dapo",
    n_problems: int = 500,
    n_samples: int = 8,
    temperature: float = 0.8,
    max_new_tokens: int = 4096,
    seed: int = 42,
) -> dict:
    """Run evaluation and return results dict."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from crisp.evaluation.bayes_at_n import bayesian_pass_at_n
    from crisp.verifier.answer_extraction import extract_answer
    from crisp.verifier.sympy_verify import check

    # Load problems
    all_problems = _load_problems(dataset)
    rng = random.Random(seed)
    n = min(n_problems, len(all_problems))
    problems = rng.sample(all_problems, n)
    logger.info("Loaded %d problems from %s", n, dataset)

    # Init vLLM
    llm = LLM(
        model=model_path,
        max_model_len=max_new_tokens + 2048,  # prompt headroom
        enforce_eager=True,
        seed=seed,
    )
    tokenizer = llm.get_tokenizer()

    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        seed=seed,
    )

    # Build prompts
    system_prompt = (
        "You are a methodical math solver. Work through problems step by step. "
        "Show your work, then put your final numerical answer within \\boxed{}. "
        "You MUST end your response with \\boxed{} containing your numerical answer."
    )

    all_prompts = []
    prompt_to_problem = []
    for i, prob in enumerate(problems):
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prob.text + "\n\nSolve this problem. "
             "Show your reasoning, then give your final answer inside \\boxed{}."},
        ]
        try:
            token_ids = tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            token_ids = tokenizer.apply_chat_template(
                chat, add_generation_prompt=True,
            )
        for _ in range(n_samples):
            all_prompts.append(token_ids)
            prompt_to_problem.append(i)

    logger.info("Generating %d rollouts (%d problems × %d samples)...",
                len(all_prompts), n, n_samples)

    # Generate
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in all_prompts],
        sampling_params=sampling,
    )

    # Score
    num_correct = [0] * n
    num_total = [0] * n
    for j, out in enumerate(outputs):
        prob_idx = prompt_to_problem[j]
        text = out.outputs[0].text
        answer = extract_answer(text)
        num_total[prob_idx] += 1
        if answer is not None and check(answer, problems[prob_idx].ground_truth):
            num_correct[prob_idx] += 1

    # Metrics
    accuracy = sum(1 for c in num_correct if c > 0) / n if n > 0 else 0.0
    pass_at_1 = bayesian_pass_at_n(num_correct, num_total, n=1)
    mean_solve_rate = sum(c / t for c, t in zip(num_correct, num_total) if t > 0) / n

    results = {
        "model": model_path,
        "dataset": dataset,
        "n_problems": n,
        "n_samples": n_samples,
        "temperature": temperature,
        "accuracy": accuracy,
        "pass_at_1": pass_at_1,
        "mean_solve_rate": mean_solve_rate,
        "per_problem": [
            {"correct": c, "total": t}
            for c, t in zip(num_correct, num_total)
        ],
    }

    return results


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Evaluate CRISP model on math benchmarks")
    parser.add_argument("--model", required=True, help="HF model path or hub ID")
    parser.add_argument("--dataset", type=str, default="dapo",
                        choices=["dapo", "aime24", "aime25"],
                        help="Evaluation dataset (default: dapo)")
    parser.add_argument("--n-problems", type=int, default=500,
                        help="Number of problems (AIME datasets have 30 total)")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON results to file")
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        level=logging.INFO,
    )

    results = evaluate(
        args.model, args.dataset, args.n_problems, args.n_samples,
        args.temperature, args.max_new_tokens, args.seed,
    )

    print(f"\n{'='*50}")
    print(f"Model:          {results['model']}")
    print(f"Dataset:        {results['dataset']}")
    print(f"Problems:       {results['n_problems']}")
    print(f"Samples/prob:   {results['n_samples']}")
    print(f"Temperature:    {results['temperature']}")
    print(f"{'='*50}")
    print(f"Accuracy:       {results['accuracy']:.3f}")
    print(f"Pass@1:         {results['pass_at_1']:.3f}")
    print(f"Mean solve rate:{results['mean_solve_rate']:.3f}")
    print(f"{'='*50}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
