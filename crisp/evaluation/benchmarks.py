"""Benchmark evaluation for CRISP."""
from __future__ import annotations

from typing import Any, Dict, List

from crisp.types import Problem
from crisp.verifier.answer_extraction import extract_answer
from crisp.verifier.sympy_verify import check


def evaluate_on_problems(
    problems: List[Problem],
    vllm_engines: List[Any],
    tokenizer: Any,
    n_samples: int = 8,
) -> Dict[str, float]:
    """Evaluate model on a set of problems, return accuracy metrics.

    Generates n_samples rollouts per problem via vLLM, verifies answers,
    and returns accuracy and per-problem correct/total counts.
    """
    from crisp.infra.experience import generate_samples

    prompts = [tokenizer.encode(p.text, add_special_tokens=True) for p in problems]
    # Repeat each prompt n_samples times
    all_prompts = []
    all_indices = []
    for i, p in enumerate(prompts):
        for _ in range(n_samples):
            all_prompts.append(p)
            all_indices.append(i)

    rollouts = generate_samples(vllm_engines, all_prompts, all_indices, player_id=0)

    # Score
    num_correct = [0] * len(problems)
    num_total = [0] * len(problems)
    for r in rollouts:
        answer = extract_answer(r.text)
        num_total[r.problem_idx] += 1
        if answer is not None and check(answer, problems[r.problem_idx].ground_truth):
            num_correct[r.problem_idx] += 1

    accuracy = sum(1 for c in num_correct if c > 0) / len(problems) if problems else 0.0
    return {
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": num_total,
    }
