"""Tests for benchmark evaluation."""
from unittest.mock import MagicMock, patch

from crisp.types import Problem


def test_evaluate_on_problems_computes_accuracy():
    """evaluate_on_problems returns accuracy from verified answers."""
    from crisp.evaluation.benchmarks import evaluate_on_problems

    problems = [
        Problem(text="What is 2+2?", ground_truth="4"),
        Problem(text="What is 3+3?", ground_truth="6"),
    ]

    tok = MagicMock()
    tok.encode.side_effect = lambda text, add_special_tokens=True: [1, 2, 3]

    # Mock rollouts: problem 0 gets correct, problem 1 gets wrong
    mock_rollout_0 = MagicMock()
    mock_rollout_0.text = "The answer is \\boxed{4}"
    mock_rollout_0.problem_idx = 0

    mock_rollout_1 = MagicMock()
    mock_rollout_1.text = "The answer is \\boxed{5}"
    mock_rollout_1.problem_idx = 1

    with patch("crisp.infra.experience.generate_samples",
               return_value=[mock_rollout_0, mock_rollout_1]):
        result = evaluate_on_problems(problems, [MagicMock()], tok, n_samples=1)

    assert result["accuracy"] == 0.5  # 1 out of 2 problems correct
    assert result["num_correct"] == [1, 0]
    assert result["num_total"] == [1, 1]


def test_evaluate_on_problems_empty():
    """evaluate_on_problems handles empty problem list."""
    from crisp.evaluation.benchmarks import evaluate_on_problems

    with patch("crisp.infra.experience.generate_samples", return_value=[]):
        result = evaluate_on_problems([], [MagicMock()], MagicMock(), n_samples=1)

    assert result["accuracy"] == 0.0
