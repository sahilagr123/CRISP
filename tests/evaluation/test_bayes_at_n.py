"""Tests for Bayesian pass@N estimation."""
from crisp.evaluation.bayes_at_n import bayesian_pass_at_n


def test_bayesian_pass_at_n_basic():
    """pass@1 with 50% solve rate."""
    result = bayesian_pass_at_n(num_correct=[1, 0, 1], num_samples=[2, 2, 2], n=1)
    assert 0.0 < result < 1.0


def test_bayesian_pass_at_n_perfect():
    """All problems solved perfectly -> pass@1 = 1.0."""
    result = bayesian_pass_at_n(num_correct=[4, 4], num_samples=[4, 4], n=1)
    assert result == 1.0


def test_bayesian_pass_at_n_zero():
    """No problems solved -> pass@1 = 0.0."""
    result = bayesian_pass_at_n(num_correct=[0, 0], num_samples=[4, 4], n=1)
    assert result == 0.0


def test_bayesian_pass_at_n_insufficient_samples():
    """Problems with fewer samples than n are skipped."""
    result = bayesian_pass_at_n(num_correct=[2, 1], num_samples=[4, 1], n=2)
    # Only first problem counted (k=4 >= n=2), second skipped (k=1 < n=2)
    assert result > 0.0

    # All insufficient
    result = bayesian_pass_at_n(num_correct=[1], num_samples=[1], n=5)
    assert result == 0.0


def test_bayesian_pass_at_n_empty():
    """Empty inputs -> 0.0."""
    result = bayesian_pass_at_n(num_correct=[], num_samples=[], n=1)
    assert result == 0.0
