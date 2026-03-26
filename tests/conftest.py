"""Shared test fixtures for CRISP."""
from __future__ import annotations

# Pre-import torch internals and transformers to prevent double-registration
# errors when pytest's assertion rewriter re-execs native extension modules.
# This forces all TORCH_LIBRARY and PyO3 modules to load once before any
# test file triggers them through the assertion rewriter.
import torch  # noqa: F401
import torch._dynamo  # noqa: F401
import transformers  # noqa: F401

# Pre-import ray_launcher so that its module-level `import ray` runs while
# ray is NOT available (ray=None). This prevents later sys.modules["ray"]
# mocks from corrupting _ray_remote_decorator (which wraps classes at
# definition time with ray.remote when ray is not None).
import crisp.infra.ray_launcher  # noqa: F401

import numpy as np
import pytest

from crisp.types import Problem, Rollout, DiscussionResult


@pytest.fixture
def rng():
    """Deterministic random generator for tests."""
    return np.random.default_rng(42)


def make_rollout(
    problem_idx: int = 0,
    player_id: int = 0,
    answer: str | None = "42",
    correct: bool | None = True,
    reward: float = 0.0,
    log_probs: list[float] | None = None,
    text: str = "",
) -> Rollout:
    """Factory for test rollouts."""
    return Rollout(
        problem_idx=problem_idx,
        player_id=player_id,
        tokens=[1, 2, 3],
        text=text or (f"The answer is \\boxed{{{answer}}}" if answer else "No answer here"),
        log_probs=log_probs or [-0.5, -0.3, -0.1],
        answer=answer,
        correct=correct,
        reward=reward,
    )


def make_problem(
    text: str = "What is 6 * 7?",
    ground_truth: str = "42",
    embedding: np.ndarray | None = None,
) -> Problem:
    """Factory for test problems."""
    return Problem(
        text=text,
        ground_truth=ground_truth,
        coach_embedding=embedding if embedding is not None else np.random.default_rng(0).random(384).astype(np.float32),
    )
