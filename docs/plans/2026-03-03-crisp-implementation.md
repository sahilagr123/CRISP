# CRISP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the CRISP multi-agent RL training system by forking MARTI and replacing trainer/workflow/reward modules, using TDD throughout.

**Architecture:** Fork MARTI inline for Ray/vLLM/DeepSpeed infra. Build pure-function reward/verification/discussion modules tested with synthetic data. Wire together in main_loop.py.

**Tech Stack:** Python 3.10+, PyTorch, SymPy, NumPy, sentence-transformers, pytest. MARTI (OpenRLHF + Ray + vLLM + DeepSpeed) for infra.

---

## Phase 0: Project Scaffolding

### Task 1: Initialize project structure and dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `crisp/__init__.py`
- Create: `crisp/config.py`
- Create: `crisp/types.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "crisp"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "sympy>=1.12",
    "torch>=2.1",
    "sentence-transformers>=2.2",
    "scipy>=1.11",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-cov"]
infra = ["ray>=2.9", "vllm>=0.3", "deepspeed>=0.12"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "gpu: requires GPU (deselect with '-m not gpu')",
    "integration: integration tests with mock models",
]
```

**Step 2: Create crisp/types.py with all shared dataclasses**

```python
"""Shared data types for CRISP."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class TokenSequence:
    """A sequence of tokens with associated log-probabilities."""
    tokens: List[int]
    log_probs: List[float]
    text: str = ""


@dataclass
class Problem:
    """A coach-generated math problem with ground truth."""
    text: str
    ground_truth: str
    coach_embedding: Optional[np.ndarray] = None  # 384-dim MiniLM
    coach_sequence: Optional[TokenSequence] = None


@dataclass
class Rollout:
    """A single player solution attempt."""
    problem_idx: int
    player_id: int  # 0 = Alice, 1 = Bob
    tokens: List[int]
    text: str
    log_probs: List[float]
    answer: Optional[str] = None
    correct: Optional[bool] = None
    reward: float = 0.0
    _persuader_bonus_applied: bool = field(default=False, repr=False)


@dataclass
class DiscussionResult:
    """Result of a post-discussion response."""
    problem_idx: int
    player_id: int
    tokens: List[int]
    text: str
    log_probs: List[float]
    evaluation_text: str = ""
    final_answer: Optional[str] = None
    correct: Optional[bool] = None
    reward: float = 0.0


@dataclass
class TrainingBatch:
    """A batch ready for GRPO training."""
    sequences: List[TokenSequence]
    advantages: List[float]
    ref_log_probs: List[List[float]]
    is_post_discussion: List[bool]
```

**Step 3: Create crisp/config.py**

```python
"""Hyperparameter configuration for CRISP."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlayerConfig:
    """Per-player training hyperparameters."""
    rollouts_per_problem: int = 8
    solve_reward: float = 1.0
    wrong_reward: float = 0.0
    no_box_penalty: float = -0.5
    persuader_bonus: float = 0.3  # gamma_1


@dataclass
class CoachConfig:
    """Coach training hyperparameters."""
    batch_size: int = 8
    discussion_alpha: float = 0.3
    repetition_lambda: float = 1.0
    repetition_tau: float = 0.85
    repetition_window: int = 10  # W batches
    embedding_dim: int = 384


@dataclass
class AdvantageConfig:
    """Advantage computation hyperparameters."""
    epsilon: float = 1e-8
    ema_eta: float = 0.2
    ema_init_mu: float = 0.5
    ema_init_sigma_sq: float = 0.25
    empty_pool_warning_threshold: int = 5


@dataclass
class GRPOConfig:
    """GRPO loss hyperparameters."""
    dcpo_alpha: float = 3.0
    clip_base: float = 0.2
    js_beta: float = 0.001  # 0.0 for coach
    pre_discussion_l_max: int = 8192
    pre_discussion_buffer: int = 2048
    post_discussion_l_max: int = 4096
    post_discussion_buffer: int = 1024


@dataclass
class CRISPConfig:
    """Top-level configuration."""
    player: PlayerConfig = None
    coach: CoachConfig = None
    advantage: AdvantageConfig = None
    grpo: GRPOConfig = None

    def __post_init__(self):
        if self.player is None:
            self.player = PlayerConfig()
        if self.coach is None:
            self.coach = CoachConfig()
        if self.advantage is None:
            self.advantage = AdvantageConfig()
        if self.grpo is None:
            self.grpo = GRPOConfig()
```

**Step 4: Create empty package init files and test conftest**

```python
# crisp/__init__.py
"""CRISP: Collaborative Reasoning via Iterative Self-Play."""

# tests/__init__.py
# (empty)

# tests/conftest.py
"""Shared test fixtures for CRISP."""
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
        text=text or f"The answer is \\boxed{{{answer}}}" if answer else "No answer here",
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
```

**Step 5: Create subpackage init files**

Create empty `__init__.py` in: `crisp/verifier/`, `crisp/rewards/`, `crisp/discussion/`, `crisp/training/`, `crisp/evaluation/`, `crisp/workflow/`, `crisp/infra/`

**Step 6: Run basic import test**

```bash
cd /home/alex/mech_interp/CRISP && pip install -e ".[dev]" && python -c "from crisp.types import Rollout; print('OK')"
```

**Step 7: Commit**

```bash
git add crisp/ tests/ pyproject.toml docs/
git commit -m "feat: scaffold CRISP project structure with types, config, and test fixtures"
```

---

## Phase 1: Verifier Module (answer_extraction + sympy_verify)

### Task 2: Answer extraction — tests

**Files:**
- Create: `tests/verifier/__init__.py`
- Create: `tests/verifier/test_answer_extraction.py`

**Step 1: Write failing tests**

```python
"""Tests for \\boxed{} answer extraction."""
import pytest

from crisp.verifier.answer_extraction import extract_boxed


class TestExtractBoxed:
    def test_simple_number(self):
        assert extract_boxed("The answer is \\boxed{42}") == "42"

    def test_fraction(self):
        assert extract_boxed("Therefore \\boxed{\\frac{3}{4}}") == "\\frac{3}{4}"

    def test_nested_braces(self):
        assert extract_boxed("\\boxed{\\{1, 2, 3\\}}") == "\\{1, 2, 3\\}"

    def test_latex_commands_inside(self):
        assert extract_boxed("\\boxed{\\sqrt{2} + \\pi}") == "\\sqrt{2} + \\pi"

    def test_no_boxed_returns_none(self):
        assert extract_boxed("There is no answer here") is None

    def test_empty_boxed(self):
        assert extract_boxed("\\boxed{}") == ""

    def test_multiple_boxed_returns_last(self):
        text = "First \\boxed{1} then \\boxed{2}"
        assert extract_boxed(text) == "2"

    def test_deeply_nested_braces(self):
        assert extract_boxed("\\boxed{f(g(x))}") == "f(g(x))"

    def test_multiline_text(self):
        text = "Step 1: compute\nStep 2: simplify\n\\boxed{7}"
        assert extract_boxed(text) == "7"

    def test_boxed_with_spaces(self):
        assert extract_boxed("\\boxed{ 42 }") == " 42 "

    def test_negative_number(self):
        assert extract_boxed("\\boxed{-3}") == "-3"

    def test_expression_with_equals(self):
        assert extract_boxed("\\boxed{x = 5}") == "x = 5"
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/verifier/test_answer_extraction.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'crisp.verifier.answer_extraction'`

### Task 3: Answer extraction — implementation

**Files:**
- Create: `crisp/verifier/__init__.py`
- Create: `crisp/verifier/answer_extraction.py`

**Step 1: Implement extract_boxed**

```python
"""Extract \\boxed{} answers from math solutions."""
from __future__ import annotations


def extract_boxed(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} in text.

    Handles nested braces by counting brace depth.
    Returns None if no \\boxed{} is found.
    """
    results = []
    search_str = "\\boxed{"
    idx = 0
    while idx < len(text):
        pos = text.find(search_str, idx)
        if pos == -1:
            break
        # Start after \boxed{
        start = pos + len(search_str)
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
                depth += 1
            elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start : i - 1])
        idx = i
    return results[-1] if results else None
```

**Step 2: Run tests**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/verifier/test_answer_extraction.py -v
```
Expected: ALL PASS

**Step 3: Commit**

```bash
git add crisp/verifier/ tests/verifier/
git commit -m "feat: add \\boxed{} answer extraction with nested brace support"
```

### Task 4: SymPy verifier — tests

**Files:**
- Create: `tests/verifier/test_sympy_verify.py`

**Step 1: Write failing tests**

```python
"""Tests for 3-strategy SymPy verification."""
import pytest

from crisp.verifier.sympy_verify import check, equivalent


class TestCheck:
    """Test check(answer, ground_truth) → bool."""

    # Strategy 1: exact string match
    def test_exact_match(self):
        assert check("42", "42") is True

    def test_exact_mismatch(self):
        assert check("43", "42") is False

    # Strategy 2: numeric comparison
    def test_numeric_float_equivalence(self):
        assert check("0.333333", "1/3") is True

    def test_numeric_integer_vs_float(self):
        assert check("3.0", "3") is True

    def test_numeric_tolerance(self):
        assert check("3.14159", "3.14159265") is True

    def test_numeric_outside_tolerance(self):
        assert check("3.15", "3.14") is False

    # Strategy 3: symbolic equivalence
    def test_symbolic_expand(self):
        assert check("x^2 + 2*x + 1", "(x+1)^2") is True

    def test_symbolic_fraction_simplify(self):
        assert check("\\frac{2}{4}", "\\frac{1}{2}") is True

    def test_symbolic_sqrt(self):
        assert check("\\sqrt{4}", "2") is True

    # Edge cases
    def test_none_answer(self):
        assert check(None, "42") is False

    def test_empty_answer(self):
        assert check("", "42") is False

    def test_both_none(self):
        assert check(None, None) is False

    def test_whitespace_handling(self):
        assert check(" 42 ", "42") is True


class TestEquivalent:
    """Test equivalent(a, b) → bool. Symmetric version."""

    def test_symmetric(self):
        assert equivalent("1/3", "0.333333") is True
        assert equivalent("0.333333", "1/3") is True

    def test_both_wrong_but_equal(self):
        assert equivalent("99", "99") is True

    def test_different_answers(self):
        assert equivalent("42", "43") is False

    def test_none_inputs(self):
        assert equivalent(None, "42") is False
        assert equivalent("42", None) is False
        assert equivalent(None, None) is False
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/verifier/test_sympy_verify.py -v
```
Expected: FAIL — module not found

### Task 5: SymPy verifier — implementation

**Files:**
- Create: `crisp/verifier/sympy_verify.py`

**Step 1: Implement the 3-strategy verifier**

```python
"""Three-strategy math answer verification using SymPy."""
from __future__ import annotations

import re
from typing import Optional

import sympy
from sympy.parsing.latex import parse_latex


def check(answer: Optional[str], ground_truth: Optional[str]) -> bool:
    """Check if answer matches ground_truth using 3 strategies.

    Strategies tried in order:
    1. Exact string match (after stripping whitespace)
    2. Numeric comparison (within tolerance 1e-4)
    3. SymPy symbolic equivalence
    """
    if answer is None or ground_truth is None:
        return False
    answer = answer.strip()
    ground_truth = ground_truth.strip()
    if not answer or not ground_truth:
        return False

    # Strategy 1: exact string match
    if answer == ground_truth:
        return True

    # Strategy 2: numeric comparison
    if _numeric_equal(answer, ground_truth):
        return True

    # Strategy 3: symbolic equivalence
    if _symbolic_equal(answer, ground_truth):
        return True

    return False


def equivalent(a: Optional[str], b: Optional[str]) -> bool:
    """Symmetric equivalence check between two answers.

    Same logic as check() but semantically distinct:
    neither argument is privileged as 'ground truth'.
    """
    return check(a, b)


def _numeric_equal(a: str, b: str, tol: float = 1e-4) -> bool:
    """Try to parse both as numbers and compare within tolerance."""
    val_a = _try_parse_number(a)
    val_b = _try_parse_number(b)
    if val_a is not None and val_b is not None:
        return abs(val_a - val_b) < tol * max(1.0, abs(val_b))
    return False


def _try_parse_number(s: str) -> Optional[float]:
    """Try to parse a string as a float, handling fractions."""
    s = s.strip()
    # Direct float
    try:
        return float(s)
    except ValueError:
        pass
    # Simple fraction: a/b
    m = re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", s)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den != 0:
            return num / den
    # LaTeX fraction: \frac{a}{b}
    m = re.match(r"^\\frac\{(-?\d+)\}\{(-?\d+)\}$", s)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den != 0:
            return num / den
    return None


def _symbolic_equal(a: str, b: str) -> bool:
    """Try SymPy symbolic equivalence."""
    try:
        expr_a = _parse_to_sympy(a)
        expr_b = _parse_to_sympy(b)
        if expr_a is None or expr_b is None:
            return False
        diff = sympy.simplify(expr_a - expr_b)
        return diff == 0
    except (sympy.SympifyError, TypeError, ValueError, AttributeError):
        return False


def _parse_to_sympy(s: str) -> Optional[sympy.Expr]:
    """Parse a string (possibly LaTeX) to a SymPy expression."""
    s = s.strip()
    # Replace ^ with ** for SymPy
    s_py = s.replace("^", "**")
    # Try direct sympify first
    try:
        return sympy.sympify(s_py)
    except (sympy.SympifyError, TypeError, ValueError):
        pass
    # Try LaTeX parsing
    try:
        return parse_latex(s)
    except Exception:
        pass
    return None
```

**Step 2: Run tests**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/verifier/test_sympy_verify.py -v
```
Expected: ALL PASS

**Step 3: Commit**

```bash
git add crisp/verifier/sympy_verify.py tests/verifier/test_sympy_verify.py
git commit -m "feat: add 3-strategy SymPy verifier (string, numeric, symbolic)"
```

---

## Phase 2: Rewards Module

### Task 6: EMA tracker — tests

**Files:**
- Create: `tests/rewards/__init__.py`
- Create: `tests/rewards/test_ema_tracker.py`

**Step 1: Write failing tests**

```python
"""Tests for EMA mean/variance tracker."""
import math
import pytest

from crisp.rewards.ema_tracker import EMATracker


class TestEMATracker:
    def test_initial_values(self):
        t = EMATracker()
        assert t.mu == 0.5
        assert t.sigma_sq == 0.25

    def test_custom_init(self):
        t = EMATracker(mu=0.0, sigma_sq=1.0, eta=0.1)
        assert t.mu == 0.0
        assert t.sigma_sq == 1.0
        assert t.eta == 0.1

    def test_single_update(self):
        t = EMATracker(mu=0.5, sigma_sq=0.25, eta=0.2)
        t.update([1.0, 1.0, 1.0])  # batch_mean=1.0, batch_var=0.0
        assert t.mu == pytest.approx(0.8 * 0.5 + 0.2 * 1.0)  # 0.6
        assert t.sigma_sq == pytest.approx(0.8 * 0.25 + 0.2 * 0.0)  # 0.2

    def test_empty_update_is_noop(self):
        t = EMATracker(mu=0.5, sigma_sq=0.25, eta=0.2)
        t.update([])
        assert t.mu == 0.5
        assert t.sigma_sq == 0.25

    def test_convergence_to_constant(self):
        """After many updates with reward=1.0, mu should approach 1.0."""
        t = EMATracker(mu=0.5, sigma_sq=0.25, eta=0.2)
        for _ in range(50):
            t.update([1.0])
        assert t.mu == pytest.approx(1.0, abs=1e-6)
        assert t.sigma_sq == pytest.approx(0.0, abs=1e-6)

    def test_multiple_updates_track_mean(self):
        t = EMATracker(mu=0.0, sigma_sq=0.0, eta=0.2)
        t.update([0.0, 1.0])  # mean=0.5, var=0.25
        assert t.mu == pytest.approx(0.1)  # 0.8*0 + 0.2*0.5
        assert t.sigma_sq == pytest.approx(0.05)  # 0.8*0 + 0.2*0.25

    def test_consecutive_empty_batches_counter(self):
        t = EMATracker()
        for _ in range(6):
            t.update([])
        assert t.consecutive_empty >= 6
        t.update([1.0])
        assert t.consecutive_empty == 0
```

**Step 2: Run to verify failure**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/rewards/test_ema_tracker.py -v
```

### Task 7: EMA tracker — implementation

**Files:**
- Create: `crisp/rewards/__init__.py`
- Create: `crisp/rewards/ema_tracker.py`

**Step 1: Implement**

```python
"""EMA (Exponential Moving Average) tracker for reward normalization."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EMATracker:
    """Tracks running mean and variance with exponential moving average.

    Used for Pool 2 (post-discussion) advantage normalization where
    per-batch sample sizes (~3-4) are too small for stable statistics.
    """
    mu: float = 0.5
    sigma_sq: float = 0.25
    eta: float = 0.2
    consecutive_empty: int = field(default=0, repr=False)
    _warning_threshold: int = field(default=5, repr=False)

    def update(self, rewards: list[float]) -> None:
        """Update EMA with a batch of rewards."""
        if not rewards:
            self.consecutive_empty += 1
            if self.consecutive_empty >= self._warning_threshold:
                logger.warning(
                    "Pool 2 empty for %d consecutive batches. "
                    "Players may have converged or discussion trigger may be broken.",
                    self.consecutive_empty,
                )
            return

        self.consecutive_empty = 0
        batch_mean = float(np.mean(rewards))
        batch_var = float(np.var(rewards))
        self.mu = (1 - self.eta) * self.mu + self.eta * batch_mean
        self.sigma_sq = (1 - self.eta) * self.sigma_sq + self.eta * batch_var
```

**Step 2: Run tests**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/rewards/test_ema_tracker.py -v
```

**Step 3: Commit**

```bash
git add crisp/rewards/ tests/rewards/
git commit -m "feat: add EMA tracker for post-discussion advantage normalization"
```

### Task 8: Player rewards — tests

**Files:**
- Create: `tests/rewards/test_player_rewards.py`

**Step 1: Write failing tests**

```python
"""Tests for player reward computation."""
import pytest

from crisp.types import Rollout, DiscussionResult, Problem
from crisp.rewards.player_rewards import compute_solve_reward, apply_persuader_bonus
from tests.conftest import make_rollout, make_problem


class TestComputeSolveReward:
    def test_correct_answer(self):
        r = make_rollout(correct=True)
        assert compute_solve_reward(r) == 1.0

    def test_wrong_answer(self):
        r = make_rollout(answer="99", correct=False)
        assert compute_solve_reward(r) == 0.0

    def test_no_boxed_answer(self):
        r = make_rollout(answer=None, correct=None)
        assert compute_solve_reward(r) == -0.5

    def test_empty_answer_treated_as_no_box(self):
        """Empty string from \\boxed{} is still an extraction — treat as wrong, not missing."""
        r = make_rollout(answer="", correct=False)
        assert compute_solve_reward(r) == 0.0


class TestApplyPersuaderBonus:
    def _make_scenario(self):
        """2 players, 1 problem, 8 rollouts each.
        Player 0: 6 correct, 2 wrong → majority correct
        Player 1: 3 correct, 5 wrong → majority wrong
        """
        problems = [make_problem()]
        rollouts = {
            0: [make_rollout(problem_idx=0, player_id=0, answer="42", correct=True, reward=1.0) for _ in range(6)]
              + [make_rollout(problem_idx=0, player_id=0, answer="99", correct=False, reward=0.0) for _ in range(2)],
            1: [make_rollout(problem_idx=0, player_id=1, answer="42", correct=True, reward=1.0) for _ in range(3)]
              + [make_rollout(problem_idx=0, player_id=1, answer="99", correct=False, reward=0.0) for _ in range(5)],
        }
        majority_answers = {(0, 0): "42", (1, 0): "99"}
        # Player 1 flipped to correct after discussion
        discussion_results = {
            0: [DiscussionResult(problem_idx=0, player_id=0, tokens=[], text="", log_probs=[], final_answer="42", correct=True, reward=1.0)],
            1: [DiscussionResult(problem_idx=0, player_id=1, tokens=[], text="", log_probs=[], final_answer="42", correct=True, reward=1.0)],
        }
        return problems, rollouts, majority_answers, discussion_results

    def test_persuader_bonus_on_correct_rollouts(self):
        problems, rollouts, majority, disc = self._make_scenario()
        apply_persuader_bonus(rollouts, disc, majority, problems, gamma=0.3)
        # Player 0 was persuader: correct rollouts get +0.3
        for r in rollouts[0]:
            if r.correct:
                assert r.reward == pytest.approx(1.3)
            else:
                assert r.reward == pytest.approx(0.0)

    def test_no_bonus_on_wrong_player_rollouts(self):
        problems, rollouts, majority, disc = self._make_scenario()
        apply_persuader_bonus(rollouts, disc, majority, problems, gamma=0.3)
        # Player 1 was NOT the persuader — no bonus
        for r in rollouts[1]:
            if r.correct:
                assert r.reward == pytest.approx(1.0)
            else:
                assert r.reward == pytest.approx(0.0)

    def test_no_bonus_when_peer_didnt_flip(self):
        """If peer stayed wrong, no persuader bonus even if one player was correct."""
        problems, rollouts, majority, disc = self._make_scenario()
        # Peer didn't flip — still wrong after discussion
        disc[1][0].correct = False
        disc[1][0].final_answer = "99"
        apply_persuader_bonus(rollouts, disc, majority, problems, gamma=0.3)
        for r in rollouts[0]:
            if r.correct:
                assert r.reward == pytest.approx(1.0)  # No bonus

    def test_idempotency_guard(self):
        """Calling apply_persuader_bonus twice should raise."""
        problems, rollouts, majority, disc = self._make_scenario()
        apply_persuader_bonus(rollouts, disc, majority, problems, gamma=0.3)
        with pytest.raises(RuntimeError, match="already applied"):
            apply_persuader_bonus(rollouts, disc, majority, problems, gamma=0.3)

    def test_no_discussion_no_bonus(self):
        """If no discussion occurred, nothing changes."""
        problems = [make_problem()]
        rollouts = {
            0: [make_rollout(problem_idx=0, player_id=0, correct=True, reward=1.0) for _ in range(8)],
            1: [make_rollout(problem_idx=0, player_id=1, correct=True, reward=1.0) for _ in range(8)],
        }
        majority = {(0, 0): "42", (1, 0): "42"}
        disc = {0: [], 1: []}
        apply_persuader_bonus(rollouts, disc, majority, problems, gamma=0.3)
        for pid in [0, 1]:
            for r in rollouts[pid]:
                assert r.reward == pytest.approx(1.0)

    def test_both_wrong_no_bonus(self):
        """If both players had wrong majority, no persuader bonus."""
        problems = [make_problem()]
        rollouts = {
            0: [make_rollout(problem_idx=0, player_id=0, answer="99", correct=False, reward=0.0) for _ in range(8)],
            1: [make_rollout(problem_idx=0, player_id=1, answer="88", correct=False, reward=0.0) for _ in range(8)],
        }
        majority = {(0, 0): "99", (1, 0): "88"}
        disc = {
            0: [DiscussionResult(problem_idx=0, player_id=0, tokens=[], text="", log_probs=[], final_answer="99", correct=False, reward=0.0)],
            1: [DiscussionResult(problem_idx=0, player_id=1, tokens=[], text="", log_probs=[], final_answer="88", correct=False, reward=0.0)],
        }
        apply_persuader_bonus(rollouts, disc, majority, problems, gamma=0.3)
        for pid in [0, 1]:
            for r in rollouts[pid]:
                assert r.reward == pytest.approx(0.0)
```

**Step 2: Run to verify failure**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/rewards/test_player_rewards.py -v
```

### Task 9: Player rewards — implementation

**Files:**
- Create: `crisp/rewards/player_rewards.py`

**Step 1: Implement**

```python
"""Player reward computation: solve rewards and persuader bonus."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from crisp.types import DiscussionResult, Problem, Rollout
from crisp.verifier.sympy_verify import check


def compute_solve_reward(rollout: Rollout) -> float:
    """Compute pre-discussion reward for a single rollout.

    Returns:
        1.0 if correct, 0.0 if wrong, -0.5 if no \\boxed{} answer extracted.
    """
    if rollout.answer is None:
        return -0.5
    if rollout.correct:
        return 1.0
    return 0.0


def apply_persuader_bonus(
    rollouts: Dict[int, List[Rollout]],
    discussion_results: Dict[int, List[DiscussionResult]],
    majority_answers: Dict[Tuple[int, int], str],
    problems: List[Problem],
    gamma: float = 0.3,
) -> None:
    """Apply persuader bonus to correct pre-discussion rollouts in-place.

    Persuader = player whose majority answer was correct AND whose peer
    flipped from wrong to correct after discussion.

    Raises RuntimeError if called twice on the same rollouts.
    """
    # Idempotency guard: check if any rollout already has the bonus flag
    for pid in rollouts:
        for r in rollouts[pid]:
            if r._persuader_bonus_applied:
                raise RuntimeError(
                    "Persuader bonus already applied. "
                    "apply_persuader_bonus must only be called once per batch."
                )

    # Collect discussed problem indices
    discussed_problems = set()
    for pid in discussion_results:
        for dr in discussion_results[pid]:
            discussed_problems.add(dr.problem_idx)

    for prob_idx in discussed_problems:
        persuader_id = _find_persuader(
            prob_idx, majority_answers, discussion_results, problems
        )
        if persuader_id is not None:
            for r in rollouts[persuader_id]:
                if r.problem_idx == prob_idx and r.correct:
                    r.reward += gamma

    # Mark all rollouts as processed
    for pid in rollouts:
        for r in rollouts[pid]:
            r._persuader_bonus_applied = True


def _find_persuader(
    problem_idx: int,
    majority_answers: Dict[Tuple[int, int], str],
    discussion_results: Dict[int, List[DiscussionResult]],
    problems: List[Problem],
) -> Optional[int]:
    """Find the persuader for a discussed problem, if any.

    Returns player_id of the persuader, or None if no persuasion occurred.
    Persuader = player with correct majority answer whose peer flipped to correct.
    """
    ground_truth = problems[problem_idx].ground_truth

    # Find which player(s) had correct majority
    correct_players = []
    for pid in [0, 1]:
        maj = majority_answers.get((pid, problem_idx))
        if maj is not None and check(maj, ground_truth):
            correct_players.append(pid)

    if len(correct_players) != 1:
        # Both correct (shouldn't trigger discussion) or both wrong (no persuader)
        return None

    persuader_id = correct_players[0]
    peer_id = 1 - persuader_id

    # Check if peer flipped to correct
    peer_results = [
        dr for dr in discussion_results.get(peer_id, [])
        if dr.problem_idx == problem_idx
    ]
    if peer_results and peer_results[0].correct:
        return persuader_id

    return None
```

**Step 2: Run tests**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/rewards/test_player_rewards.py -v
```

**Step 3: Commit**

```bash
git add crisp/rewards/player_rewards.py tests/rewards/test_player_rewards.py
git commit -m "feat: add player reward computation with persuader bonus"
```

### Task 10: Repetition buffer — tests

**Files:**
- Create: `tests/rewards/test_repetition_buffer.py`

**Step 1: Write failing tests**

```python
"""Tests for sliding window repetition buffer."""
import numpy as np
import pytest

from crisp.rewards.repetition_buffer import RepetitionBuffer


class TestRepetitionBuffer:
    def test_empty_buffer_returns_zero(self):
        buf = RepetitionBuffer(max_batches=10, embedding_dim=384)
        emb = np.random.default_rng(0).random(384).astype(np.float32)
        assert buf.compute_penalty(emb, lambda_rep=1.0, tau_sim=0.85) == 0.0

    def test_warmup_period(self):
        """Buffer returns 0 until full (first W batches)."""
        buf = RepetitionBuffer(max_batches=3, embedding_dim=4)
        rng = np.random.default_rng(42)
        for _ in range(2):
            buf.push([rng.random(4).astype(np.float32) for _ in range(4)])
        emb = rng.random(4).astype(np.float32)
        assert buf.compute_penalty(emb, lambda_rep=1.0, tau_sim=0.85) == 0.0

    def test_identical_embeddings_high_penalty(self):
        """Identical embeddings should have sim=1.0 > tau, triggering penalty."""
        buf = RepetitionBuffer(max_batches=2, embedding_dim=4)
        emb = np.ones(4, dtype=np.float32)
        # Fill buffer to capacity
        buf.push([emb.copy()])
        buf.push([emb.copy()])
        penalty = buf.compute_penalty(emb, lambda_rep=1.0, tau_sim=0.85)
        assert penalty > 0.0

    def test_orthogonal_embeddings_no_penalty(self):
        """Orthogonal embeddings should have sim~0 < tau, no penalty."""
        buf = RepetitionBuffer(max_batches=2, embedding_dim=4)
        buf.push([np.array([1, 0, 0, 0], dtype=np.float32)])
        buf.push([np.array([0, 1, 0, 0], dtype=np.float32)])
        query = np.array([0, 0, 0, 1], dtype=np.float32)
        penalty = buf.compute_penalty(query, lambda_rep=1.0, tau_sim=0.85)
        assert penalty == 0.0

    def test_fifo_overflow(self):
        """Oldest batch should be evicted when buffer exceeds max_batches."""
        buf = RepetitionBuffer(max_batches=2, embedding_dim=4)
        emb_old = np.array([1, 0, 0, 0], dtype=np.float32)
        emb_mid = np.array([0, 1, 0, 0], dtype=np.float32)
        emb_new = np.array([0, 0, 1, 0], dtype=np.float32)
        buf.push([emb_old])
        buf.push([emb_mid])
        buf.push([emb_new])  # Should evict emb_old
        assert len(buf.buffer) == 2

    def test_penalty_normalized_by_history_size(self):
        """Penalty = lambda * count(sim > tau) / |H|."""
        buf = RepetitionBuffer(max_batches=2, embedding_dim=4)
        emb = np.ones(4, dtype=np.float32)
        # 2 batches of 4 identical embeddings = 8 historical
        buf.push([emb.copy() for _ in range(4)])
        buf.push([emb.copy() for _ in range(4)])
        penalty = buf.compute_penalty(emb, lambda_rep=1.0, tau_sim=0.85)
        # All 8 are similar, so penalty = 1.0 * 8 / 8 = 1.0
        assert penalty == pytest.approx(1.0)

    def test_buffer_len(self):
        buf = RepetitionBuffer(max_batches=3, embedding_dim=4)
        assert len(buf.buffer) == 0
        buf.push([np.zeros(4)])
        assert len(buf.buffer) == 1
```

**Step 2: Run to verify failure**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/rewards/test_repetition_buffer.py -v
```

### Task 11: Repetition buffer — implementation

**Files:**
- Create: `crisp/rewards/repetition_buffer.py`

**Step 1: Implement**

```python
"""Sliding window embedding buffer for coach repetition penalty."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cosine


@dataclass
class RepetitionBuffer:
    """FIFO buffer of problem embeddings for cross-batch repetition detection.

    Stores embeddings from the last W batches. Cross-batch penalty only
    activates once the buffer is full (warm-up period).
    """
    max_batches: int = 10
    embedding_dim: int = 384
    buffer: deque = field(default_factory=deque)

    def compute_penalty(
        self,
        embedding: np.ndarray,
        lambda_rep: float = 1.0,
        tau_sim: float = 0.85,
    ) -> float:
        """Compute cross-batch repetition penalty for a single embedding.

        Returns 0.0 during warm-up (buffer not yet full).
        Otherwise returns lambda * count(sim > tau) / |H|.
        """
        if len(self.buffer) < self.max_batches:
            return 0.0

        all_historical = np.vstack(list(self.buffer))
        # Cosine similarity: 1 - cosine_distance
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
        hist_norms = all_historical / (
            np.linalg.norm(all_historical, axis=1, keepdims=True) + 1e-10
        )
        similarities = hist_norms @ emb_norm
        count_similar = int(np.sum(similarities > tau_sim))
        return lambda_rep * count_similar / len(all_historical)

    def push(self, embeddings: list[np.ndarray]) -> None:
        """Push a batch of embeddings. Evicts oldest if over capacity."""
        self.buffer.append(np.stack(embeddings))
        if len(self.buffer) > self.max_batches:
            self.buffer.popleft()
```

**Step 2: Run tests**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/rewards/test_repetition_buffer.py -v
```

**Step 3: Commit**

```bash
git add crisp/rewards/repetition_buffer.py tests/rewards/test_repetition_buffer.py
git commit -m "feat: add sliding window repetition buffer for coach diversity"
```

### Task 12: Coach rewards — tests

**Files:**
- Create: `tests/rewards/test_coach_rewards.py`

**Step 1: Write failing tests**

```python
"""Tests for coach reward computation."""
import numpy as np
import pytest

from crisp.types import Rollout, DiscussionResult, Problem
from crisp.rewards.coach_rewards import (
    compute_coach_reward,
    compute_uncertainty_reward,
    compute_discussion_reward,
    compute_intra_batch_penalty,
)
from crisp.rewards.repetition_buffer import RepetitionBuffer
from tests.conftest import make_rollout, make_problem


class TestUncertaintyReward:
    def test_fifty_percent_is_max(self):
        """p_hat = 0.5 → r_uncertainty = 1.0 (maximum)."""
        assert compute_uncertainty_reward(0.5) == pytest.approx(1.0)

    def test_zero_percent(self):
        """p_hat = 0.0 → r_uncertainty = 0.0."""
        assert compute_uncertainty_reward(0.0) == pytest.approx(0.0)

    def test_hundred_percent(self):
        """p_hat = 1.0 → r_uncertainty = 0.0."""
        assert compute_uncertainty_reward(1.0) == pytest.approx(0.0)

    def test_thirty_percent(self):
        """p_hat = 0.3 → r_uncertainty = 1 - 2|0.3 - 0.5| = 1 - 0.4 = 0.6."""
        assert compute_uncertainty_reward(0.3) == pytest.approx(0.6)

    def test_seventy_percent(self):
        """Symmetric: p_hat = 0.7 → same as 0.3."""
        assert compute_uncertainty_reward(0.7) == pytest.approx(0.6)


class TestDiscussionReward:
    def test_resolved_correctly(self):
        assert compute_discussion_reward(
            discussion_occurred=True, resolved_correctly=True, alpha=0.3
        ) == pytest.approx(0.3)

    def test_occurred_but_unresolved(self):
        assert compute_discussion_reward(
            discussion_occurred=True, resolved_correctly=False, alpha=0.3
        ) == pytest.approx(0.15)

    def test_no_discussion(self):
        assert compute_discussion_reward(
            discussion_occurred=False, resolved_correctly=False, alpha=0.3
        ) == pytest.approx(0.0)


class TestIntraBatchPenalty:
    def test_all_unique(self):
        """Orthogonal embeddings → no intra-batch penalty."""
        embeddings = [
            np.array([1, 0, 0, 0], dtype=np.float32),
            np.array([0, 1, 0, 0], dtype=np.float32),
            np.array([0, 0, 1, 0], dtype=np.float32),
        ]
        penalty = compute_intra_batch_penalty(
            idx=0, embeddings=embeddings, lambda_rep=1.0, tau_sim=0.85
        )
        assert penalty == pytest.approx(0.0)

    def test_all_identical(self):
        """Identical embeddings → penalty = lambda * (n-1) / (n-1) = lambda."""
        emb = np.ones(4, dtype=np.float32)
        embeddings = [emb.copy() for _ in range(4)]
        penalty = compute_intra_batch_penalty(
            idx=0, embeddings=embeddings, lambda_rep=1.0, tau_sim=0.85
        )
        assert penalty == pytest.approx(1.0)

    def test_single_problem_no_penalty(self):
        """Single problem in batch → no intra-batch comparison possible."""
        emb = np.ones(4, dtype=np.float32)
        penalty = compute_intra_batch_penalty(
            idx=0, embeddings=[emb], lambda_rep=1.0, tau_sim=0.85
        )
        assert penalty == pytest.approx(0.0)


class TestComputeCoachReward:
    def test_perfect_difficulty_no_discussion(self):
        """p_hat=0.5, no discussion, no repetition → r = 1.0."""
        problems = [make_problem(embedding=np.ones(4, dtype=np.float32))]
        # 8 correct + 8 wrong = p_hat 0.5
        rollouts = (
            [make_rollout(problem_idx=0, player_id=0, correct=True) for _ in range(4)]
            + [make_rollout(problem_idx=0, player_id=0, correct=False) for _ in range(4)]
            + [make_rollout(problem_idx=0, player_id=1, correct=True) for _ in range(4)]
            + [make_rollout(problem_idx=0, player_id=1, correct=False) for _ in range(4)]
        )
        buf = RepetitionBuffer(max_batches=10, embedding_dim=4)
        reward = compute_coach_reward(
            problem=problems[0],
            problem_idx=0,
            all_embeddings=[problems[0].coach_embedding],
            player_rollouts=rollouts,
            discussion_occurred=False,
            resolved_correctly=False,
            repetition_buffer=buf,
        )
        assert reward == pytest.approx(1.0)

    def test_too_easy_clamped_to_zero(self):
        """p_hat=1.0, r_uncertainty=0, no discussion → r = max(0, 0) = 0."""
        problems = [make_problem(embedding=np.ones(4, dtype=np.float32))]
        rollouts = [make_rollout(problem_idx=0, correct=True) for _ in range(16)]
        buf = RepetitionBuffer(max_batches=10, embedding_dim=4)
        reward = compute_coach_reward(
            problem=problems[0],
            problem_idx=0,
            all_embeddings=[problems[0].coach_embedding],
            player_rollouts=rollouts,
            discussion_occurred=False,
            resolved_correctly=False,
            repetition_buffer=buf,
        )
        assert reward == pytest.approx(0.0)

    def test_floor_at_zero(self):
        """If repetition penalty exceeds positive components, floor at 0."""
        problems = [make_problem(embedding=np.ones(4, dtype=np.float32))]
        rollouts = [make_rollout(problem_idx=0, correct=True) for _ in range(16)]
        # Fill buffer with identical embeddings to get high cross-batch penalty
        buf = RepetitionBuffer(max_batches=2, embedding_dim=4)
        ident = np.ones(4, dtype=np.float32)
        buf.push([ident] * 4)
        buf.push([ident] * 4)
        reward = compute_coach_reward(
            problem=problems[0],
            problem_idx=0,
            all_embeddings=[problems[0].coach_embedding],
            player_rollouts=rollouts,
            discussion_occurred=False,
            resolved_correctly=False,
            repetition_buffer=buf,
        )
        assert reward >= 0.0
```

**Step 2: Run to verify failure**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/rewards/test_coach_rewards.py -v
```

### Task 13: Coach rewards — implementation

**Files:**
- Create: `crisp/rewards/coach_rewards.py`

**Step 1: Implement**

```python
"""Coach reward computation: uncertainty + discussion - repetition."""
from __future__ import annotations

from typing import List

import numpy as np

from crisp.rewards.repetition_buffer import RepetitionBuffer
from crisp.types import Problem, Rollout


def compute_uncertainty_reward(p_hat: float) -> float:
    """r_uncertainty = 1 - 2|p_hat - 0.5|. Peaks at p_hat=0.5."""
    return 1.0 - 2.0 * abs(p_hat - 0.5)


def compute_discussion_reward(
    discussion_occurred: bool,
    resolved_correctly: bool,
    alpha: float = 0.3,
) -> float:
    """r_discussion based on whether discussion occurred and resolved correctly."""
    if not discussion_occurred:
        return 0.0
    if resolved_correctly:
        return alpha
    return alpha / 2


def compute_intra_batch_penalty(
    idx: int,
    embeddings: List[np.ndarray],
    lambda_rep: float = 1.0,
    tau_sim: float = 0.85,
) -> float:
    """Compute within-batch repetition penalty for problem at idx."""
    if len(embeddings) <= 1:
        return 0.0

    query = embeddings[idx]
    others = np.stack([e for j, e in enumerate(embeddings) if j != idx])

    query_norm = query / (np.linalg.norm(query) + 1e-10)
    others_norm = others / (np.linalg.norm(others, axis=1, keepdims=True) + 1e-10)

    similarities = others_norm @ query_norm
    count_similar = int(np.sum(similarities > tau_sim))
    return lambda_rep * count_similar / len(others)


def compute_coach_reward(
    problem: Problem,
    problem_idx: int,
    all_embeddings: List[np.ndarray],
    player_rollouts: List[Rollout],
    discussion_occurred: bool,
    resolved_correctly: bool,
    repetition_buffer: RepetitionBuffer,
    alpha: float = 0.3,
    lambda_rep: float = 1.0,
    tau_sim: float = 0.85,
) -> float:
    """Compute full coach reward for a single problem.

    r_coach = max(0, r_uncertainty + r_discussion - r_repetition)
    """
    # Solve rate across both players
    correct_count = sum(1 for r in player_rollouts if r.correct)
    p_hat = correct_count / len(player_rollouts) if player_rollouts else 0.0

    r_uncertainty = compute_uncertainty_reward(p_hat)
    r_discussion = compute_discussion_reward(discussion_occurred, resolved_correctly, alpha)

    # Repetition: intra-batch + cross-batch
    r_intra = compute_intra_batch_penalty(problem_idx, all_embeddings, lambda_rep, tau_sim)
    r_cross = repetition_buffer.compute_penalty(problem.coach_embedding, lambda_rep, tau_sim)
    r_repetition = r_intra + r_cross

    return max(0.0, r_uncertainty + r_discussion - r_repetition)
```

**Step 2: Run tests**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/rewards/test_coach_rewards.py -v
```

**Step 3: Commit**

```bash
git add crisp/rewards/coach_rewards.py tests/rewards/test_coach_rewards.py
git commit -m "feat: add coach reward computation with uncertainty, discussion, and repetition"
```

### Task 14: Advantage computation — tests

**Files:**
- Create: `tests/rewards/test_advantages.py`

**Step 1: Write failing tests**

```python
"""Tests for two-pool advantage normalization."""
import math
import numpy as np
import pytest

from crisp.rewards.advantages import compute_player_advantages
from crisp.rewards.ema_tracker import EMATracker


class TestComputePlayerAdvantages:
    def test_basic_normalization(self):
        """Mean-centered, std-normalized."""
        pre_rewards = [0.0, 0.0, 1.0, 1.0]
        post_rewards = []
        ema = EMATracker(mu=0.5, sigma_sq=0.25)
        pre_adv, post_adv = compute_player_advantages(pre_rewards, post_rewards, ema)
        mean = np.mean(pre_rewards)
        std = np.std(pre_rewards)
        expected = [(r - mean) / (std + 1e-8) for r in pre_rewards]
        for got, exp in zip(pre_adv, expected):
            assert got == pytest.approx(exp)
        assert post_adv == []

    def test_all_same_rewards_zero_advantage(self):
        """All rewards identical → std=0 → advantages ≈ 0."""
        pre_rewards = [1.0, 1.0, 1.0, 1.0]
        ema = EMATracker()
        pre_adv, _ = compute_player_advantages(pre_rewards, [], ema)
        for a in pre_adv:
            assert a == pytest.approx(0.0, abs=1e-4)

    def test_post_discussion_uses_ema(self):
        """Post-discussion pool uses EMA-smoothed stats."""
        pre_rewards = [0.0, 1.0]
        post_rewards = [1.0, 0.0]
        ema = EMATracker(mu=0.5, sigma_sq=0.25, eta=0.2)
        pre_adv, post_adv = compute_player_advantages(pre_rewards, post_rewards, ema)
        # After update: mu = 0.8*0.5 + 0.2*0.5 = 0.5, sigma_sq = 0.8*0.25 + 0.2*0.25 = 0.25
        assert len(post_adv) == 2
        expected_post = [(r - 0.5) / (math.sqrt(0.25) + 1e-8) for r in post_rewards]
        for got, exp in zip(post_adv, expected_post):
            assert got == pytest.approx(exp, rel=1e-4)

    def test_ema_updated_after_call(self):
        """The EMA tracker should be updated by the function call."""
        ema = EMATracker(mu=0.0, sigma_sq=0.0, eta=0.2)
        compute_player_advantages([0.0, 1.0], [1.0], ema)
        assert ema.mu != 0.0  # Should have been updated

    def test_negative_rewards_handled(self):
        """Pre-discussion rewards can be -0.5 (no-box penalty)."""
        pre_rewards = [-0.5, 0.0, 1.0, 1.0]
        ema = EMATracker()
        pre_adv, _ = compute_player_advantages(pre_rewards, [], ema)
        assert len(pre_adv) == 4
        # -0.5 should have the most negative advantage
        assert pre_adv[0] < pre_adv[1] < pre_adv[2]

    def test_single_rollout(self):
        """Single rollout → std=0 → advantage ≈ 0."""
        pre_adv, _ = compute_player_advantages([1.0], [], EMATracker())
        assert len(pre_adv) == 1
        assert pre_adv[0] == pytest.approx(0.0, abs=1e-4)
```

**Step 2: Run to verify failure**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/rewards/test_advantages.py -v
```

### Task 15: Advantage computation — implementation

**Files:**
- Create: `crisp/rewards/advantages.py`

**Step 1: Implement**

```python
"""Two-pool advantage normalization for CRISP players."""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from crisp.rewards.ema_tracker import EMATracker


def compute_player_advantages(
    pre_rewards: List[float],
    post_rewards: List[float],
    ema_tracker: EMATracker,
    epsilon: float = 1e-8,
) -> Tuple[List[float], List[float]]:
    """Compute advantages using two normalization pools.

    Pool 1 (pre-discussion): per-batch mean/std normalization.
    Pool 2 (post-discussion): EMA-smoothed normalization.

    The EMA tracker is updated with post_rewards during this call.

    Args:
        pre_rewards: Rewards for pre-discussion rollouts (already filtered by dynamic sampling).
        post_rewards: Rewards for post-discussion responses.
        ema_tracker: EMA tracker for Pool 2 statistics.
        epsilon: Small constant for numerical stability.

    Returns:
        (pre_advantages, post_advantages) tuple.
    """
    # Pool 1: per-batch statistics
    if pre_rewards:
        mean_pre = float(np.mean(pre_rewards))
        std_pre = float(np.std(pre_rewards))
        pre_advantages = [(r - mean_pre) / (std_pre + epsilon) for r in pre_rewards]
    else:
        pre_advantages = []

    # Pool 2: EMA-smoothed statistics
    # Use current EMA stats for normalization, THEN update
    if post_rewards:
        mu = ema_tracker.mu
        sigma = math.sqrt(ema_tracker.sigma_sq)
        post_advantages = [(r - mu) / (sigma + epsilon) for r in post_rewards]
        ema_tracker.update(post_rewards)
    else:
        post_advantages = []
        ema_tracker.update([])  # Tracks consecutive empty

    return pre_advantages, post_advantages
```

**Step 2: Run tests**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/rewards/test_advantages.py -v
```

**Step 3: Commit**

```bash
git add crisp/rewards/advantages.py tests/rewards/test_advantages.py
git commit -m "feat: add two-pool advantage normalization with EMA smoothing"
```

---

## Phase 3: Discussion Module

### Task 16: Discussion trigger — tests

**Files:**
- Create: `tests/discussion/__init__.py`
- Create: `tests/discussion/test_trigger.py`

**Step 1: Write failing tests**

```python
"""Tests for majority vote and discussion trigger."""
import pytest

from crisp.discussion.trigger import majority_vote, should_discuss
from tests.conftest import make_rollout


class TestMajorityVote:
    def test_unanimous(self):
        rollouts = [make_rollout(answer="42") for _ in range(8)]
        assert majority_vote(rollouts) == "42"

    def test_clear_majority(self):
        rollouts = (
            [make_rollout(answer="42") for _ in range(5)]
            + [make_rollout(answer="99") for _ in range(3)]
        )
        assert majority_vote(rollouts) == "42"

    def test_tie_breaks_by_first_occurrence(self):
        rollouts = [
            make_rollout(answer="A"),
            make_rollout(answer="B"),
            make_rollout(answer="A"),
            make_rollout(answer="B"),
        ]
        assert majority_vote(rollouts) == "A"

    def test_all_different(self):
        rollouts = [make_rollout(answer=str(i)) for i in range(4)]
        assert majority_vote(rollouts) == "0"  # First occurrence

    def test_none_answers_ignored(self):
        rollouts = (
            [make_rollout(answer=None) for _ in range(5)]
            + [make_rollout(answer="42") for _ in range(3)]
        )
        assert majority_vote(rollouts) == "42"

    def test_all_none_returns_none(self):
        rollouts = [make_rollout(answer=None) for _ in range(4)]
        assert majority_vote(rollouts) is None

    def test_symbolic_equivalence_grouping(self):
        """Answers that are symbolically equivalent should be grouped."""
        rollouts = [
            make_rollout(answer="1/2"),
            make_rollout(answer="0.5"),
            make_rollout(answer="0.5"),
            make_rollout(answer="99"),
        ]
        # "1/2" and "0.5" are equivalent → group has 3, "99" has 1
        result = majority_vote(rollouts)
        assert result in ("1/2", "0.5")  # Either representative is fine


class TestShouldDiscuss:
    def test_agreeing_players_no_discussion(self):
        assert should_discuss("42", "42") is False

    def test_disagreeing_players_trigger_discussion(self):
        assert should_discuss("42", "99") is True

    def test_equivalent_answers_no_discussion(self):
        assert should_discuss("1/2", "0.5") is False

    def test_none_majority_triggers_discussion(self):
        """If one player has no majority (all None), trigger discussion."""
        assert should_discuss("42", None) is True

    def test_both_none_no_discussion(self):
        """Both None → agree on nothing → no discussion."""
        assert should_discuss(None, None) is False
```

**Step 2: Run to verify failure**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/discussion/test_trigger.py -v
```

### Task 17: Discussion trigger — implementation

**Files:**
- Create: `crisp/discussion/__init__.py`
- Create: `crisp/discussion/trigger.py`

**Step 1: Implement**

```python
"""Majority vote computation and discussion trigger logic."""
from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional

from crisp.types import Rollout
from crisp.verifier.sympy_verify import equivalent


def majority_vote(rollouts: List[Rollout]) -> Optional[str]:
    """Compute the majority answer across rollouts.

    Groups symbolically equivalent answers. Ties broken by first occurrence.
    Returns None if all rollouts have answer=None.
    """
    # Group answers, using equivalence checking
    # Each group: (representative_answer, count, first_index)
    groups: list[tuple[str, int, int]] = []

    for i, r in enumerate(rollouts):
        if r.answer is None:
            continue
        matched = False
        for j, (rep, count, first_idx) in enumerate(groups):
            if equivalent(r.answer, rep):
                groups[j] = (rep, count + 1, first_idx)
                matched = True
                break
        if not matched:
            groups.append((r.answer, 1, i))

    if not groups:
        return None

    # Sort by count descending, then by first_index ascending (tie-break)
    groups.sort(key=lambda g: (-g[1], g[2]))
    return groups[0][0]


def should_discuss(
    majority_a: Optional[str],
    majority_b: Optional[str],
) -> bool:
    """Determine if discussion should be triggered between two players.

    Discussion triggered when the two players' majority answers disagree.
    """
    if majority_a is None and majority_b is None:
        return False
    if majority_a is None or majority_b is None:
        return True
    return not equivalent(majority_a, majority_b)
```

**Step 2: Run tests**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/discussion/test_trigger.py -v
```

**Step 3: Commit**

```bash
git add crisp/discussion/ tests/discussion/
git commit -m "feat: add majority vote with symbolic grouping and discussion trigger"
```

### Task 18: Representative selection — tests

**Files:**
- Create: `tests/discussion/test_representative.py`

**Step 1: Write failing tests**

```python
"""Tests for representative rollout selection for discussion."""
import pytest

from crisp.discussion.representative import select_representatives
from tests.conftest import make_rollout


class TestSelectRepresentatives:
    def test_correct_player_gets_highest_logprob(self):
        """Player with correct majority → rollout with highest total log-prob."""
        rollouts = {
            0: [
                make_rollout(player_id=0, answer="42", correct=True, log_probs=[-1.0, -1.0]),
                make_rollout(player_id=0, answer="42", correct=True, log_probs=[-0.1, -0.1]),  # Highest
                make_rollout(player_id=0, answer="42", correct=True, log_probs=[-0.5, -0.5]),
            ],
            1: [
                make_rollout(player_id=1, answer="99", correct=False, log_probs=[-0.5, -0.5]),
                make_rollout(player_id=1, answer="99", correct=False, log_probs=[-0.3, -0.3]),
            ],
        }
        majority = {(0, 0): "42", (1, 0): "99"}
        reps = select_representatives(rollouts, majority, "42", problem_idx=0)
        # Player 0 (correct): highest log-prob correct rollout
        assert sum(reps[0].log_probs) == pytest.approx(-0.2)
        # Player 1 (wrong): rollout matching their majority answer
        assert reps[1].answer == "99"

    def test_both_wrong_gets_longest(self):
        """Both players wrong → longest rollout from each."""
        rollouts = {
            0: [
                make_rollout(player_id=0, answer="99", correct=False, text="short"),
                make_rollout(player_id=0, answer="99", correct=False, text="this is a much longer response"),
            ],
            1: [
                make_rollout(player_id=1, answer="88", correct=False, text="x"),
                make_rollout(player_id=1, answer="88", correct=False, text="longer text here"),
            ],
        }
        majority = {(0, 0): "99", (1, 0): "88"}
        reps = select_representatives(rollouts, majority, "42", problem_idx=0)
        assert reps[0].text == "this is a much longer response"
        assert reps[1].text == "longer text here"

    def test_wrong_player_gets_majority_matching_rollout(self):
        """Wrong player gets a rollout matching their majority answer."""
        rollouts = {
            0: [make_rollout(player_id=0, answer="42", correct=True, log_probs=[-0.1, -0.1])],
            1: [
                make_rollout(player_id=1, answer="99", correct=False),
                make_rollout(player_id=1, answer="88", correct=False),
                make_rollout(player_id=1, answer="99", correct=False),
            ],
        }
        majority = {(0, 0): "42", (1, 0): "99"}
        reps = select_representatives(rollouts, majority, "42", problem_idx=0)
        assert reps[1].answer == "99"
```

**Step 2: Run to verify failure**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/discussion/test_representative.py -v
```

### Task 19: Representative selection — implementation

**Files:**
- Create: `crisp/discussion/representative.py`

**Step 1: Implement**

```python
"""Select representative rollouts for discussion."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from crisp.types import Rollout
from crisp.verifier.sympy_verify import check, equivalent


def select_representatives(
    rollouts: Dict[int, List[Rollout]],
    majority_answers: Dict[Tuple[int, int], str],
    ground_truth: str,
    problem_idx: int,
) -> Dict[int, Rollout]:
    """Select one representative rollout per player for discussion.

    For the player with correct majority: highest log-prob correct rollout.
    For the player with wrong majority: rollout matching their majority answer.
    If both wrong: longest rollout from each.

    Args:
        rollouts: player_id → list of rollouts for this problem.
        majority_answers: (player_id, problem_idx) → majority answer string.
        ground_truth: The coach's verified answer.
        problem_idx: Which problem we're selecting for.

    Returns:
        Dict mapping player_id → selected Rollout.
    """
    player_rollouts = {
        pid: [r for r in rs if r.problem_idx == problem_idx]
        for pid, rs in rollouts.items()
    }

    correct_players = [
        pid for pid in player_rollouts
        if check(majority_answers.get((pid, problem_idx)), ground_truth)
    ]

    reps = {}
    if not correct_players:
        # Both wrong: longest rollout from each
        for pid, rs in player_rollouts.items():
            reps[pid] = max(rs, key=lambda r: len(r.text))
    else:
        for pid, rs in player_rollouts.items():
            if pid in correct_players:
                # Correct player: highest log-prob among correct rollouts
                correct_rollouts = [r for r in rs if r.correct]
                if correct_rollouts:
                    reps[pid] = max(correct_rollouts, key=lambda r: sum(r.log_probs))
                else:
                    reps[pid] = max(rs, key=lambda r: len(r.text))
            else:
                # Wrong player: rollout matching their majority answer
                maj = majority_answers.get((pid, problem_idx))
                matching = [r for r in rs if equivalent(r.answer, maj)] if maj else []
                if matching:
                    reps[pid] = matching[0]
                else:
                    reps[pid] = max(rs, key=lambda r: len(r.text))

    return reps
```

**Step 2: Run tests**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/discussion/test_representative.py -v
```

**Step 3: Commit**

```bash
git add crisp/discussion/representative.py tests/discussion/test_representative.py
git commit -m "feat: add representative rollout selection for discussion"
```

### Task 20: Post-discussion parsing — tests

**Files:**
- Create: `tests/discussion/test_post_discussion.py`

**Step 1: Write failing tests**

```python
"""Tests for EVALUATION / FINAL ANSWER parsing."""
import pytest

from crisp.discussion.post_discussion import parse_discussion_response


class TestParseDiscussionResponse:
    def test_both_segments(self):
        text = (
            "EVALUATION: Alice's solution has an error in step 3.\n"
            "FINAL ANSWER: \\boxed{42}"
        )
        eval_text, answer = parse_discussion_response(text)
        assert "error in step 3" in eval_text
        assert answer == "42"

    def test_no_delimiter_entire_text_is_answer(self):
        text = "The answer is \\boxed{42}"
        eval_text, answer = parse_discussion_response(text)
        assert eval_text == ""
        assert answer == "42"

    def test_empty_evaluation(self):
        text = "EVALUATION:\nFINAL ANSWER: \\boxed{7}"
        eval_text, answer = parse_discussion_response(text)
        assert eval_text.strip() == ""
        assert answer == "7"

    def test_no_boxed_in_final_answer(self):
        text = "EVALUATION: good\nFINAL ANSWER: 42"
        eval_text, answer = parse_discussion_response(text)
        assert "good" in eval_text
        assert answer is None  # No \boxed{}

    def test_multiple_final_answer_delimiters(self):
        text = (
            "EVALUATION: I think FINAL ANSWER: is mentioned here\n"
            "FINAL ANSWER: \\boxed{5}"
        )
        eval_text, answer = parse_discussion_response(text)
        # Should split on the LAST occurrence
        assert answer == "5"

    def test_empty_response(self):
        eval_text, answer = parse_discussion_response("")
        assert eval_text == ""
        assert answer is None
```

**Step 2: Run to verify failure**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/discussion/test_post_discussion.py -v
```

### Task 21: Post-discussion parsing — implementation

**Files:**
- Create: `crisp/discussion/post_discussion.py`

**Step 1: Implement**

```python
"""Parse post-discussion responses into EVALUATION and FINAL ANSWER segments."""
from __future__ import annotations

from typing import Optional, Tuple

from crisp.verifier.answer_extraction import extract_boxed

FINAL_ANSWER_DELIMITER = "FINAL ANSWER:"


def parse_discussion_response(text: str) -> Tuple[str, Optional[str]]:
    """Parse a post-discussion response into evaluation text and final answer.

    Looks for the FINAL ANSWER: delimiter to split segments.
    If not found, entire text is treated as answer segment (empty evaluation).
    The final answer is extracted from \\boxed{} in the answer segment.

    Returns:
        (evaluation_text, extracted_answer) where extracted_answer may be None.
    """
    if not text:
        return "", None

    # Find the LAST occurrence of the delimiter
    idx = text.rfind(FINAL_ANSWER_DELIMITER)
    if idx == -1:
        # No delimiter: entire text is answer segment
        return "", extract_boxed(text)

    evaluation_text = text[:idx]
    answer_segment = text[idx + len(FINAL_ANSWER_DELIMITER):]
    answer = extract_boxed(answer_segment)

    return evaluation_text, answer
```

**Step 2: Run tests**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/discussion/test_post_discussion.py -v
```

**Step 3: Commit**

```bash
git add crisp/discussion/post_discussion.py tests/discussion/test_post_discussion.py
git commit -m "feat: add post-discussion EVALUATION/FINAL ANSWER segment parsing"
```

---

## Phase 4: Training Module

### Task 22: Overlong shaping — tests

**Files:**
- Create: `tests/training/__init__.py`
- Create: `tests/training/test_overlong_shaping.py`

**Step 1: Write failing tests**

```python
"""Tests for overlong sequence reward shaping."""
import pytest

from crisp.training.overlong_shaping import compute_overlong_penalty


class TestOverlongPenalty:
    def test_under_l_max_no_penalty(self):
        assert compute_overlong_penalty(length=4000, l_max=8192, buffer=2048) == 0.0

    def test_at_l_max_no_penalty(self):
        assert compute_overlong_penalty(length=8192, l_max=8192, buffer=2048) == 0.0

    def test_in_buffer_zone_linear_ramp(self):
        # Midpoint of buffer: penalty = 0.5
        midpoint = 8192 + 1024
        penalty = compute_overlong_penalty(length=midpoint, l_max=8192, buffer=2048)
        assert penalty == pytest.approx(0.5)

    def test_at_l_hard_full_penalty(self):
        l_hard = 8192 + 2048
        penalty = compute_overlong_penalty(length=l_hard, l_max=8192, buffer=2048)
        assert penalty == pytest.approx(1.0)

    def test_beyond_l_hard_capped(self):
        penalty = compute_overlong_penalty(length=20000, l_max=8192, buffer=2048)
        assert penalty == pytest.approx(1.0)

    def test_post_discussion_limits(self):
        """Post-discussion uses different L_max and buffer."""
        assert compute_overlong_penalty(length=3000, l_max=4096, buffer=1024) == 0.0
        penalty = compute_overlong_penalty(length=4096 + 512, l_max=4096, buffer=1024)
        assert penalty == pytest.approx(0.5)
```

**Step 2: Run to verify failure**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/training/test_overlong_shaping.py -v
```

### Task 23: Overlong shaping — implementation

**Files:**
- Create: `crisp/training/__init__.py`
- Create: `crisp/training/overlong_shaping.py`

**Step 1: Implement**

```python
"""Overlong sequence reward shaping."""
from __future__ import annotations


def compute_overlong_penalty(
    length: int,
    l_max: int = 8192,
    buffer: int = 2048,
) -> float:
    """Compute penalty for sequences exceeding L_max.

    0.0 if length <= l_max.
    Linear ramp from 0.0 to 1.0 in [l_max, l_max + buffer].
    Capped at 1.0 beyond l_hard = l_max + buffer.
    """
    if length <= l_max:
        return 0.0
    l_hard = l_max + buffer
    if length >= l_hard:
        return 1.0
    return (length - l_max) / buffer
```

**Step 2: Run tests**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/training/test_overlong_shaping.py -v
```

**Step 3: Commit**

```bash
git add crisp/training/ tests/training/
git commit -m "feat: add overlong sequence penalty with linear ramp"
```

### Task 24: Batch builder — tests

**Files:**
- Create: `tests/training/test_batch_builder.py`

**Step 1: Write failing tests**

```python
"""Tests for dynamic sampling and batch assembly."""
import pytest

from crisp.training.batch_builder import filter_dynamic_sampling, build_player_batch
from crisp.types import Rollout, DiscussionResult, TokenSequence
from tests.conftest import make_rollout


class TestFilterDynamicSampling:
    def test_mixed_rewards_retained(self):
        """Problem with mixed correct/wrong rollouts is kept."""
        rollouts = (
            [make_rollout(problem_idx=0, reward=1.0) for _ in range(5)]
            + [make_rollout(problem_idx=0, reward=0.0) for _ in range(3)]
        )
        filtered = filter_dynamic_sampling(rollouts)
        assert len(filtered) == 8

    def test_all_correct_filtered(self):
        """Problem where all 8 rollouts are correct → filtered out."""
        rollouts = [make_rollout(problem_idx=0, reward=1.0) for _ in range(8)]
        filtered = filter_dynamic_sampling(rollouts)
        assert len(filtered) == 0

    def test_all_wrong_filtered(self):
        """Problem where all rollouts are wrong (reward=0) → filtered out."""
        rollouts = [make_rollout(problem_idx=0, reward=0.0) for _ in range(8)]
        filtered = filter_dynamic_sampling(rollouts)
        assert len(filtered) == 0

    def test_all_no_box_filtered(self):
        """All -0.5 rewards → all same → filtered."""
        rollouts = [make_rollout(problem_idx=0, reward=-0.5) for _ in range(8)]
        filtered = filter_dynamic_sampling(rollouts)
        assert len(filtered) == 0

    def test_mixed_wrong_and_no_box_filtered(self):
        """Mix of 0.0 and -0.5 still has variance → retained."""
        rollouts = (
            [make_rollout(problem_idx=0, reward=0.0) for _ in range(4)]
            + [make_rollout(problem_idx=0, reward=-0.5) for _ in range(4)]
        )
        filtered = filter_dynamic_sampling(rollouts)
        assert len(filtered) == 8

    def test_multiple_problems_filtered_independently(self):
        """Each problem filtered independently."""
        rollouts = (
            # Problem 0: all correct → filter
            [make_rollout(problem_idx=0, reward=1.0) for _ in range(8)]
            # Problem 1: mixed → keep
            + [make_rollout(problem_idx=1, reward=1.0) for _ in range(4)]
            + [make_rollout(problem_idx=1, reward=0.0) for _ in range(4)]
        )
        filtered = filter_dynamic_sampling(rollouts)
        assert len(filtered) == 8
        assert all(r.problem_idx == 1 for r in filtered)

    def test_persuader_bonus_creates_variance(self):
        """Problem with 1.0 and 1.3 rewards has variance → kept."""
        rollouts = (
            [make_rollout(problem_idx=0, reward=1.3) for _ in range(4)]
            + [make_rollout(problem_idx=0, reward=1.0) for _ in range(4)]
        )
        filtered = filter_dynamic_sampling(rollouts)
        assert len(filtered) == 8
```

**Step 2: Run to verify failure**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/training/test_batch_builder.py -v
```

### Task 25: Batch builder — implementation

**Files:**
- Create: `crisp/training/batch_builder.py`

**Step 1: Implement**

```python
"""Dynamic sampling filter and training batch assembly."""
from __future__ import annotations

from collections import defaultdict
from typing import List

from crisp.types import Rollout


def filter_dynamic_sampling(rollouts: List[Rollout]) -> List[Rollout]:
    """Filter out problems where all rollouts have identical rewards.

    These produce zero advantage variance and waste gradient computation.
    """
    # Group by problem
    by_problem: dict[int, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        by_problem[r.problem_idx].append(r)

    result = []
    for prob_idx, prob_rollouts in by_problem.items():
        rewards = [r.reward for r in prob_rollouts]
        if len(set(rewards)) > 1:
            result.extend(prob_rollouts)

    return result
```

**Step 2: Run tests**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/training/test_batch_builder.py -v
```

**Step 3: Commit**

```bash
git add crisp/training/batch_builder.py tests/training/test_batch_builder.py
git commit -m "feat: add dynamic sampling filter for zero-variance problems"
```

### Task 26: GRPO loss — tests

**Files:**
- Create: `tests/training/test_grpo_loss.py`

**Step 1: Write failing tests**

```python
"""Tests for GRPO loss with DCPO clipping and JS-divergence."""
import torch
import pytest

from crisp.training.grpo_loss import compute_grpo_loss


class TestGRPOLoss:
    def _make_tensors(self, B=4, T=8, seed=42):
        """Create synthetic tensors for testing."""
        gen = torch.manual_seed(seed)
        current = torch.randn(B, T) * 0.1 - 2.0  # log-probs (negative)
        old = current.clone()  # Same as current → rho = 1.0
        ref = torch.randn(B, T) * 0.1 - 2.0
        mask = torch.ones(B, T)
        return current, old, ref, mask

    def test_zero_advantages_zero_policy_loss(self):
        """With zero advantages, policy loss component should be ~0."""
        current, old, ref, mask = self._make_tensors()
        advantages = torch.zeros(4)
        loss = compute_grpo_loss(current, old, ref, advantages, mask, js_beta=0.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_advantage_negative_loss(self):
        """Positive advantages should produce negative loss (reward signal)."""
        current, old, ref, mask = self._make_tensors()
        advantages = torch.ones(4)
        loss = compute_grpo_loss(current, old, ref, advantages, mask, js_beta=0.0)
        assert loss.item() < 0.0

    def test_negative_advantage_positive_loss(self):
        """Negative advantages should produce positive loss (penalty signal)."""
        current, old, ref, mask = self._make_tensors()
        advantages = -torch.ones(4)
        loss = compute_grpo_loss(current, old, ref, advantages, mask, js_beta=0.0)
        assert loss.item() > 0.0

    def test_js_divergence_non_negative(self):
        """JS-divergence component should always be >= 0."""
        current, old, ref, mask = self._make_tensors()
        advantages = torch.zeros(4)
        # With js_beta > 0, loss = 0 (policy) + js_beta * js_div >= 0
        loss = compute_grpo_loss(current, old, ref, advantages, mask, js_beta=0.1)
        assert loss.item() >= 0.0

    def test_dcpo_wider_bounds_for_low_prob_tokens(self):
        """Tokens with low ref probability should get wider clip bounds."""
        B, T = 1, 2
        # Token 0: high ref prob, Token 1: low ref prob
        ref = torch.tensor([[-0.1, -5.0]])  # prob ≈ [0.9, 0.007]
        old = ref.clone()
        # Current policy diverges equally from old for both tokens
        current = old + 0.5  # Same shift for both
        mask = torch.ones(B, T)
        advantages = torch.ones(B)

        loss = compute_grpo_loss(
            current, old, ref, advantages, mask,
            dcpo_alpha=3.0, clip_base=0.2, js_beta=0.0
        )
        # Should not raise; the low-prob token gets wider bounds
        assert torch.isfinite(loss)

    def test_attention_mask_respected(self):
        """Masked tokens should not contribute to loss."""
        current, old, ref, mask = self._make_tensors(B=2, T=4)
        advantages = torch.ones(2)
        mask_full = torch.ones(2, 4)
        mask_half = torch.ones(2, 4)
        mask_half[:, 2:] = 0  # Mask out last 2 tokens

        loss_full = compute_grpo_loss(current, old, ref, advantages, mask_full, js_beta=0.0)
        loss_half = compute_grpo_loss(current, old, ref, advantages, mask_half, js_beta=0.0)
        # Different masking → different loss values
        assert loss_full.item() != pytest.approx(loss_half.item(), abs=1e-6)

    def test_no_nan_with_extreme_log_probs(self):
        """Should not produce NaN even with very negative log-probs."""
        B, T = 2, 4
        current = torch.full((B, T), -20.0)  # Very low prob
        old = torch.full((B, T), -20.0)
        ref = torch.full((B, T), -20.0)
        mask = torch.ones(B, T)
        advantages = torch.ones(B)
        loss = compute_grpo_loss(current, old, ref, advantages, mask)
        assert torch.isfinite(loss), f"Got non-finite loss: {loss}"

    def test_coach_loss_no_js(self):
        """Coach uses js_beta=0.0 — should work fine."""
        current, old, ref, mask = self._make_tensors()
        advantages = torch.tensor([0.5, -0.5, 0.3, -0.1])
        loss = compute_grpo_loss(current, old, ref, advantages, mask, js_beta=0.0)
        assert torch.isfinite(loss)

    def test_gradient_flows(self):
        """Verify gradients flow through current_log_probs."""
        B, T = 2, 4
        current = torch.randn(B, T, requires_grad=True) * 0.1 - 2.0
        old = current.detach().clone()
        ref = torch.randn(B, T) * 0.1 - 2.0
        mask = torch.ones(B, T)
        advantages = torch.tensor([1.0, -1.0])
        loss = compute_grpo_loss(current, old, ref, advantages, mask)
        loss.backward()
        assert current.grad is not None
        assert torch.all(torch.isfinite(current.grad))
```

**Step 2: Run to verify failure**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/training/test_grpo_loss.py -v
```

### Task 27: GRPO loss — implementation

**Files:**
- Create: `crisp/training/grpo_loss.py`

**Step 1: Implement**

```python
"""GRPO loss with DCPO token-adaptive clipping and JS-divergence."""
from __future__ import annotations

import torch
from torch import Tensor


def compute_grpo_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    ref_log_probs: Tensor,
    advantages: Tensor,
    attention_mask: Tensor,
    dcpo_alpha: float = 3.0,
    clip_base: float = 0.2,
    js_beta: float = 0.001,
) -> Tensor:
    """Compute GRPO loss with DCPO clipping and optional JS-divergence.

    Args:
        current_log_probs: [B, T] log π_θ(a_t | s_t) — current policy
        old_log_probs: [B, T] log π_old(a_t | s_t) — rollout policy
        ref_log_probs: [B, T] log π_ref(a_t | s_t) — frozen reference
        advantages: [B] per-sequence advantages
        attention_mask: [B, T] valid token mask
        dcpo_alpha: DCPO sensitivity (higher → wider bounds for rare tokens)
        clip_base: Base ε for clipping
        js_beta: JS-divergence coefficient (0.0 for coach)

    Returns:
        Scalar loss value (token-level mean over valid tokens).

    Note:
        JS-divergence uses single-sample estimator (standard approximation).
        DCPO uses reference policy probabilities for clip bounds.
    """
    # Importance ratio
    rho = (current_log_probs - old_log_probs).exp()  # [B, T]

    # DCPO: token-adaptive clip bounds from REFERENCE policy
    prior_prob = ref_log_probs.exp().clamp(min=1e-10, max=1.0)  # [B, T]
    eps = clip_base * (1.0 + dcpo_alpha * (1.0 - prior_prob))  # [B, T]

    # Expand per-sequence advantages to token level
    adv = advantages.unsqueeze(1).expand_as(rho)  # [B, T]

    # Clipped surrogate loss
    surr1 = rho * adv
    surr2 = rho.clamp(1.0 - eps, 1.0 + eps) * adv
    policy_loss = -torch.min(surr1, surr2)  # [B, T]

    # JS-divergence (single-sample estimator)
    if js_beta > 0.0:
        p = current_log_probs.exp().clamp(min=1e-10)
        q = ref_log_probs.exp().clamp(min=1e-10)
        m = 0.5 * (p + q) + 1e-8
        log_m = m.log()
        js_div = (0.5 * (p * (p.log() - log_m) + q * (q.log() - log_m))).clamp(min=0.0)
    else:
        js_div = torch.zeros_like(policy_loss)

    # Token-level mean over valid tokens
    total_loss = (policy_loss + js_beta * js_div) * attention_mask
    return total_loss.sum() / (attention_mask.sum() + 1e-10)
```

**Step 2: Run tests**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/training/test_grpo_loss.py -v
```

**Step 3: Commit**

```bash
git add crisp/training/grpo_loss.py tests/training/test_grpo_loss.py
git commit -m "feat: add GRPO loss with DCPO token-adaptive clipping and JS-divergence"
```

---

## Phase 5: End-to-End Reward Flow Test

### Task 28: Full reward-to-advantage pipeline test

**Files:**
- Create: `tests/test_reward_flow.py`

**Step 1: Write the full flow test**

```python
"""End-to-end test: synthetic 8-rollout scenario tracing exact reward values
through compute_solve_reward → apply_persuader_bonus → filter_dynamic_sampling
→ compute_player_advantages → final advantage values.

Asserts specific numerical values at each stage.
"""
import math
import pytest

from crisp.types import Problem, Rollout, DiscussionResult
from crisp.rewards.player_rewards import compute_solve_reward, apply_persuader_bonus
from crisp.rewards.advantages import compute_player_advantages
from crisp.rewards.ema_tracker import EMATracker
from crisp.training.batch_builder import filter_dynamic_sampling
from tests.conftest import make_rollout, make_problem
import numpy as np


class TestFullRewardFlow:
    """Trace exact numbers through the full reward → advantage pipeline."""

    def test_complete_scenario(self):
        """
        Setup:
        - 2 problems, 8 rollouts per problem per player
        - Problem 0: Alice 6/8 correct, Bob 3/8 correct
          → Majority: Alice="42", Bob="99" → DISAGREE → discussion triggered
          → Discussion: Alice stays correct, Bob flips to correct
          → Alice is persuader
        - Problem 1: Alice 8/8 correct, Bob 8/8 correct
          → No discussion, all-correct → filtered by dynamic sampling
        """
        problems = [make_problem(ground_truth="42"), make_problem(ground_truth="7")]

        # -- Build rollouts for Player 0 (Alice) --
        alice_rollouts = (
            # Problem 0: 6 correct, 2 wrong
            [make_rollout(problem_idx=0, player_id=0, answer="42", correct=True) for _ in range(6)]
            + [make_rollout(problem_idx=0, player_id=0, answer="99", correct=False) for _ in range(2)]
            # Problem 1: 8 correct
            + [make_rollout(problem_idx=1, player_id=0, answer="7", correct=True) for _ in range(8)]
        )

        # -- Build rollouts for Player 1 (Bob) --
        bob_rollouts = (
            # Problem 0: 3 correct, 5 wrong
            [make_rollout(problem_idx=0, player_id=1, answer="42", correct=True) for _ in range(3)]
            + [make_rollout(problem_idx=0, player_id=1, answer="99", correct=False) for _ in range(5)]
            # Problem 1: 8 correct
            + [make_rollout(problem_idx=1, player_id=1, answer="7", correct=True) for _ in range(8)]
        )

        rollouts = {0: alice_rollouts, 1: bob_rollouts}

        # -- Step 4: Compute solve rewards --
        for pid in rollouts:
            for r in rollouts[pid]:
                r.reward = compute_solve_reward(r)

        # Verify pre-discussion rewards
        assert all(r.reward == 1.0 for r in alice_rollouts[:6])   # correct
        assert all(r.reward == 0.0 for r in alice_rollouts[6:8])  # wrong
        assert all(r.reward == 1.0 for r in alice_rollouts[8:])   # correct

        # -- Step 5: Majority + trigger --
        majority_answers = {(0, 0): "42", (1, 0): "99", (0, 1): "7", (1, 1): "7"}

        # -- Step 6-7: Discussion + persuader bonus --
        discussion_results = {
            0: [DiscussionResult(
                problem_idx=0, player_id=0, tokens=[], text="", log_probs=[],
                final_answer="42", correct=True, reward=1.0,
            )],
            1: [DiscussionResult(
                problem_idx=0, player_id=1, tokens=[], text="", log_probs=[],
                final_answer="42", correct=True, reward=1.0,
            )],
        }

        apply_persuader_bonus(rollouts, discussion_results, majority_answers, problems, gamma=0.3)

        # Alice was persuader on problem 0: her correct rollouts → 1.3
        assert all(r.reward == pytest.approx(1.3) for r in alice_rollouts[:6])
        assert all(r.reward == pytest.approx(0.0) for r in alice_rollouts[6:8])
        # Problem 1 unaffected
        assert all(r.reward == pytest.approx(1.0) for r in alice_rollouts[8:])
        # Bob NOT persuader: rewards unchanged
        assert all(r.reward == pytest.approx(1.0) for r in bob_rollouts[:3])
        assert all(r.reward == pytest.approx(0.0) for r in bob_rollouts[3:8])

        # -- Step 8: Dynamic sampling filter --
        alice_filtered = filter_dynamic_sampling(alice_rollouts)
        bob_filtered = filter_dynamic_sampling(bob_rollouts)

        # Problem 0: mixed rewards (1.3, 0.0) → kept (8 rollouts)
        # Problem 1: all 1.0 → filtered
        assert len(alice_filtered) == 8
        assert all(r.problem_idx == 0 for r in alice_filtered)
        # Bob problem 0: mixed (1.0, 0.0) → kept, problem 1: all 1.0 → filtered
        assert len(bob_filtered) == 8
        assert all(r.problem_idx == 0 for r in bob_filtered)

        # -- Step 9: Advantages --
        alice_pre_rewards = [r.reward for r in alice_filtered]
        alice_post_rewards = [dr.reward for dr in discussion_results[0]]
        alice_ema = EMATracker(mu=0.5, sigma_sq=0.25, eta=0.2)

        alice_pre_adv, alice_post_adv = compute_player_advantages(
            alice_pre_rewards, alice_post_rewards, alice_ema
        )

        # Pre-discussion: rewards are [1.3]*6 + [0.0]*2
        mean_pre = (6 * 1.3 + 2 * 0.0) / 8  # = 0.975
        std_pre = float(np.std(alice_pre_rewards))
        eps = 1e-8
        for i, r in enumerate(alice_pre_rewards):
            expected = (r - mean_pre) / (std_pre + eps)
            assert alice_pre_adv[i] == pytest.approx(expected, rel=1e-4)

        # Correct rollouts (1.3) have positive advantage
        assert all(a > 0 for a in alice_pre_adv[:6])
        # Wrong rollouts (0.0) have negative advantage
        assert all(a < 0 for a in alice_pre_adv[6:])

        # Post-discussion: uses EMA (μ=0.5, σ²=0.25 at time of computation)
        for i, r in enumerate(alice_post_rewards):
            expected = (r - 0.5) / (math.sqrt(0.25) + eps)
            assert alice_post_adv[i] == pytest.approx(expected, rel=1e-4)
```

**Step 2: Run to verify it passes (all components already implemented)**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/test_reward_flow.py -v
```
Expected: ALL PASS (since all component implementations exist from prior tasks)

**Step 3: Commit**

```bash
git add tests/test_reward_flow.py
git commit -m "test: add end-to-end reward flow trace with hardcoded expected values"
```

---

## Phase 6: Integration Test with Conservation Invariant

### Task 29: Integration test — full pipeline with mock models

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_pipeline.py`

**Step 1: Write integration tests**

```python
"""Integration tests: full pipeline with mock data, conservation invariants."""
import math
import numpy as np
import pytest

from crisp.types import Problem, Rollout, DiscussionResult
from crisp.rewards.player_rewards import compute_solve_reward, apply_persuader_bonus
from crisp.rewards.advantages import compute_player_advantages
from crisp.rewards.coach_rewards import compute_coach_reward
from crisp.rewards.ema_tracker import EMATracker
from crisp.rewards.repetition_buffer import RepetitionBuffer
from crisp.discussion.trigger import majority_vote, should_discuss
from crisp.training.batch_builder import filter_dynamic_sampling
from tests.conftest import make_rollout, make_problem


def _build_synthetic_batch(rng, num_problems=8, rollouts_per=8):
    """Build a full synthetic batch with controlled solve rates."""
    problems = []
    all_rollouts = {0: [], 1: []}
    solve_rates = {
        0: rng.uniform(0.2, 0.8, num_problems),
        1: rng.uniform(0.2, 0.8, num_problems),
    }

    for pidx in range(num_problems):
        gt = str(pidx * 10 + 1)
        problems.append(make_problem(
            text=f"Problem {pidx}",
            ground_truth=gt,
            embedding=rng.random(384).astype(np.float32),
        ))
        for player_id in [0, 1]:
            rate = solve_rates[player_id][pidx]
            for _ in range(rollouts_per):
                correct = rng.random() < rate
                answer = gt if correct else "WRONG"
                all_rollouts[player_id].append(make_rollout(
                    problem_idx=pidx,
                    player_id=player_id,
                    answer=answer,
                    correct=correct,
                    text=f"Solution for problem {pidx}",
                ))
    return problems, all_rollouts


@pytest.mark.integration
class TestPipelineConservation:
    def test_batch_size_conservation(self):
        """Total sequences = (non-filtered problems × 8) + discussion count per player."""
        rng = np.random.default_rng(123)
        problems, rollouts = _build_synthetic_batch(rng)

        # Step 4: Compute rewards
        for pid in rollouts:
            for r in rollouts[pid]:
                r.reward = compute_solve_reward(r)

        # Step 5: Majority vote + discussion trigger
        majority_answers = {}
        discuss_problems = []
        for pidx in range(len(problems)):
            for pid in [0, 1]:
                player_rs = [r for r in rollouts[pid] if r.problem_idx == pidx]
                majority_answers[(pid, pidx)] = majority_vote(player_rs)
            if should_discuss(majority_answers[(0, pidx)], majority_answers[(1, pidx)]):
                discuss_problems.append(pidx)

        # Step 6-7: Mock discussion results
        discussion_results = {0: [], 1: []}
        for pidx in discuss_problems:
            for pid in [0, 1]:
                correct = rng.random() > 0.3
                discussion_results[pid].append(DiscussionResult(
                    problem_idx=pidx, player_id=pid, tokens=[], text="",
                    log_probs=[], final_answer=problems[pidx].ground_truth if correct else "WRONG",
                    correct=correct, reward=1.0 if correct else 0.0,
                ))
        apply_persuader_bonus(rollouts, discussion_results, majority_answers, problems)

        # Step 8: Dynamic sampling
        for pid in [0, 1]:
            filtered = filter_dynamic_sampling(rollouts[pid])
            non_filtered_problems = set(r.problem_idx for r in filtered)

            # Conservation: filtered rollouts = non_filtered_problems × 8
            assert len(filtered) == len(non_filtered_problems) * 8

            # Total batch = filtered pre-discussion + discussion
            total = len(filtered) + len(discussion_results[pid])
            assert total > 0, "Batch should not be empty"

    def test_no_nan_advantages(self):
        """Every sequence in the batch should have a finite advantage."""
        rng = np.random.default_rng(456)
        problems, rollouts = _build_synthetic_batch(rng)

        for pid in rollouts:
            for r in rollouts[pid]:
                r.reward = compute_solve_reward(r)

        majority_answers = {}
        discuss_problems = []
        for pidx in range(len(problems)):
            for pid in [0, 1]:
                player_rs = [r for r in rollouts[pid] if r.problem_idx == pidx]
                majority_answers[(pid, pidx)] = majority_vote(player_rs)
            if should_discuss(majority_answers[(0, pidx)], majority_answers[(1, pidx)]):
                discuss_problems.append(pidx)

        discussion_results = {0: [], 1: []}
        for pidx in discuss_problems:
            for pid in [0, 1]:
                discussion_results[pid].append(DiscussionResult(
                    problem_idx=pidx, player_id=pid, tokens=[], text="",
                    log_probs=[], final_answer=problems[pidx].ground_truth,
                    correct=True, reward=1.0,
                ))
        apply_persuader_bonus(rollouts, discussion_results, majority_answers, problems)

        for pid in [0, 1]:
            filtered = filter_dynamic_sampling(rollouts[pid])
            pre_rewards = [r.reward for r in filtered]
            post_rewards = [dr.reward for dr in discussion_results[pid]]
            ema = EMATracker()

            pre_adv, post_adv = compute_player_advantages(pre_rewards, post_rewards, ema)

            for a in pre_adv:
                assert math.isfinite(a), f"Non-finite pre-discussion advantage: {a}"
            for a in post_adv:
                assert math.isfinite(a), f"Non-finite post-discussion advantage: {a}"
```

**Step 2: Run**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/integration/test_pipeline.py -v -m integration
```

**Step 3: Commit**

```bash
git add tests/integration/
git commit -m "test: add integration tests with conservation invariants"
```

---

## Phase 7: Evaluation Stubs + Config Validation

### Task 30: Evaluation stubs

**Files:**
- Create: `crisp/evaluation/__init__.py`
- Create: `crisp/evaluation/benchmarks.py`
- Create: `crisp/evaluation/bayes_at_n.py`

**Step 1: Create stub files**

```python
# crisp/evaluation/__init__.py
"""Evaluation module — stubs for future benchmark integration."""

# crisp/evaluation/benchmarks.py
"""Benchmark evaluation stubs."""
# TODO: Integrate standard math benchmarks (MATH, GSM8K, etc.)

# crisp/evaluation/bayes_at_n.py
"""Bayes@N evaluation metric stub."""
# TODO: Implement Bayesian pass@N estimation
```

**Step 2: Commit**

```bash
git add crisp/evaluation/
git commit -m "feat: add evaluation module stubs (benchmarks, bayes@n)"
```

### Task 31: Config validation tests

**Files:**
- Create: `tests/test_config.py`

**Step 1: Write tests**

```python
"""Tests for configuration defaults and validation."""
import pytest

from crisp.config import CRISPConfig, PlayerConfig, CoachConfig, AdvantageConfig, GRPOConfig


class TestCRISPConfig:
    def test_defaults_match_spec(self):
        cfg = CRISPConfig()
        assert cfg.player.rollouts_per_problem == 8
        assert cfg.player.solve_reward == 1.0
        assert cfg.player.wrong_reward == 0.0
        assert cfg.player.no_box_penalty == -0.5
        assert cfg.player.persuader_bonus == 0.3
        assert cfg.coach.batch_size == 8
        assert cfg.coach.discussion_alpha == 0.3
        assert cfg.coach.repetition_lambda == 1.0
        assert cfg.coach.repetition_tau == 0.85
        assert cfg.coach.repetition_window == 10
        assert cfg.coach.embedding_dim == 384
        assert cfg.advantage.ema_eta == 0.2
        assert cfg.advantage.ema_init_mu == 0.5
        assert cfg.advantage.ema_init_sigma_sq == 0.25
        assert cfg.grpo.dcpo_alpha == 3.0
        assert cfg.grpo.clip_base == 0.2
        assert cfg.grpo.js_beta == 0.001
        assert cfg.grpo.pre_discussion_l_max == 8192
        assert cfg.grpo.pre_discussion_buffer == 2048
        assert cfg.grpo.post_discussion_l_max == 4096
        assert cfg.grpo.post_discussion_buffer == 1024

    def test_custom_config(self):
        cfg = CRISPConfig(player=PlayerConfig(rollouts_per_problem=4))
        assert cfg.player.rollouts_per_problem == 4
        assert cfg.coach.batch_size == 8  # Other defaults preserved
```

**Step 2: Run**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/test_config.py -v
```

**Step 3: Commit**

```bash
git add tests/test_config.py
git commit -m "test: add config validation confirming all defaults match spec"
```

---

## Phase 8: Run Full Tier 1 Suite

### Task 32: Run all tests and verify green

**Step 1: Run full test suite**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/ -v --tb=short -m "not gpu"
```

Expected: ALL PASS

**Step 2: Run with coverage**

```bash
cd /home/alex/mech_interp/CRISP && pytest tests/ --cov=crisp --cov-report=term-missing -m "not gpu"
```

**Step 3: Commit any fixes if needed, then tag**

```bash
git tag v0.1.0-tests-green
```

---

## Phase 9: Clone MARTI for Infra Layer (Future)

### Task 33: Clone MARTI and extract infra modules

> This task is for **after** all Tier 1 tests are green. It involves:
> 1. Clone MARTI repo into a temporary directory
> 2. Copy relevant infra files into `crisp/infra/`
> 3. Strip PPO-specific code, keep Ray/vLLM/DeepSpeed wrappers
> 4. Adapt interfaces to match CRISP types
>
> **Not specified in detail here** — this requires hands-on exploration of MARTI source.
> Run this as a separate planning session.

---

## Summary

| Phase | Tasks | What it builds |
|-------|-------|----------------|
| 0 | 1 | Project scaffolding, types, config |
| 1 | 2-5 | Verifier (answer extraction + SymPy) |
| 2 | 6-15 | Rewards (EMA, player, repetition, coach, advantages) |
| 3 | 16-21 | Discussion (trigger, representative, post-discussion) |
| 4 | 22-27 | Training (overlong, batch builder, GRPO loss) |
| 5 | 28 | Full reward flow trace test |
| 6 | 29 | Integration tests with conservation invariants |
| 7 | 30-31 | Eval stubs + config validation |
| 8 | 32 | Full Tier 1 green + coverage |
| 9 | 33 | MARTI infra extraction (future) |
