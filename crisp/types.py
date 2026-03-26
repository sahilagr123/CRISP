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
    self_solvable: bool = True  # False if coach failed to solve its own problem


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
    prompt_len: int = 0  # number of prompt tokens (for truncation detection)
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
