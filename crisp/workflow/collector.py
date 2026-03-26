"""Observability hook for step() — captures intermediate data per iteration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from crisp.types import DiscussionResult, Problem, Rollout
from crisp.workflow.context import StepResult


@dataclass
class IterationData:
    """All intermediate data from one training iteration."""
    iteration: int
    problems: List[Problem]
    rollouts: Dict[int, List[Rollout]]
    majority_answers: Dict[Tuple[int, int], str]
    discussion_results: Dict[int, List[DiscussionResult]]
    player_loss: float
    coach_loss: Optional[float]
    coach_rewards: Optional[List[float]]
    result: StepResult


class StepCollector:
    """Accumulates IterationData across the training loop."""

    def __init__(self) -> None:
        self.iterations: List[IterationData] = []

    def record(self, data: IterationData) -> None:
        self.iterations.append(data)
