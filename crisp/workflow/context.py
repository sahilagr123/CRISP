"""WorkflowContext and StepResult for CRISP workflow orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from crisp.config import CRISPConfig
from crisp.rewards.ema_tracker import EMATracker
from crisp.rewards.repetition_buffer import RepetitionBuffer


@dataclass
class WorkflowContext:
    """Holds all infra handles and stateful objects for the training loop.

    Passed to each step function for dependency injection.
    Two independent player strategies (Alice, Bob) with independent EMA trackers
    implement the Dr. MAS principle: each agent lives in its own reward universe.
    """
    player_vllm: List[Any]       # shared vLLM engines (same base model)
    coach_vllm: Optional[List[Any]]  # vLLM engine actors (None → HF generate)
    ref_model: Any               # Frozen reference policy (NEVER updated, shared)
    ds_alice: Any                # DeepSpeed strategy for Alice (player 0)
    ds_bob: Any                  # DeepSpeed strategy for Bob (player 1)
    ds_coach: Any                # DeepSpeed strategy for coach
    config: CRISPConfig
    # Independent per-player EMA trackers (Dr. MAS: separate reward universes)
    alice_ema: EMATracker
    bob_ema: EMATracker
    coach_ema: EMATracker
    rep_buffer: RepetitionBuffer
    iteration: int = 0
    pad_token_id: int = 0
    tokenizer: Any = None
    coach_tokenizer: Any = None  # separate tokenizer when coach model differs
    accuracy_history: List[float] = field(default_factory=list)


@dataclass
class StepResult:
    """Output metrics from a single training step."""
    alice_loss: float
    bob_loss: float
    coach_loss: Optional[float]  # None on non-coach-update iterations
    num_problems: int
    num_discussions: int
    player_accuracy: float       # fraction correct pre-discussion (both players)
    coach_iteration: bool        # whether coach was updated this step
