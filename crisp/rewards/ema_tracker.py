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
    _has_been_updated: bool = field(default=False, repr=False)

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
        if not self._has_been_updated:
            # First update: initialize directly from batch stats
            self.mu = batch_mean
            self.sigma_sq = batch_var
            self._has_been_updated = True
        else:
            self.mu = (1 - self.eta) * self.mu + self.eta * batch_mean
            self.sigma_sq = (1 - self.eta) * self.sigma_sq + self.eta * batch_var

        # Prevent sigma collapse: when all rewards are identical across batches,
        # sigma_sq decays exponentially toward 0, causing advantage explosion
        # (advantage = (r - mu) / (sqrt(sigma_sq) + eps) → ±1e8).
        MIN_SIGMA_SQ = 0.01
        if self.sigma_sq < MIN_SIGMA_SQ:
            logger.warning(
                "EMA sigma_sq collapsed to %.4e (min=%.2f), clamping",
                self.sigma_sq, MIN_SIGMA_SQ,
            )
            self.sigma_sq = MIN_SIGMA_SQ
