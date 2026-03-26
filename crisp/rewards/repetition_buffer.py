"""Sliding window embedding buffer for coach repetition penalty."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RepetitionBuffer:
    """FIFO buffer of problem embeddings for cross-batch repetition detection.

    Stores embeddings from the last W batches. Penalty activates immediately
    using whatever history is available (no warm-up gate).
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

        Returns lambda * count(sim > tau) / |H| using all available history.
        Returns 0.0 only if the buffer is empty.
        """
        if len(self.buffer) == 0:
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
