"""Tests for sliding window repetition buffer."""
import numpy as np
import pytest

from crisp.rewards.repetition_buffer import RepetitionBuffer


class TestRepetitionBuffer:
    def test_empty_buffer_returns_zero(self):
        buf = RepetitionBuffer(max_batches=10, embedding_dim=384)
        emb = np.random.default_rng(0).random(384).astype(np.float32)
        assert buf.compute_penalty(emb, lambda_rep=1.0, tau_sim=0.85) == 0.0

    def test_partial_buffer_still_penalizes(self):
        """Buffer penalizes even before reaching max_batches (no warm-up gate)."""
        buf = RepetitionBuffer(max_batches=3, embedding_dim=4)
        emb = np.ones(4, dtype=np.float32)
        buf.push([emb.copy() for _ in range(4)])  # 1 of 3 batches
        penalty = buf.compute_penalty(emb, lambda_rep=1.0, tau_sim=0.85)
        # All 4 historical are identical → penalty = 1.0 * 4/4 = 1.0
        assert penalty == pytest.approx(1.0)

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
