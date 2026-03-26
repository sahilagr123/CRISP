"""Tests for Actor model wrapper — no GPU required (tests use mocks)."""
from unittest.mock import patch, MagicMock
import torch


def test_actor_from_existing_model():
    """Actor can wrap an existing nn.Module."""
    from crisp.infra.actor_model import Actor
    mock_model = MagicMock()
    actor = Actor(mock_model)
    assert actor.model is mock_model


def test_actor_gradient_checkpointing():
    """Actor delegates gradient checkpointing to inner model."""
    from crisp.infra.actor_model import Actor
    mock_model = MagicMock()
    actor = Actor(mock_model)
    actor.gradient_checkpointing_enable()
    mock_model.gradient_checkpointing_enable.assert_called_once()
    actor.gradient_checkpointing_disable()
    mock_model.gradient_checkpointing_disable.assert_called_once()


def test_actor_has_use_cache_false():
    """Actor loaded from pretrained disables use_cache."""
    from crisp.infra.actor_model import Actor
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained:
        mock_model = MagicMock()
        mock_model.config.to_dict.return_value = {}
        mock_from_pretrained.return_value = mock_model
        actor = Actor("fake-model-path", bf16=True)
        assert actor.model.config.use_cache is False


def test_log_probs_from_logits():
    """log_probs_from_logits computes correct values."""
    from crisp.infra.actor_model import log_probs_from_logits
    logits = torch.tensor([[[2.0, 1.0], [1.0, 2.0], [0.5, 0.5]]])
    labels = torch.tensor([[1, 0, 1]])
    log_probs = log_probs_from_logits(logits, labels)
    assert log_probs.shape == (1, 3)
    expected_0 = torch.log_softmax(torch.tensor([2.0, 1.0]), dim=0)[1]
    assert abs(log_probs[0, 0].item() - expected_0.item()) < 1e-5


def test_chunked_lm_head_matches_direct():
    """chunked_lm_head_log_probs produces identical results to direct computation."""
    from crisp.infra.actor_model import (
        chunked_lm_head_log_probs,
        log_probs_from_logits,
        LM_HEAD_CHUNK_SIZE,
    )

    B, L, H, V = 2, 50, 16, 32
    torch.manual_seed(42)
    hidden = torch.randn(B, L, H)
    labels = torch.randint(0, V, (B, L))
    lm_head = torch.nn.Linear(H, V, bias=False)

    # Direct (unchunked) computation
    logits = lm_head(hidden).to(torch.float32)
    expected = log_probs_from_logits(logits, labels)

    # Chunked computation (use small chunk to force multiple chunks)
    import crisp.infra.actor_model as _mod
    old_chunk = _mod.LM_HEAD_CHUNK_SIZE
    _mod.LM_HEAD_CHUNK_SIZE = 8  # force chunking with L=50
    try:
        result = chunked_lm_head_log_probs(hidden, lm_head, labels)
    finally:
        _mod.LM_HEAD_CHUNK_SIZE = old_chunk

    assert result.shape == expected.shape
    assert torch.allclose(result, expected, atol=1e-5), \
        f"max diff: {(result - expected).abs().max().item()}"


def test_chunked_lm_head_grad_flows():
    """Gradients flow correctly through chunked lm_head computation."""
    from crisp.infra.actor_model import chunked_lm_head_log_probs
    import crisp.infra.actor_model as _mod

    B, L, H, V = 1, 20, 8, 16
    torch.manual_seed(0)
    hidden = torch.randn(B, L, H, requires_grad=True)
    labels = torch.randint(0, V, (B, L))
    lm_head = torch.nn.Linear(H, V, bias=False)

    old_chunk = _mod.LM_HEAD_CHUNK_SIZE
    _mod.LM_HEAD_CHUNK_SIZE = 4
    try:
        log_probs = chunked_lm_head_log_probs(hidden, lm_head, labels)
        loss = log_probs.sum()
        loss.backward()
    finally:
        _mod.LM_HEAD_CHUNK_SIZE = old_chunk

    assert hidden.grad is not None
    assert hidden.grad.shape == hidden.shape
    assert not torch.all(hidden.grad == 0)
