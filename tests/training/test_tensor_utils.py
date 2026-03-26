"""Tests for tensor preparation utilities."""
import torch

from crisp.types import TokenSequence


def test_pad_sequences_basic():
    """pad_sequences pads to max length and builds attention mask."""
    from crisp.training.tensor_utils import pad_sequences

    seqs = [
        TokenSequence(tokens=[1, 2, 3], log_probs=[-0.1, -0.2, -0.3]),
        TokenSequence(tokens=[4, 5], log_probs=[-0.4, -0.5]),
    ]
    input_ids, attention_mask, old_log_probs = pad_sequences(seqs, pad_token_id=0)

    assert input_ids.shape == (2, 3)
    assert attention_mask.shape == (2, 3)
    assert old_log_probs.shape == (2, 3)

    # Second sequence padded on the right
    assert input_ids[1].tolist() == [4, 5, 0]
    assert attention_mask[1].tolist() == [1, 1, 0]
    assert old_log_probs[1].tolist()[-1] == 0.0


def test_pad_sequences_single():
    """pad_sequences handles a single sequence (no padding needed)."""
    from crisp.training.tensor_utils import pad_sequences

    seqs = [TokenSequence(tokens=[10, 20], log_probs=[-1.0, -2.0])]
    input_ids, attention_mask, old_log_probs = pad_sequences(seqs, pad_token_id=0)

    assert input_ids.shape == (1, 2)
    assert attention_mask[0].tolist() == [1, 1]


def test_pad_sequences_empty():
    """pad_sequences returns empty tensors for empty input."""
    from crisp.training.tensor_utils import pad_sequences

    input_ids, attention_mask, old_log_probs = pad_sequences([], pad_token_id=0)
    assert input_ids.shape[0] == 0
