"""Tensor preparation utilities for training."""
from __future__ import annotations

from typing import List, Tuple

import torch

from crisp.types import TokenSequence


def pad_sequences(
    sequences: List[TokenSequence],
    pad_token_id: int = 0,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
    """Pad variable-length TokenSequences into batched tensors.

    Returns:
        input_ids: [B, T] padded token IDs
        attention_mask: [B, T] binary mask (1 = real token, 0 = pad)
        old_log_probs: [B, T] padded log-probabilities
    """
    if not sequences:
        return (
            torch.zeros(0, 0, dtype=torch.long),
            torch.zeros(0, 0, dtype=torch.long),
            torch.zeros(0, 0),
        )

    max_len = max(len(seq.tokens) for seq in sequences)
    batch_size = len(sequences)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    old_log_probs = torch.zeros(batch_size, max_len)

    for i, seq in enumerate(sequences):
        length = len(seq.tokens)
        input_ids[i, :length] = torch.tensor(seq.tokens, dtype=torch.long)
        attention_mask[i, :length] = 1
        old_log_probs[i, :length] = torch.tensor(seq.log_probs)

    return input_ids, attention_mask, old_log_probs
