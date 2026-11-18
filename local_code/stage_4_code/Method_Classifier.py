"""Classification-only recurrent models for Stage 4 IMDb experiments.

This module leaves the existing generation model and baseline checkpoints intact.
Right-padded reviews are packed before the recurrent layer so reverse states
never see padding. Pooling considers only real token positions.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ClassifierModel(nn.Module):
    """Learned embeddings followed by an RNN, GRU, or LSTM sentiment classifier."""

    def __init__(
        self,
        vocab_size: int,
        cell: str = "gru",
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
        pooling: str = "mean",
        pad_id: int = 0,
    ):
        super().__init__()
        cell = cell.lower()
        if cell not in {"rnn", "gru", "lstm"}:
            raise ValueError(f"Unknown recurrent cell: {cell}")
        if pooling not in {"mean", "max", "meanmax"}:
            raise ValueError(f"Unknown pooling mode: {pooling}")
        if min(vocab_size, embedding_dim, hidden_dim, num_layers) < 1:
            raise ValueError("Vocabulary and network dimensions must be positive")
        if not 0 <= pad_id < vocab_size:
            raise ValueError("pad_id must be within the vocabulary")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.cell = cell
        self.pooling = pooling
        self.bidirectional = bool(bidirectional)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        recurrent_type = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.recurrent = recurrent_type(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        state_dim = hidden_dim * (2 if self.bidirectional else 1)
        pooled_dim = state_dim * (2 if pooling == "meanmax" else 1)
        self.output = nn.Linear(pooled_dim, 2)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.size(1) < 1:
            raise ValueError("tokens must have shape [batch, positive sequence length]")
        lengths = torch.as_tensor(lengths, device=tokens.device)
        if lengths.ndim != 1 or lengths.numel() != tokens.size(0):
            raise ValueError("lengths must contain one length per review")
        if lengths.is_floating_point() or lengths.dtype == torch.bool:
            raise ValueError("lengths must be integers")
        if torch.any(lengths < 1) or torch.any(lengths > tokens.size(1)):
            raise ValueError("Review lengths must be between 1 and the padded width")

        embedded = self.dropout(self.embedding(tokens))
        packed = pack_padded_sequence(
            embedded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_states, _ = self.recurrent(packed)
        states, _ = pad_packed_sequence(
            packed_states, batch_first=True, total_length=tokens.size(1)
        )
        mask = (
            torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        if self.pooling in {"mean", "meanmax"}:
            mean_states = (states * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1)
        if self.pooling in {"max", "meanmax"}:
            max_states = states.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(dim=1).values
        if self.pooling == "mean":
            pooled = mean_states
        elif self.pooling == "max":
            pooled = max_states
        else:
            pooled = torch.cat((mean_states, max_states), dim=-1)
        return self.output(self.dropout(pooled))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
