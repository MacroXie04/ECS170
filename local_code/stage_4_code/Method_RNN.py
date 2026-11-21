"""Shared recurrent networks for Stage 4 sentiment and word generation.

The instructor's ``method`` base supplies project metadata. Training lives in
the experiment runner: ``nn.Module.train()`` keeps its normal PyTorch meaning.
All networks are causal and unidirectional. Classification averages only real
token states, so right padding cannot change a review's representation.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from local_code.base_class.method import method


class SequenceModel(method, nn.Module):
    """One architecture with an interchangeable RNN, LSTM, or GRU cell."""

    def __init__(self, vocab_size, task="classification", cell="rnn",
                 embedding_dim=128, hidden_dim=128, num_layers=1,
                 dropout=0.3, pad_id=0):
        if task not in {"classification", "generation"}:
            raise ValueError(f"Unknown task: {task}")
        cell = cell.lower()
        if cell not in {"rnn", "lstm", "gru"}:
            raise ValueError(f"Unknown recurrent cell: {cell}")
        if min(vocab_size, embedding_dim, hidden_dim, num_layers) < 1:
            raise ValueError("Vocabulary and network dimensions must be positive")
        if not 0 <= pad_id < vocab_size or not 0 <= dropout < 1:
            raise ValueError("Invalid padding ID or dropout probability")
        nn.Module.__init__(self)
        method.__init__(self, f"{cell.upper()} {task}",
                        "Causal recurrent model with learned word embeddings")
        self.task, self.cell = task, cell
        self.vocab_size, self.pad_id = vocab_size, pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        recurrent_type = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.recurrent = recurrent_type(
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=False,
        )
        self.output = nn.Linear(hidden_dim, 2 if task == "classification" else vocab_size)

    def _sequence(self, tokens, hidden=None):
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must have shape [batch, positive sequence length]")
        embedded = self.dropout(self.embedding(tokens))
        return self.recurrent(embedded, hidden)

    def forward(self, tokens, lengths=None):
        states, _ = self._sequence(tokens)
        if self.task == "generation":
            return self.output(self.dropout(states))
        if lengths is None:
            lengths = tokens.ne(self.pad_id).sum(dim=1)
        lengths = torch.as_tensor(lengths, device=tokens.device)
        if lengths.ndim != 1 or lengths.shape[0] != tokens.shape[0]:
            raise ValueError("lengths must contain one length per review")
        if lengths.is_floating_point() or lengths.dtype == torch.bool:
            raise ValueError("lengths must be integers")
        if torch.any(lengths <= 0) or torch.any(lengths > tokens.shape[1]):
            raise ValueError("Review lengths must be between 1 and the padded width")
        mask = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < lengths[:, None]
        pooled = (states * mask[:, :, None]).sum(dim=1) / lengths[:, None]
        return self.output(self.dropout(pooled))

    def step(self, tokens, hidden=None):
        """Consume new tokens while retaining recurrent state for generation."""
        if self.task != "generation":
            raise ValueError("step is only defined for a generation model")
        states, hidden = self._sequence(tokens, hidden)
        return self.output(self.dropout(states)), hidden

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def run(self, trainData, trainLabel, testData):
        raise NotImplementedError(
            "Fit and evaluate with the Stage 4 experiment runner; "
            "forward() and step() implement network inference."
        )


@torch.no_grad()
def generate(model, prefix_ids, eos_id=3, max_new_tokens=40,
             temperature=0, top_k=20, seed=1704):
    """Return prefix plus a greedy or reproducible sampled continuation.

    PAD (0), UNK (1), and BOS (2) cannot be generated. The caller must provide
    at least one prefix token (normally BOS plus three words). Sampling uses a
    private CPU generator, leaving the experiment's global RNG state intact.
    """
    if model.task != "generation":
        raise ValueError("A generation model is required")
    prefix = [int(token) for token in prefix_ids]
    if not prefix:
        raise ValueError("The generation prefix cannot be empty")
    if any(token < 0 or token >= model.vocab_size for token in prefix):
        raise ValueError("Prefix token outside the model vocabulary")
    if not 3 <= eos_id < model.vocab_size:
        raise ValueError("EOS must be a valid non-blocked vocabulary ID")
    if max_new_tokens < 0 or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if prefix[-1] == eos_id or max_new_tokens == 0:
        return prefix

    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        tokens = torch.tensor([prefix], device=device, dtype=torch.long)
        logits, hidden = model.step(tokens)
        output = list(prefix)
        for _ in range(max_new_tokens):
            scores = logits[0, -1].float().clone()
            scores[:3] = -torch.inf
            if temperature == 0:
                chosen = int(scores.argmax().item())
            else:
                # Restrict k to eligible tokens, excluding the blocked specials.
                values, indices = torch.topk(scores, min(top_k, model.vocab_size - 3))
                probabilities = torch.softmax(values.cpu() / temperature, dim=-1)
                offset = int(torch.multinomial(probabilities, 1, generator=generator).item())
                chosen = int(indices[offset].item())
            output.append(chosen)
            if chosen == eos_id:
                break
            token = torch.tensor([[chosen]], device=device, dtype=torch.long)
            logits, hidden = model.step(token, hidden)
        return output
    finally:
        model.train(was_training)
