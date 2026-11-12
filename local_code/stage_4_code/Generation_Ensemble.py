"""Probability mixtures of independently recurrent generation checkpoints.

This combines next-token distributions, not parameter tensors. Each component
keeps its own recurrent state while consuming the same generated token prefix.
Mixture outputs are normalized log probabilities and can also be used as logits
by cross-entropy evaluation or the shared generation decoder.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class MixtureModel(nn.Module):
    task = 'generation'
    cell = 'early_late_probability_mixture'

    def __init__(self, models, late_weight=.5):
        super().__init__()
        models = list(models)
        if len(models) != 2:
            raise ValueError('Provide exactly two models, ordered [early, late]')
        if isinstance(late_weight, bool) or not math.isfinite(late_weight) or not 0 <= late_weight <= 1:
            raise ValueError('late_weight must be finite and in [0,1]')
        if any(getattr(model, 'task', None) != 'generation' for model in models):
            raise ValueError('Both components must be generation models')
        sizes = [int(model.vocab_size) for model in models]
        if len(set(sizes)) != 1 or sizes[0] < 5:
            raise ValueError('Both components must have the same valid vocabulary size')
        self.models = nn.ModuleList(models)
        self.vocab_size = sizes[0]
        self.late_weight = float(late_weight)
        weights = torch.tensor([1 - self.late_weight, self.late_weight], dtype=torch.float64)
        self.register_buffer('log_weights', weights.log())

    def _mixture(self, logits):
        if logits[0].shape != logits[1].shape or logits[0].shape[-1] != self.vocab_size:
            raise ValueError('Component logits must have matching shapes and vocabulary axes')
        if logits[0].device != logits[1].device:
            raise ValueError('Component models must be on the same device')
        log_probs = torch.stack([value.log_softmax(-1) for value in logits], dim=0)
        shape = (2,) + (1,) * (log_probs.ndim - 1)
        weights = self.log_weights.to(device=log_probs.device, dtype=log_probs.dtype).reshape(shape)
        return torch.logsumexp(log_probs + weights, dim=0)

    def forward(self, tokens, lengths=None):
        return self._mixture([model(tokens, lengths) for model in self.models])

    def step(self, tokens, hidden=None):
        if hidden is None:
            hidden = (None, None)
        elif not isinstance(hidden, (tuple, list)) or len(hidden) != len(self.models):
            raise ValueError('Mixture hidden state must contain one state per component')
        logits, states = [], []
        for model, state in zip(self.models, hidden):
            output, next_state = model.step(tokens, state)
            logits.append(output)
            states.append(next_state)
        return self._mixture(logits), tuple(states)

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
