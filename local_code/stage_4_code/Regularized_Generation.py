"""Tied-embedding GRU experiments, isolated from frozen Stage 4 baselines.

The tied PAD row receives gradients from the output projection too. Its hook
explicitly zeros that row; zero initialization plus AdamW keeps it exactly zero.
Only the output bias for PAD is learned, as for other output classes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from local_code.stage_4_code.Dataset_Loader import Vocabulary


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def vocabulary_digest(vocab):
    return hashlib.sha256(b''.join(token.encode() + b'\0' for token in vocab.tokens)).hexdigest()


class TiedGenerationGRU(nn.Module):
    """A causal GRU whose input embeddings also parameterize output words."""

    def __init__(self, vocab_size, dimension=128, dropout=.35, pad_id=0):
        super().__init__()
        if vocab_size < 4 or dimension < 1 or pad_id != 0:
            raise ValueError('Expected positive dimensions and the frozen PAD index 0')
        if not 0 <= dropout < 1:
            raise ValueError('Dropout must lie in [0, 1)')
        self.task, self.cell = 'generation', 'tied_gru'
        self.vocab_size, self.pad_id = vocab_size, pad_id
        self.embedding = nn.Embedding(vocab_size, dimension, padding_idx=pad_id)
        self.recurrent = nn.GRU(dimension, dimension, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dimension, vocab_size)
        self.output.weight = self.embedding.weight
        nn.init.normal_(self.embedding.weight, mean=0., std=.1)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.embedding.weight[pad_id].zero_()
        self.embedding.weight.register_hook(self._zero_pad_gradient)

    def _zero_pad_gradient(self, gradient):
        gradient = gradient.clone()
        gradient[self.pad_id].zero_()
        return gradient

    def step(self, tokens, hidden=None):
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError('tokens must have shape [batch, positive sequence length]')
        states, hidden = self.recurrent(self.dropout(self.embedding(tokens)), hidden)
        return self.output(self.dropout(states)), hidden

    def forward(self, tokens, lengths=None):
        return self.step(tokens)[0]

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def load_regularized_checkpoint(checkpoint, device='cpu'):
    """Load a bound experimental checkpoint without changing the caller's RNG."""
    checkpoint = Path(checkpoint).resolve()
    run = checkpoint.parent.parent
    config = json.loads((run / 'config.json').read_text())
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    digest = config_digest(config)
    if saved.get('config_hash') != digest or config.get('architecture') != 'tied_gru_v1':
        raise ValueError('Mismatched configuration or unsupported architecture')
    repo = Path(__file__).resolve().parents[2]
    for name, expected in config.get('source_sha256', {}).items():
        source = (repo / name).resolve()
        if not source.is_relative_to(repo) or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('Training source changed: ' + name)
    vocab_state = saved['vocab']
    vocab = Vocabulary.from_dict(vocab_state)
    if vocab.to_dict() != vocab_state or len(vocab) != config['model']['vocab_size']:
        raise ValueError('Checkpoint vocabulary size or token ordering changed')
    if vocabulary_digest(vocab) != config['data_metadata']['vocabulary_sha256']:
        raise ValueError('Checkpoint vocabulary does not match configuration')
    state = saved['model']
    if not torch.equal(state['embedding.weight'], state['output.weight']):
        raise ValueError('Saved input/output weights are not tied')
    if torch.count_nonzero(state['embedding.weight'][0]):
        raise ValueError('Saved PAD embedding is not zero')
    with torch.random.fork_rng(devices=[]):
        model = TiedGenerationGRU(**config['model'])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, vocab, {'epoch': saved['epoch'], 'config': config,
                         'config_hash': digest,
                         'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
