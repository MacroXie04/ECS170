"""Load and assess generation-only follow-up experiments without altering baselines."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from local_code.stage_4_code.Dataset_Loader import Vocabulary, tokenize
from local_code.stage_4_code.Method_RNN import SequenceModel


def load_generation_checkpoint(checkpoint, device='cpu'):
    checkpoint = Path(checkpoint).resolve()
    run = checkpoint.parent.parent
    config = json.loads((run / 'config.json').read_text())
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if saved['config_hash'] != digest or config['model']['task'] != 'generation':
        raise ValueError('Mismatched configuration or non-generation checkpoint')
    repo = Path(__file__).resolve().parents[2]
    for name, expected in config.get('source_sha256', {}).items():
        source = (repo / name).resolve()
        if not source.is_relative_to(repo) or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('Training source changed: ' + name)
    vocab_state = saved.get('vocab')
    if vocab_state is None:
        vocab_state = json.loads((run.parent / 'generation_vocabulary.json').read_text())
    vocab = Vocabulary.from_dict(vocab_state)
    if len(vocab) != config['model']['vocab_size']:
        raise ValueError('Checkpoint vocabulary size does not match the model')
    expected_vocab_hash = config.get('data_metadata', {}).get('vocabulary_sha256')
    if expected_vocab_hash is not None:
        actual_vocab_hash = hashlib.sha256(b''.join(token.encode()+b'\0' for token in vocab.tokens)).hexdigest()
        if actual_vocab_hash != expected_vocab_hash:
            raise ValueError('Saved vocabulary has changed since training')
    with torch.random.fork_rng(devices=[]):
        model = SequenceModel(**config['model'])
    model.load_state_dict(saved['model'], strict=True)
    model.to(device).eval()
    epoch = saved['epoch'] if 'epoch' in saved else len(saved['history'])
    return model, vocab, {'epoch': epoch, 'config': config, 'config_hash': digest,
                          'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}


def validate_prefix(prompt, vocab):
    tokens = tokenize(prompt)
    words = sum(any(character.isalnum() for character in token) for token in tokens)
    if words != 3:
        raise ValueError(f'Expected exactly three words, found {words}')
    unknown = sorted({token for token in tokens if token not in vocab.stoi})
    if unknown:
        raise ValueError('Unknown prompt tokens: ' + ', '.join(unknown))
    return vocab.encode(prompt, bos=True)


def format_completion(prompt, ids, prefix_length, vocab):
    continuation = vocab.decode(ids[prefix_length:])
    separator = '' if not continuation or continuation[0] in '.,!?;:%)]}' else ' '
    return prompt.strip() + separator + continuation
