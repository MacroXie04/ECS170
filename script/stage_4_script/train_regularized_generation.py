#!/usr/bin/env python3
"""Train two fixed tied-GRU experiments using training/validation only.

This script never generates examples and never evaluates the test partition.
The existing loader necessarily constructs all partitions; test records are
removed immediately. The original vocabulary and splits stay unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Experiment_Runner import collate, run_epoch, write_json, save_checkpoint
from local_code.stage_4_code.Regularized_Generation import (
    TiedGenerationGRU, config_digest, load_regularized_checkpoint,
)

EXPERIMENTS = {
    'tied128_dropout35': {'dimension': 128, 'dropout': .35, 'label_smoothing': 0.},
    'tied256_dropout35_smooth05': {'dimension': 256, 'dropout': .35, 'label_smoothing': .05},
}
SOURCES = [
    'local_code/stage_4_code/Dataset_Loader.py',
    'local_code/stage_4_code/Experiment_Runner.py',
    'local_code/stage_4_code/Method_RNN.py',
    'local_code/stage_4_code/Regularized_Generation.py',
    'script/stage_4_script/train_regularized_generation.py',
]


def train_epoch(model, loader, optimizer, device, smoothing):
    model.train()
    summed_nll, summed_objective, correct, count = 0., 0., 0, 0
    for tokens, lengths, _, _ in loader:
        tokens = tokens.to(device)
        targets = tokens[:, 1:]
        logits = model(tokens[:, :-1])
        mask = targets.ne(0)
        n = int(mask.sum())
        flat_logits, flat_targets = logits.flatten(0, 1), targets.flatten()
        objective = nn.functional.cross_entropy(flat_logits, flat_targets, ignore_index=0,
            reduction='sum', label_smoothing=smoothing)
        nll = nn.functional.cross_entropy(flat_logits.detach(), flat_targets, ignore_index=0,
            reduction='sum')
        loss = objective / n
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if torch.count_nonzero(model.embedding.weight[0]):
            raise RuntimeError('Tied PAD embedding drifted from zero')
        summed_nll += float(nll)
        summed_objective += float(objective.detach())
        correct += int((logits.argmax(-1).eq(targets) & mask).sum())
        count += n
    return {'loss': summed_nll / count, 'objective': summed_objective / count,
            'accuracy': correct / count, 'count': count}


def experiment(name, settings, data, args, device):
    output = Path(args.output) / name
    checkpoint_dir = output / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vocab = data['vocab']
    kwargs = dict(vocab_size=len(vocab), dimension=settings['dimension'], dropout=settings['dropout'])
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    config = dict(architecture='tied_gru_v1', model=kwargs, seed=1704, max_epochs=args.max_epochs,
        patience=12, batch_size=64, learning_rate=.001, optimizer='AdamW', weight_decay=.01,
        label_smoothing=settings['label_smoothing'], gradient_clip_norm=1.,
        selection='Lowest plain validation NLL; earliest tie. No generated examples/test evaluation.',
        initialization='Embedding normal std 0.1, zero PAD; default GRU; output tied, bias zero.',
        pad_policy='Zero tied embedding PAD gradient from input and output; AdamW keeps zero row.',
        data_metadata=data['metadata'], source_sha256=source_hashes)
    config_hash = config_digest(config)
    if (output / 'config.json').exists() and json.loads((output / 'config.json').read_text()) != config:
        raise ValueError('Existing run settings/source differ; use a new output directory')
    write_json(output / 'config.json', config)
    if (output / 'metrics.json').exists():
        load_regularized_checkpoint(checkpoint_dir / 'best.pt', device)
        print('SKIP completed ' + name, flush=True)
        return
    write_json(output / 'environment.json', dict(python=sys.version, torch=str(torch.__version__),
        numpy=np.__version__, platform=platform.platform(), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), command=sys.argv,
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256=source_hashes, parameter_dtype='float32', cudnn_deterministic=True,
        note='No external data or pretrained weights. Test partition not evaluated.'))
    random.seed(1704); np.random.seed(1704); torch.manual_seed(1704)
    if device.type == 'cuda': torch.cuda.manual_seed_all(1704)
    model = TiedGenerationGRU(**kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    generator = torch.Generator().manual_seed(1704)
    loaders = {split: DataLoader(data[split], batch_size=64, shuffle=split == 'train',
        generator=generator if split == 'train' else None, collate_fn=collate, num_workers=0)
        for split in ('train', 'validation')}
    history, best_loss, best_epoch, seconds = [], float('inf'), 0, 0.
    last = checkpoint_dir / 'last.pt'
    if last.exists():
        saved = torch.load(last, map_location=device, weights_only=True)
        if saved['config_hash'] != config_hash: raise ValueError('Resume configuration mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        history, best_loss, best_epoch, seconds = saved['history'], saved['best_loss'], saved['best_epoch'], saved['seconds']
        torch.set_rng_state(saved['rng'].cpu()); generator.set_state(saved['loader_rng'].cpu())
        if device.type == 'cuda': torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    print('START ' + name + ' parameters=' + str(model.parameter_count()), flush=True)
    for epoch in range(len(history) + 1, args.max_epochs + 1):
        if len(history) - best_epoch >= 12: break
        started = time.monotonic()
        train = train_epoch(model, loaders['train'], optimizer, device, settings['label_smoothing'])
        validation = run_epoch(model, loaders['validation'], 'generation', device)
        elapsed = time.monotonic() - started; seconds += elapsed
        row = dict(epoch=epoch, train_loss=train['loss'], train_objective=train['objective'],
            train_accuracy=train['accuracy'], validation_loss=validation['loss'],
            validation_accuracy=validation['accuracy'], seconds=elapsed)
        history.append(row)
        common = dict(model=model.state_dict(), model_kwargs=kwargs, vocab=vocab.to_dict(),
            epoch=epoch, config_hash=config_hash)
        if validation['loss'] < best_loss:
            best_loss, best_epoch = validation['loss'], epoch
            save_checkpoint(checkpoint_dir / 'best.pt', common)
        write_json(output / 'history.json', history)
        save_checkpoint(last, dict(**common, optimizer=optimizer.state_dict(), history=history,
            best_loss=best_loss, best_epoch=best_epoch, seconds=seconds, rng=torch.get_rng_state(),
            loader_rng=generator.get_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else []))
        print(json.dumps({'run': name, **row, 'best_epoch': best_epoch}), flush=True)
    selected, loaded_vocab, provenance = load_regularized_checkpoint(checkpoint_dir / 'best.pt', device)
    validation = run_epoch(selected, loaders['validation'], 'generation', device)
    if abs(validation['loss'] - best_loss) > 1e-6:
        raise RuntimeError('Selected checkpoint validation replay mismatch')
    result = dict(name=name, parameters=model.parameter_count(), best_epoch=best_epoch,
        epochs_completed=len(history), validation=validation, last_validation=history[-1]['validation_loss'],
        train_seconds=seconds, checkpoint_sha256=provenance['checkpoint_sha256'],
        config_hash=config_hash, test_evaluated=False)
    write_json(output / 'metrics.json', result)
    print('COMPLETE ' + json.dumps(result), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/stage_4_data')
    parser.add_argument('--output', default='result/stage_4_result/generation_generalization/regularized_runs')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    args = parser.parse_args(argv)
    if args.max_epochs < 1: parser.error('--max-epochs must be positive')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    data = Dataset_Loader(args.data_dir, seed=1704).load_generation()
    del data['test']
    for name in args.experiments:
        experiment(name, EXPERIMENTS[name], data, args, device)


if __name__ == '__main__': main()
