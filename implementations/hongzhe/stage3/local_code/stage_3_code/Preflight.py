"""Checks that must pass before formal training.

Trains only on small slices of the training split. Test data is inspected for
shape, labels, and overlap with training images; no model ever sees it here.
"""
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile

import numpy as np
import torch

from local_code.stage_3_code.Dataset_Loader import (DATASET_SPECS, Dataset_Loader,
                                                    compute_channel_stats, make_eval_loader,
                                                    make_train_validation_split)
from local_code.stage_3_code.Experiment_Config import (DATASET_ORDER, REGISTRY, SPLIT_SEED)
from local_code.stage_3_code.Method_CNN import load_checkpoint
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Runtime import (capture_environment, configure_threads,
                                             resolve_device, seed_everything, write_json,
                                             write_text)
from local_code.stage_3_code.Setting_Stage3 import attach_training_data

METHOD_CLASSES = {'mnist': Method_CNN_MNIST, 'orl': Method_CNN_ORL, 'cifar': Method_CNN_CIFAR}
REPO = Path(__file__).resolve().parents[2]
SEED = 170
OVERFIT_SAMPLES = {'mnist': 64, 'cifar': 64, 'orl': 80}
OVERFIT_EPOCHS = 60


def _record(checks, name, passed, detail=''):
    checks.append({'check': name, 'passed': bool(passed), 'detail': detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


def check_data(dataset_key, data_dir, checks):
    spec = DATASET_SPECS[dataset_key]
    loader = Dataset_Loader(dataset_key, spec.file_name, folder=data_dir)
    data = loader.load()
    train, test = data['train'], data['test']
    _record(checks, f'{dataset_key}: instructor checksum', True, spec.sha256[:16])
    _record(checks, f'{dataset_key}: sample counts',
            len(train['y']) == spec.train_size and len(test['y']) == spec.test_size,
            f"train={len(train['y'])} test={len(test['y'])}")
    _record(checks, f'{dataset_key}: tensor shape',
            train['X'].shape[1:] == spec.input_shape == test['X'].shape[1:],
            f"{train['X'].shape} {train['X'].dtype}")
    _record(checks, f'{dataset_key}: zero-based labels',
            all(split['y'].min() == 0 and split['y'].max() == spec.num_classes - 1
                for split in (train, test)),
            f"train {train['y'].min()}..{train['y'].max()} "
            f"test {test['y'].min()}..{test['y'].max()}")
    fingerprints = [{hashlib.md5(image.tobytes()).digest() for image in split['X']}
                    for split in (train, test)]
    overlap = len(fingerprints[0] & fingerprints[1])
    _record(checks, f'{dataset_key}: no image in both train and test', overlap == 0,
            f'{overlap} shared')
    train_idx, val_idx = make_train_validation_split(train['y'], spec.validation_size,
                                                     SPLIT_SEED)
    disjoint = not set(train_idx.tolist()) & set(val_idx.tolist())
    complete = len(train_idx) + len(val_idx) == spec.train_size
    counts = np.bincount(train['y'][train_idx], minlength=spec.num_classes)
    _record(checks, f'{dataset_key}: validation split', disjoint and complete,
            f'optimization={len(train_idx)} validation={len(val_idx)}')
    _record(checks, f'{dataset_key}: every class keeps training images', counts.min() > 0,
            f'smallest class has {counts.min()} images')
    return train, train_idx, val_idx


def check_all_configs(dataset_key, device, checks):
    """Forward shape, finite loss, and gradients for every registered config."""
    spec = DATASET_SPECS[dataset_key]
    for name, config in REGISTRY[dataset_key].items():
        seed_everything(SEED)
        method = METHOD_CLASSES[dataset_key](config, device=device)
        method.train()
        images = torch.randn(4, *spec.input_shape, device=device)
        labels = torch.arange(4, device=device) % spec.num_classes
        logits = method(images)
        loss = method.objective(logits, labels)
        loss.backward()
        missing = [key for key, value in method.named_parameters()
                   if value.grad is None or not torch.isfinite(value.grad).all()]
        method.eval()
        passed = (tuple(logits.shape) == (4, spec.num_classes) and torch.isfinite(loss).item()
                  and not missing and not method.training)
        _record(checks, f'{dataset_key}/{name}: forward, loss, backward', passed,
                f'logits={tuple(logits.shape)} loss={loss.item():.4f} '
                f'params={method.parameter_count():,}')


def _ignore_epoch_end(_improved):
    pass


def _slice_training_data(train, indices):
    return {'X': train['X'][indices], 'y': train['y'][indices]}


def check_overfit(dataset_key, train, train_idx, device, checks):
    """A correct pipeline can memorize a small sample."""
    config = REGISTRY[dataset_key]['baseline']
    samples = OVERFIT_SAMPLES[dataset_key]
    chosen = np.random.default_rng(SEED).permutation(train_idx)
    if dataset_key == 'orl':  # two images each from all 40 people
        chosen = np.concatenate([chosen[train['y'][chosen] == person][:2] for person in range(40)])
    subset = _slice_training_data(train, chosen[:samples])
    mean, std = compute_channel_stats(subset['X'])
    seed_everything(SEED)
    config = replace(config, max_epoch=OVERFIT_EPOCHS, batch_size=16, aug_shift=0,
                     aug_hflip=False, dropout=0.0, learning_rate=1e-3, optimizer='adam',
                     momentum=0.0, nesterov=False, weight_decay=0.0)
    method = METHOD_CLASSES[dataset_key](config, device=device)
    method.log = lambda _message: None
    attach_training_data(method, subset, subset, {'mean': mean, 'std': std}, SEED,
                         _ignore_epoch_end)
    eval_loader = make_eval_loader(subset['X'], subset['y'], mean, std, 256)
    before = method.predict(eval_loader)['cross_entropy']
    method.fit()
    after = method.predict(eval_loader)
    passed = after['cross_entropy'] < 0.2 * before and after['accuracy'] >= 0.95
    _record(checks, f'{dataset_key}: loss falls on {len(subset["y"])} samples', passed,
            f"cross-entropy {before:.3f} -> {after['cross_entropy']:.4f}, "
            f"accuracy {after['accuracy']:.3f}")


def check_checkpoint_cycle(dataset_key, train, train_idx, val_idx, device, checks):
    """Save, reload, and resume, comparing against an uninterrupted run."""
    baseline = REGISTRY[dataset_key]['baseline']
    config = replace(baseline, max_epoch=4, batch_size=32)
    subset = _slice_training_data(train, train_idx[:256])
    validation = _slice_training_data(train, val_idx[:40])
    mean, std = compute_channel_stats(subset['X'])
    normalization = {'mean': mean, 'std': std}

    def fresh_method(saver=None):
        seed_everything(SEED)
        method = METHOD_CLASSES[dataset_key](config, device=device)
        method.log = lambda _message: None
        attach_training_data(method, subset, validation, normalization, SEED,
                             saver or _ignore_epoch_end)
        return method

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'state.pt'
        straight = fresh_method()
        straight.fit()

        interrupted = fresh_method()

        def save_at_two(_improved):
            if interrupted.history[-1]['epoch'] == 2:
                interrupted.save_checkpoint(path)
        interrupted.data['on_epoch_end'] = save_at_two
        interrupted.fit()

        resumed = fresh_method()
        resumed.restore_training_state(load_checkpoint(path))
        resumed.fit()

        reloaded = fresh_method()
        reloaded.load_state_dict(load_checkpoint(path)['model_state'])
        loader = make_eval_loader(validation['X'], validation['y'], mean, std, 64)
        first = reloaded.predict(loader)
        second = reloaded.predict(loader)
        repeat_passed = bool(torch.equal(first['logits'], second['logits']))
        _record(checks, f'{dataset_key}: reloaded model evaluates repeatably', repeat_passed,
                f"accuracy {first['accuracy']:.4f}")

    differences = [(a - b).abs().max().item() for a, b in
                   zip(straight.state_dict().values(), resumed.state_dict().values())
                   if a.is_floating_point()]
    worst = max(differences)
    history_match = np.allclose([row['train_loss'] for row in straight.history],
                                [row['train_loss'] for row in resumed.history], atol=1e-5)
    _record(checks, f'{dataset_key}: resume matches an uninterrupted run',
            worst <= 1e-5 and history_match, f'largest parameter difference {worst:.2e}')


def run_preflight(device='auto', threads=None, root=None, data_dir=None, datasets=None):
    configure_threads(threads)
    resolved = resolve_device(device)
    seed_everything(SEED)
    checks = []
    for dataset_key in datasets or DATASET_ORDER:
        print(f'== {dataset_key} on {resolved}', flush=True)
        train, train_idx, val_idx = check_data(dataset_key, data_dir, checks)
        check_all_configs(dataset_key, resolved, checks)
        check_overfit(dataset_key, train, train_idx, resolved, checks)
        check_checkpoint_cycle(dataset_key, train, train_idx, val_idx, resolved, checks)
    report = {'all_passed': all(item['passed'] for item in checks), 'checks': checks,
              'environment': capture_environment(resolved, REPO)}
    folder = Path(root) / 'preflight'
    write_json(folder / f'preflight_report_{resolved.type}.json', report)
    lines = [f"{'PASS' if item['passed'] else 'FAIL'}  {item['check']}: {item['detail']}"
             for item in checks]
    write_text(folder / f'preflight_report_{resolved.type}.txt', '\n'.join(lines) + '\n')
    failed = sum(not item['passed'] for item in checks)
    print(f'{len(checks) - failed}/{len(checks)} checks passed', flush=True)
    return report
