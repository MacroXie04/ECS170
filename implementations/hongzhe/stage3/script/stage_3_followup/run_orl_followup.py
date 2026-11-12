"""Supplementary ORL experiment: cross-validated choice, then a refit on all data.

The formal ORL validation set is saturated (7 of 8 configs reach 100% on 40
images) and holds back one training image per person. This experiment addresses
both, and neither changes nor replaces the formal Stage 3 results.

Run the stages in order; each refuses to start until the previous one finished:
    python script/stage_3_followup/run_orl_followup.py register
    python script/stage_3_followup/run_orl_followup.py cv --jobs 3
    python script/stage_3_followup/run_orl_followup.py refit
    python script/stage_3_followup/run_orl_followup.py test

Only `test` reads the test split, and it runs once.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from local_code.stage_3_code.Dataset_Loader import (Dataset_Loader,  # noqa: E402
                                                    compute_channel_stats, make_eval_loader)
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics  # noqa: E402
from local_code.stage_3_code.Experiment_Config import (FORMAL_SEEDS, REGISTRY,  # noqa: E402
                                                       SPLIT_SEED)
from local_code.stage_3_code.Method_CNN import load_checkpoint  # noqa: E402
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL  # noqa: E402
from local_code.stage_3_code.Result_Saver import Result_Saver  # noqa: E402
from local_code.stage_3_code.Runtime import (atomic_write, capture_environment,  # noqa: E402
                                             configure_threads, read_json, resolve_device,
                                             seed_everything, sha256_file, write_json,
                                             write_text)
from local_code.stage_3_code.Setting_Stage3 import attach_training_data  # noqa: E402

ROOT = REPO / 'result' / 'stage_3_result' / 'orl_followup'
DATA = REPO / 'data' / 'stage_3_data'
FORMAL_RUNS = REPO / 'result' / 'stage_3_result' / 'runs' / 'orl'
FOLDS = 9
CV_SEED_BASE = 4000
NUM_CLASSES = 40
CANDIDATES = REGISTRY['orl']

PROTOCOL = {
    'purpose': 'Supplementary experiment. It does not replace the formal ORL results.',
    'disclosure': 'Proposed after the formal ORL test scores were known: the formal final '
                  'model scored 95.00% and the per-config test means were visible. Any '
                  'result here is therefore not a blind estimate.',
    'candidates': 'The 8 locked ORL configurations, unchanged; no new hyperparameters.',
    'cv': f'{FOLDS}-fold StratifiedKFold over the 360 training images, random_state '
          f'{SPLIT_SEED}; each fold holds out exactly one image per person. Training seed '
          f'{CV_SEED_BASE} + fold.',
    'cv_score': 'Held-out predictions from the final epoch, matching the refit, which has '
                'no validation set. No epoch is chosen using held-out data.',
    'selection_rule': 'Most correct of the 360 pooled held-out predictions; then lowest '
                      'mean held-out cross-entropy; then fewest parameters.',
    'refit_rule': 'Selected config on all 360 training images, its fixed epoch budget, '
                  f'seeds {list(FORMAL_SEEDS)}, final epoch kept. Normalization from those '
                  '360 images.',
    'primary_result': 'Test accuracy of the 3-seed ensemble (mean of softmax outputs), which '
                      'needs no seed choice. Per-seed results are reported alongside.',
    'test_reads': 'Once, after refit, with no further changes.',
}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def make_folds(labels):
    """Held-out indices per fold; every fold has exactly one image of each person."""
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SPLIT_SEED)
    folds = [held for _, held in splitter.split(np.zeros(len(labels)), labels)]
    people = np.arange(labels.max() + 1)
    for held in folds:
        if not np.array_equal(np.sort(labels[held]), people):
            raise AssertionError('A fold does not hold exactly one image per person')
    return folds


def load_training_split():
    return Dataset_Loader('orl', 'ORL', folder=DATA).load(splits=('train',))['train']


def subset(data, indices):
    return {'X': data['X'][indices], 'y': data['y'][indices]}


def train_to_final_epoch(config, seed, train, evaluation, device):
    """Train the full budget. The final epoch is kept; no checkpoint is chosen."""
    seed_everything(seed)
    method = Method_CNN_ORL(config, device=device)
    method.log = lambda _message: None
    mean, std = compute_channel_stats(train['X'])
    normalization = {'mean': mean, 'std': std, 'computed_on': 'this run\'s training images'}
    attach_training_data(method, train, evaluation, normalization, seed, lambda _improved: None)
    method.fit()
    return method


def renamed_history(history, prefix):
    """fit() names its evaluation columns val_*; say what they really measured."""
    return [{key.replace('val_', prefix): value for key, value in row.items()
             if key != 'selected_so_far'} for row in history]


def run_fold(task):
    name, fold, device, threads = task
    folder = ROOT / 'cv' / name / f'fold_{fold}'
    if (folder / 'result.json').exists():
        return name, fold, 'skipped'
    configure_threads(threads)
    data = load_training_split()
    held = make_folds(data['y'])[fold]
    kept = np.setdiff1d(np.arange(len(data['y'])), held)
    seed = CV_SEED_BASE + fold
    method = train_to_final_epoch(CANDIDATES[name], seed, subset(data, kept),
                                  subset(data, held), resolve_device(device))
    output = method.predict(method.data['val_loader'])
    write_json(folder / 'history.json', renamed_history(method.history, 'held_out_'))

    def write_predictions(target):
        with open(target, 'wb') as handle:
            np.savez_compressed(handle, train_indices=held, true_y=output['true_y'],
                                pred_y=output['pred_y'])
    atomic_write(folder / 'held_out_predictions.npz', write_predictions)
    write_json(folder / 'result.json', {
        'config': name, 'fold': fold, 'seed': seed, 'held_out': int(held.size),
        'correct': int((output['true_y'] == output['pred_y']).sum()),
        'cross_entropy': output['cross_entropy'], 'epochs': method.history[-1]['epoch'],
        'train_seconds': method.train_seconds, 'parameters': method.parameter_count(),
    })
    return name, fold, 'trained'


def summarize_cv():
    rows = []
    for name in CANDIDATES:
        results = [read_json(ROOT / 'cv' / name / f'fold_{fold}' / 'result.json')
                   for fold in range(FOLDS)]
        folds_correct = [item['correct'] for item in results]
        rows.append({
            'config': name, 'correct': sum(folds_correct),
            'held_out': sum(item['held_out'] for item in results),
            'accuracy': sum(folds_correct) / sum(item['held_out'] for item in results),
            'fold_accuracy_std': float(np.std([item['correct'] / item['held_out']
                                               for item in results], ddof=1)),
            'mean_cross_entropy': float(np.mean([item['cross_entropy'] for item in results])),
            'parameters': results[0]['parameters'],
            'train_seconds_mean': float(np.mean([item['train_seconds'] for item in results])),
        })
    return rows


def select_config(rows):
    """PROTOCOL['selection_rule']; counts are integers, so ties are exact."""
    return min(rows, key=lambda row: (-row['correct'], row['mean_cross_entropy'],
                                      row['parameters']))


def ensemble_predictions(probabilities):
    """Mean of per-seed softmax outputs, shaped (seeds, samples, classes)."""
    return np.mean(probabilities, axis=0).argmax(axis=1)


def require_registered():
    path = ROOT / 'protocol.json'
    if not path.exists():
        raise PermissionError('Run the register stage first')
    registered = read_json(path)
    if registered['script_sha256'] != sha256_file(__file__):
        raise PermissionError('This script changed after registration')
    return registered


def stage_register(device):
    path = ROOT / 'protocol.json'
    if path.exists():
        raise FileExistsError(f'{path} already exists')
    write_json(path, {
        **PROTOCOL, 'registered_at': utc_now(), 'script_sha256': sha256_file(__file__),
        'candidate_sha256': {name: config.fingerprint() for name, config in CANDIDATES.items()},
        'environment': capture_environment(resolve_device(device), REPO),
    })
    print(path)


def stage_cv(device, threads, jobs):
    require_registered()
    tasks = [(name, fold, device, threads) for name in CANDIDATES for fold in range(FOLDS)]
    context = multiprocessing.get_context('spawn')  # CUDA cannot survive a fork
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
        for index, (name, fold, action) in enumerate(pool.map(run_fold, tasks), start=1):
            print(f'{utc_now()} {index}/{len(tasks)} {name} fold {fold}: {action}', flush=True)
    rows = summarize_cv()
    chosen = select_config(rows)
    write_json(ROOT / 'cv_summary.json', rows)
    write_json(ROOT / 'selection.json', {'config': chosen['config'], 'selected_at': utc_now(),
                                         'rule': PROTOCOL['selection_rule'], 'row': chosen})
    for row in sorted(rows, key=lambda item: -item['correct']):
        print(f"{row['config']:14} {row['correct']}/{row['held_out']} "
              f"acc={row['accuracy']:.4f} ce={row['mean_cross_entropy']:.4f}")
    print('selected:', chosen['config'])


def stage_refit(device, threads):
    require_registered()
    selection_path = ROOT / 'selection.json'
    if not selection_path.exists():
        raise PermissionError('Run the cv stage first')
    name = read_json(selection_path)['config']
    configure_threads(threads)
    data = load_training_split()
    for seed in FORMAL_SEEDS:
        folder = ROOT / 'refit' / name / f'seed_{seed}'
        if (folder / 'result.json').exists():
            continue
        # No held-out data exists, so the per-epoch evaluation uses training images.
        method = train_to_final_epoch(CANDIDATES[name], seed, data, data, resolve_device(device))
        method.save_checkpoint(folder / 'checkpoints' / 'final.pt')
        write_json(folder / 'history.json', renamed_history(method.history, 'train_eval_'))
        last = method.history[-1]
        write_json(folder / 'result.json', {
            'config': name, 'seed': seed, 'train_images': int(len(data['y'])),
            'epochs': last['epoch'], 'final_train_loss': last['train_loss'],
            'train_seconds': method.train_seconds,
            'environment': capture_environment(resolve_device(device), REPO),
        })
        print(f"{name} seed {seed}: epochs={last['epoch']} train_loss={last['train_loss']:.5f} "
              f"seconds={method.train_seconds:.1f}", flush=True)


def evaluate_predictions(true_y, pred_y):
    evaluator = Evaluate_Metrics(num_classes=NUM_CLASSES)
    evaluator.data = {'true_y': true_y, 'pred_y': pred_y}
    return evaluator.evaluate()


def formal_reference():
    """Formal ORL numbers, read from their own files for comparison."""
    best = read_json(REPO / 'result' / 'stage_3_result' / 'summary' / 'best_configs.json')['orl']
    return {key: best[key] for key in ('config', 'final_seed', 'final_test_accuracy',
                                       'test_accuracy_mean')}


def stage_test(device, threads):
    require_registered()
    results_path = ROOT / 'test_results.json'
    if results_path.exists():
        raise FileExistsError('The test set has already been read for this experiment')
    name = read_json(ROOT / 'selection.json')['config']
    folders = {seed: ROOT / 'refit' / name / f'seed_{seed}' for seed in FORMAL_SEEDS}
    for folder in folders.values():
        if not (folder / 'result.json').exists():
            raise PermissionError('Run the refit stage first')
    configure_threads(threads)
    resolved = resolve_device(device)
    seed_everything(SPLIT_SEED)
    with open(ROOT / 'test_access_log.jsonl', 'a', encoding='utf-8') as handle:
        handle.write(json.dumps({'at': utc_now(), 'config': name, 'device': str(resolved)}) + '\n')
    test = Dataset_Loader('orl', 'ORL', folder=DATA).load(splits=('test',))['test']
    per_seed, probabilities = {}, []
    for seed, folder in folders.items():
        checkpoint = load_checkpoint(folder / 'checkpoints' / 'final.pt')
        method = Method_CNN_ORL(CANDIDATES[name], device=resolved)
        method.load_state_dict(checkpoint['model_state'])
        normalization = checkpoint['normalization']
        output = method.predict(make_eval_loader(test['X'], test['y'], normalization['mean'],
                                                 normalization['std'], 256))
        probabilities.append(output['logits'].softmax(dim=1).numpy())
        metrics = evaluate_predictions(output['true_y'], output['pred_y'])
        metrics['test_cross_entropy'] = output['cross_entropy']
        saver = Result_Saver('saver', 'ORL follow-up')
        saver.result_destination_folder_path = str(folder)
        saver.data = {'metrics': metrics, 'predictions': {
            'true_y': output['true_y'], 'pred_y': output['pred_y'],
            'true_label_original': output['true_y'] + 1,
            'pred_label_original': output['pred_y'] + 1}}
        saver.save()
        per_seed[seed] = metrics
    ensemble_pred = ensemble_predictions(np.stack(probabilities))
    ensemble = evaluate_predictions(test['y'], ensemble_pred)
    accuracies = [metrics['accuracy'] for metrics in per_seed.values()]
    results = {
        'tested_at': utc_now(), 'config': name, 'device': str(resolved),
        'primary_ensemble': ensemble, 'per_seed': per_seed,
        'per_seed_accuracy_mean': float(np.mean(accuracies)),
        'per_seed_accuracy_std': float(np.std(accuracies, ddof=1)),
        'formal_reference': formal_reference(),
    }

    def write_predictions(target):
        with open(target, 'wb') as handle:
            np.savez_compressed(handle, true_y=test['y'], pred_y=ensemble_pred,
                                true_label_original=test['y'] + 1,
                                pred_label_original=ensemble_pred + 1)
    atomic_write(ROOT / 'ensemble_test_predictions.npz', write_predictions)
    write_json(results_path, results)
    write_text(ROOT / 'FOLLOWUP.md', followup_markdown(results))
    print((ROOT / 'FOLLOWUP.md').read_text(encoding='utf-8'))


def followup_markdown(results):
    rows = sorted(read_json(ROOT / 'cv_summary.json'), key=lambda item: -item['correct'])
    formal = results['formal_reference']
    ensemble = results['primary_ensemble']
    lines = [
        '# ORL follow-up experiment\n', PROTOCOL['purpose'], '', PROTOCOL['disclosure'], '',
        '## Cross-validation on the 360 training images\n',
        f'Selection rule: {PROTOCOL["selection_rule"]}\n',
        '| Config | Correct | Accuracy % | Mean cross-entropy | Train s |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        marker = ' **(selected)**' if row['config'] == results['config'] else ''
        lines.append(f"| {row['config']}{marker} | {row['correct']}/{row['held_out']} | "
                     f"{100 * row['accuracy']:.2f} | {row['mean_cross_entropy']:.4f} | "
                     f"{row['train_seconds_mean']:.0f} |")
    lines += [
        '', '## Test results\n',
        '| Model | Accuracy % | Precision (macro) % | Recall (macro) % | F1 (macro) % |',
        '|---|---:|---:|---:|---:|',
    ]
    models = [('3-seed ensemble (primary)', ensemble)]
    models += [(f'Seed {seed}', metrics) for seed, metrics in results['per_seed'].items()]
    for label, metrics in models:
        lines.append(f"| {label} | {100 * metrics['accuracy']:.2f} | "
                     f"{100 * metrics['precision_macro']:.2f} | "
                     f"{100 * metrics['recall_macro']:.2f} | {100 * metrics['f1_macro']:.2f} |")
    lines += [
        '', f"Per-seed mean: {100 * results['per_seed_accuracy_mean']:.2f} ± "
            f"{100 * results['per_seed_accuracy_std']:.2f}%.",
        '', f"Formal result, for comparison: `{formal['config']}`, final model "
            f"{100 * formal['final_test_accuracy']:.2f}%, config mean "
            f"{100 * formal['test_accuracy_mean']:.2f}%.",
        '', 'The test set has 40 images, so one image is 2.5 percentage points.', '',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage', choices=('register', 'cv', 'refit', 'test'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--jobs', type=int, default=3)
    args = parser.parse_args()
    if args.stage == 'register':
        stage_register(args.device)
    elif args.stage == 'cv':
        stage_cv(args.device, args.threads, args.jobs)
    elif args.stage == 'refit':
        stage_refit(args.device, args.threads)
    else:
        stage_test(args.device, args.threads)


if __name__ == '__main__':
    main()
