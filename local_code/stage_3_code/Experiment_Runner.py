"""Compose template objects into runs: train, resume, evaluate, verify, run-all.

Run layout: <root>/runs/<dataset>/<config>/seed_<seed>/. Runs that change a
registered config or carry a tag are pilots: they live under <root>/pilot_runs/
and may never read the test set.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import traceback

import numpy as np

from local_code.stage_3_code.Dataset_Loader import DATASET_SPECS, Dataset_Loader
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Experiment_Config import (BEST_CONFIG_RULE, CHECKPOINT_RULE,
                                                       DATASET_ORDER, FORMAL_SEEDS,
                                                       REFERENCE_ACCURACY, REGISTRY,
                                                       SPLIT_SEED, CNNConfig,
                                                       apply_overrides, get_config)
from local_code.stage_3_code.Method_CNN import load_checkpoint
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Report_Artifacts import model_summary_text
from local_code.stage_3_code.Result_Loader import Result_Loader
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Runtime import (capture_environment, configure_threads,
                                             exclusive_run_lock, read_json, resolve_device,
                                             seed_everything, source_fingerprint, write_json,
                                             write_text)
from local_code.stage_3_code.Setting_Stage3 import Setting_Stage3

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / 'result' / 'stage_3_result'
DEFAULT_DATA = REPO / 'data' / 'stage_3_data'
METHOD_CLASSES = {'mnist': Method_CNN_MNIST, 'orl': Method_CNN_ORL, 'cifar': Method_CNN_CIFAR}
METRIC_NAMES = ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                'precision_weighted', 'recall_weighted', 'f1_weighted')
METRIC_TOLERANCE = 1e-6
LOSS_TOLERANCE = 1e-4


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def run_directory(root, dataset, config_name, seed, tag=None):
    if tag:
        return Path(root) / 'pilot_runs' / dataset / config_name / f'seed_{seed}__{tag}'
    return Path(root) / 'runs' / dataset / config_name / f'seed_{seed}'


class RunLog:
    """Timestamped lines to both stdout and the run's train.log."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message):
        line = f'{utc_now()} {message}'
        print(line, flush=True)
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write(line + '\n')


def build_setting(config, seed, run_dir, device, data_dir, root, log=print):
    """Instantiate the course template's four roles for one run."""
    spec = DATASET_SPECS[config.dataset]
    dataset_obj = Dataset_Loader(config.dataset, f'ECS 170 Stage 3 {spec.file_name}',
                                 folder=data_dir)
    method_obj = METHOD_CLASSES[config.dataset](config, device=device)
    method_obj.log = log
    result_obj = Result_Saver('saver', 'Stage 3 test results')
    result_obj.result_destination_folder_path = str(run_dir)
    evaluate_obj = Evaluate_Metrics('multiclass metrics', 'accuracy, precision, recall, F1',
                                    num_classes=spec.num_classes)
    setting_obj = Setting_Stage3('fixed split', 'Instructor train/test split with a '
                                 'stratified validation hold-out', run_dir, seed,
                                 splits_dir=Path(root) / 'splits')
    setting_obj.prepare(dataset_obj, method_obj, result_obj, evaluate_obj)
    return setting_obj


def _record_run_start(run_dir, config, seed, formal, overrides, tag, concurrent_jobs, method):
    write_json(run_dir / 'config.json', {
        'config': config.to_dict(), 'config_sha256': config.fingerprint(),
        'seed': seed, 'split_seed': SPLIT_SEED, 'formal': formal,
        'overrides': list(overrides), 'tag': tag,
        'parameters': method.parameter_count(),
        'checkpoint_selection_rule': CHECKPOINT_RULE,
        'concurrent_jobs': concurrent_jobs,
    })
    write_text(run_dir / 'model_summary.txt', model_summary_text(method))


def _append_environment(run_dir, device, start_epoch, concurrent_jobs):
    """One entry per training session, so a resumed run's timing can be split."""
    path = run_dir / 'environment.json'
    sessions = read_json(path)['sessions'] if path.exists() else []
    sessions.append({'started_at': utc_now(), 'start_epoch': start_epoch,
                     'concurrent_jobs': concurrent_jobs,
                     **capture_environment(device, REPO)})
    write_json(path, {'sessions': sessions})


def train_run(dataset, config_name, seed, *, device='auto', threads=None, overrides=(),
              tag=None, concurrent_jobs=1, root=DEFAULT_ROOT, data_dir=DEFAULT_DATA):
    """Train one configuration from scratch. Never reads the test set."""
    config = apply_overrides(get_config(dataset, config_name), overrides)
    if overrides and not tag:
        tag = 'override_' + hashlib.sha256(json.dumps(sorted(overrides)).encode()).hexdigest()[:8]
    formal = tag is None
    if formal and seed not in FORMAL_SEEDS:
        raise ValueError(f'Seed {seed} is not a formal seed {FORMAL_SEEDS}; add --tag '
                         'for a pilot run')
    run_dir = run_directory(root, dataset, config_name, seed, tag)
    with exclusive_run_lock(run_dir):
        if (run_dir / 'checkpoints' / 'last.pt').exists():
            raise FileExistsError(f'{run_dir} already has results; use resume or a new --tag')
        configure_threads(threads)
        resolved = resolve_device(device)
        seed_everything(seed)
        log = RunLog(run_dir / 'train.log')
        setting_obj = build_setting(config, seed, run_dir, resolved, data_dir, root, log)
        _record_run_start(run_dir, config, seed, formal, overrides, tag, concurrent_jobs,
                          setting_obj.method)
        _append_environment(run_dir, resolved, 1, concurrent_jobs)
        log(f'start {dataset}/{config_name} seed={seed} device={resolved} '
            f'params={setting_obj.method.parameter_count()}')
        return _train_guarded(setting_obj, log)


def resume_run(dataset, config_name, seed, *, checkpoint=None, device='auto', threads=None,
               tag=None, concurrent_jobs=1, root=DEFAULT_ROOT, data_dir=DEFAULT_DATA):
    """Continue a run from last.pt, or from an explicit checkpoint file."""
    run_dir = run_directory(root, dataset, config_name, seed, tag)
    checkpoint = Path(checkpoint) if checkpoint else run_dir / 'checkpoints' / 'last.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(f'No checkpoint to resume from: {checkpoint}')
    with exclusive_run_lock(run_dir):
        config = CNNConfig.from_dict(read_json(run_dir / 'config.json')['config'])
        saved = _require_run_checkpoint(checkpoint, run_dir, config, seed)
        log = RunLog(run_dir / 'train.log')
        if saved['epoch'] >= config.max_epoch:
            # A stop between the final checkpoint and the status update lands here.
            log(f'{checkpoint} already reached epoch {saved["epoch"]}; nothing to resume')
            Setting_Stage3('', '', run_dir, seed).finish_from_checkpoint(saved)
            return saved['best']
        if (run_dir / 'test_metrics.json').exists():
            raise PermissionError(f'{run_dir} is evaluated; more training would replace '
                                  'the model its test result describes')
        configure_threads(threads)
        resolved = resolve_device(device)
        seed_everything(seed)
        setting_obj = build_setting(config, seed, run_dir, resolved, data_dir, root, log)
        _append_environment(run_dir, resolved, saved['epoch'] + 1, concurrent_jobs)
        return _train_guarded(setting_obj, log, resume_checkpoint=checkpoint)


def _train_guarded(setting_obj, log, resume_checkpoint=None):
    try:
        selected = setting_obj.train_phase(resume_checkpoint=resume_checkpoint)
    except Exception as error:
        setting_obj.update_status('failed', error=repr(error), failed_at=utc_now())
        log(f'FAILED: {error!r}\n{traceback.format_exc()}')
        raise
    log(f'finished: selected epoch {selected["epoch"]}, '
        f'val_acc={selected["val_accuracy"]:.4f}, val_ce={selected["val_loss"]:.5f}')
    return selected


def registry_snapshot():
    return {dataset: {name: {'sha256': config.fingerprint(),
                             'changed_factor': config.changed_factor,
                             'description': config.description}
                      for name, config in configs.items()}
            for dataset, configs in REGISTRY.items()}


def lock_protocol(root=DEFAULT_ROOT):
    """Fix configurations and selection rules before any test evaluation."""
    path = Path(root) / 'protocol.json'
    snapshot = registry_snapshot()
    if path.exists():
        if read_json(path)['configs'] != snapshot:
            raise RuntimeError(f'{path} differs from the current registry. Changing the '
                               'plan after locking it needs a new result root.')
        return path
    write_json(path, {
        'locked_at': utc_now(), 'split_seed': SPLIT_SEED, 'formal_seeds': list(FORMAL_SEEDS),
        'checkpoint_rule': CHECKPOINT_RULE, 'best_config_rule': BEST_CONFIG_RULE,
        'reference_accuracy': REFERENCE_ACCURACY, 'configs': snapshot,
        'stage3_source_sha256': source_fingerprint(REPO),
    })
    return path


def _require_protocol(root, config, meta):
    if not meta.get('formal'):
        raise PermissionError('Pilot runs may not read the test set')
    path = Path(root) / 'protocol.json'
    if not path.exists():
        raise PermissionError('Run lock-protocol before reading the test set')
    protocol = read_json(path)
    if meta['seed'] not in protocol['formal_seeds']:
        raise PermissionError(f"Seed {meta['seed']} is not in the locked protocol")
    locked = protocol['configs'].get(config.dataset, {}).get(config.name)
    if not locked or locked['sha256'] != config.fingerprint():
        raise PermissionError(f'{config.dataset}/{config.name} is not the locked config')


def _require_finished_run(setting_obj, config):
    """Only a run that completed its epoch budget may read the test set."""
    status = setting_obj.read_status()
    if (status.get('state') not in ('trained', 'evaluated')
            or status.get('epoch') != config.max_epoch):
        raise PermissionError(f'{setting_obj.run_dir} has not finished training: '
                              f"state={status.get('state')} epoch={status.get('epoch')}")
    return status


def _require_run_checkpoint(path, run_dir, config, seed):
    """A checkpoint must come from this run; a pilot's file cannot stand in for it."""
    if not path.resolve().is_relative_to(Path(run_dir).resolve()):
        raise PermissionError(f'{path} is outside {run_dir}')
    checkpoint = load_checkpoint(path)
    fingerprint = CNNConfig.from_dict(checkpoint['config']).fingerprint()
    if fingerprint != config.fingerprint() or checkpoint['seed'] != seed:
        raise PermissionError(f'{path} does not belong to {run_dir}')
    return checkpoint


def _log_test_access(root, run_dir, checkpoint, purpose, device):
    with open(Path(root) / 'test_access_log.jsonl', 'a', encoding='utf-8') as handle:
        handle.write(json.dumps({'at': utc_now(), 'run': str(run_dir), 'purpose': purpose,
                                 'checkpoint': str(checkpoint), 'device': str(device)}) + '\n')


def compare_evaluations(saved, fresh_metrics, fresh_predictions, num_classes):
    """Compare a stored evaluation with a fresh one and with its own predictions."""
    differences = {name: abs(saved['metrics'][name] - fresh_metrics[name])
                   for name in METRIC_NAMES}
    loss_difference = abs(saved['metrics']['test_cross_entropy']
                          - fresh_metrics['test_cross_entropy'])
    changed = int((saved['predictions']['pred_y'] != fresh_predictions['pred_y']).sum())
    labels_match = bool(np.array_equal(saved['predictions']['true_y'],
                                       fresh_predictions['true_y']))
    worst = max(differences.values())
    recomputed = recheck_saved_metrics(saved, num_classes)
    # Rounding can flip a borderline prediction across devices, but not on one.
    same_device = saved['metrics'].get('device') == fresh_metrics.get('device')
    return {'max_metric_difference': worst, 'test_cross_entropy_difference': loss_difference,
            'changed_predictions': changed, 'true_labels_match': labels_match,
            'recomputed_from_predictions_difference': recomputed, 'same_device': same_device,
            'passed': (worst <= METRIC_TOLERANCE and loss_difference <= LOSS_TOLERANCE
                       and labels_match and recomputed <= METRIC_TOLERANCE
                       and (changed == 0 or not same_device))}


def recheck_saved_metrics(saved, num_classes):
    """Recompute metrics from saved predictions alone, with no model involved."""
    evaluator = Evaluate_Metrics(num_classes=num_classes)
    evaluator.data = saved['predictions']
    recomputed = evaluator.evaluate()
    return max(abs(saved['metrics'][name] - recomputed[name]) for name in METRIC_NAMES)


def evaluate_run(dataset, config_name, seed, *, checkpoint=None, device='auto', threads=None,
                 root=DEFAULT_ROOT, data_dir=DEFAULT_DATA):
    """Score a formal run's checkpoint on the full test set.

    The first evaluation of the selected checkpoint is saved. A later call
    verifies against that record instead of replacing it. An explicitly named
    checkpoint is reported under evaluations/ and never changes the formal result.
    """
    run_dir = run_directory(root, dataset, config_name, seed)
    with exclusive_run_lock(run_dir):
        return _evaluate_locked(dataset, seed, run_dir, checkpoint, device, threads, root,
                                data_dir)


def _evaluate_locked(dataset, seed, run_dir, checkpoint, device, threads, root, data_dir):
    meta = read_json(run_dir / 'config.json')
    config = CNNConfig.from_dict(meta['config'])
    _require_protocol(root, config, meta)
    configure_threads(threads)
    resolved = resolve_device(device)
    seed_everything(seed)
    setting_obj = build_setting(config, seed, run_dir, resolved, data_dir, root)
    status = _require_finished_run(setting_obj, config)
    selected = setting_obj.selected_checkpoint
    path = Path(checkpoint) if checkpoint else selected
    is_selected = path.resolve() == selected.resolve()
    loaded = _require_run_checkpoint(path, run_dir, config, seed)
    if is_selected and loaded['epoch'] != status['selected_epoch']:
        raise ValueError(f"{path} holds epoch {loaded['epoch']}, but the run selected "
                         f"epoch {status['selected_epoch']}")
    saver = setting_obj.result
    if not is_selected:
        saver.result_destination_folder_path = str(run_dir / 'evaluations')
        saver.result_destination_file_name = path.stem
    already_saved = saver.metrics_path.exists()
    _log_test_access(root, run_dir, path, 'verify' if already_saved else 'evaluate', resolved)
    metrics, predictions = setting_obj.test_phase(path)
    if already_saved:
        loader = Result_Loader('loader', '')
        loader.result_destination_folder_path = saver.result_destination_folder_path
        loader.result_destination_file_name = saver.result_destination_file_name
        saved = loader.load()
        report = compare_evaluations(saved, metrics, predictions,
                                     DATASET_SPECS[dataset].num_classes)
        if is_selected and report['passed'] and status['state'] != 'evaluated':
            # A stop after saving but before this update lands here.
            setting_obj.update_status('evaluated', test_accuracy=saved['metrics']['accuracy'],
                                      evaluated_at=utc_now())
        return {'run': str(run_dir), 'action': 'verified',
                'accuracy': saved['metrics']['accuracy'], **report}
    saver.data = {'metrics': metrics, 'predictions': predictions}
    saver.save()
    if is_selected:
        setting_obj.update_status('evaluated', test_accuracy=metrics['accuracy'],
                                  evaluated_at=utc_now())
    return {'run': str(run_dir), 'action': 'evaluated', 'accuracy': metrics['accuracy']}


def planned_runs(datasets=None, configs=None, seeds=None):
    """Seed-major and CIFAR first, so one full panel finishes as early as possible."""
    chosen = [dataset for dataset in DATASET_ORDER if not datasets or dataset in datasets]
    plan = []
    for seed in seeds or FORMAL_SEEDS:
        for dataset in chosen:
            for name in REGISTRY[dataset]:
                if not configs or name in configs:
                    plan.append((dataset, name, seed))
    return plan


def _run_state(root, dataset, name, seed):
    run_dir = run_directory(root, dataset, name, seed)
    status_path = run_dir / 'status.json'
    state = read_json(status_path).get('state') if status_path.exists() else None
    if state in ('trained', 'evaluated'):
        return state
    return 'resumable' if (run_dir / 'checkpoints' / 'last.pt').exists() else 'new'


class _TrainingProcesses:
    """Tracks child trainers so that stopping run-all stops them too."""

    def __init__(self):
        self.stopping = threading.Event()
        self._guard = threading.Lock()
        self._running = set()

    def run(self, arguments, output):
        with self._guard:
            if self.stopping.is_set():
                return None
            process = subprocess.Popen(arguments, stdout=output, stderr=subprocess.STDOUT)
            self._running.add(process)
        code = process.wait()
        with self._guard:
            self._running.discard(process)
        return code

    def stop(self, _signal_number=None, _frame=None):
        self.stopping.set()
        with self._guard:
            for process in self._running:
                process.terminate()


def _launch(processes, entry_script, command, dataset, name, seed, device, threads, jobs, root,
            data_dir):
    arguments = [sys.executable, str(entry_script), command, '--dataset', dataset,
                 '--config', name, '--seed', str(seed), '--device', device,
                 '--result-root', str(root), '--data-dir', str(data_dir),
                 '--concurrent-jobs', str(jobs)]
    if threads:
        arguments += ['--threads', str(threads)]
    run_dir = run_directory(root, dataset, name, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'process_output.log', 'a', encoding='utf-8') as output:
        return processes.run(arguments, output)


def _train_one(processes, entry_script, run, device, threads, jobs, root, data_dir, attempts,
               progress):
    dataset, name, seed = run
    for attempt in range(1, attempts + 1):
        if processes.stopping.is_set():
            return False
        state = _run_state(root, dataset, name, seed)
        if state in ('trained', 'evaluated'):
            progress(f'skip   {dataset}/{name} seed={seed}: already {state}')
            return True
        command = 'resume' if state == 'resumable' else 'train'
        progress(f'{command:6} {dataset}/{name} seed={seed} (attempt {attempt})')
        code = _launch(processes, entry_script, command, dataset, name, seed, device, threads,
                       jobs, root, data_dir)
        if code == 0:
            status = read_json(run_directory(root, dataset, name, seed) / 'status.json')
            progress(f'done   {dataset}/{name} seed={seed}: val_acc='
                     f'{status["val_accuracy"]:.4f} in {status["train_seconds"]:.0f}s')
            return True
        progress(f'FAILED {dataset}/{name} seed={seed}: exit code {code}')
    return False


def run_all(entry_script, *, datasets=None, configs=None, seeds=None, jobs=1, threads=None,
            device='auto', attempts=2, root=DEFAULT_ROOT, data_dir=DEFAULT_DATA):
    """Train the whole plan, then evaluate. Rerun it to continue after a stop.

    Must run on the main thread, which owns the signal handlers.
    """
    lock_protocol(root)
    plan = planned_runs(datasets, configs, seeds)
    threads = threads or max(1, (os.cpu_count() or 1) // jobs)
    log = RunLog(Path(root) / 'run_all.log')
    log(f'plan: {len(plan)} runs, jobs={jobs}, threads={threads}, device={device}')
    processes = _TrainingProcesses()
    previous = {number: signal.signal(number, processes.stop)
                for number in (signal.SIGINT, signal.SIGTERM)}

    def train(run):
        return _train_one(processes, entry_script, run, device, threads, jobs, root, data_dir,
                          attempts, log)

    try:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            outcomes = list(pool.map(train, plan))
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    failures = [run for run, passed in zip(plan, outcomes) if not passed]
    if failures:
        # The test set stays unread until every planned run has finished training.
        log(f'stopped before testing; {len(failures)} of {len(plan)} runs unfinished: '
            f'{failures}')
        return failures
    for run in plan:
        dataset, name, seed = run
        if _run_state(root, dataset, name, seed) != 'trained':
            continue
        try:
            outcome = evaluate_run(dataset, name, seed, device=device, threads=threads,
                                   root=root, data_dir=data_dir)
        except Exception as error:  # one bad run must not block the others
            log(f'FAILED test {dataset}/{name} seed={seed}: {error!r}')
            failures.append(run)
            continue
        if not outcome.get('passed', True):
            log(f'FAILED test {dataset}/{name} seed={seed}: saved result does not match')
            failures.append(run)
            continue
        log(f"test   {dataset}/{name} seed={seed}: {outcome['action']} "
            f"accuracy={outcome['accuracy']:.4f}")
    log(f'finished: {len(plan) - len(failures)} of {len(plan)} runs complete, '
        f'failed: {failures}')
    return failures


def verify_all(*, device='auto', threads=None, root=DEFAULT_ROOT, data_dir=DEFAULT_DATA):
    """Reload every selected checkpoint and compare with its saved evaluation."""
    reports, missing = [], []
    for dataset, name, seed in planned_runs():
        if not (run_directory(root, dataset, name, seed) / 'test_metrics.json').exists():
            missing.append(f'{dataset}/{name}/seed_{seed}')
            continue
        report = evaluate_run(dataset, name, seed, device=device, threads=threads,
                              root=root, data_dir=data_dir)
        report.update(dataset=dataset, config=name, seed=seed)
        reports.append(report)
        print(f"{dataset}/{name} seed={seed}: passed={report['passed']} "
              f"max_diff={report['max_metric_difference']:.2e} "
              f"changed={report['changed_predictions']}", flush=True)
    summary = {
        'verified_at': utc_now(), 'runs_checked': len(reports), 'runs_missing': missing,
        'all_passed': (bool(reports) and not missing
                       and all(item['passed'] for item in reports)),
        'metric_tolerance': METRIC_TOLERANCE, 'loss_tolerance': LOSS_TOLERANCE,
        'device': str(resolve_device(device)), 'reports': reports,
    }
    write_json(Path(root) / 'summary' / 'reload_verification.json', summary)
    return summary

