"""Stage 3 command line: train, resume, evaluate, and summarize CNN experiments.

Run from anywhere, for example:
    python script/stage_3_script/run_stage3.py list
    python script/stage_3_script/run_stage3.py train --dataset mnist --config baseline
See the README for every command.
"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from local_code.stage_3_code import Experiment_Runner as runner  # noqa: E402
from local_code.stage_3_code.Experiment_Config import (DATASET_ORDER, FORMAL_SEEDS,  # noqa: E402
                                                       REGISTRY)
from local_code.stage_3_code.Preflight import run_preflight  # noqa: E402
from local_code.stage_3_code.Report_Artifacts import summarize  # noqa: E402


def add_common(parser):
    parser.add_argument('--device', default='auto',
                        help='auto (CUDA, then MPS, then CPU), cpu, cuda, cuda:N, or mps')
    parser.add_argument('--threads', type=int, help='CPU threads (default: PyTorch default)')
    parser.add_argument('--result-root', type=Path, default=runner.DEFAULT_ROOT)
    parser.add_argument('--data-dir', type=Path, default=runner.DEFAULT_DATA)


def add_run_identity(parser, seed_default=FORMAL_SEEDS[0]):
    parser.add_argument('--dataset', required=True, choices=sorted(REGISTRY))
    parser.add_argument('--config', default='baseline')
    parser.add_argument('--seed', type=int, default=seed_default)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)

    commands.add_parser('list', help='show every registered configuration')

    command = commands.add_parser('lock-protocol', help='fix configs and rules before testing')
    command.add_argument('--result-root', type=Path, default=runner.DEFAULT_ROOT)

    command = commands.add_parser('preflight', help='check data, models, and checkpoints')
    add_common(command)

    command = commands.add_parser('train', help='train one configuration (no test access)')
    add_run_identity(command)
    add_common(command)
    command.add_argument('--set', dest='overrides', action='append', default=[],
                         metavar='FIELD=VALUE', help='pilot only: change a config field')
    command.add_argument('--tag', help='pilot only: name a run under pilot_runs/')
    command.add_argument('--concurrent-jobs', type=int, default=1, help=argparse.SUPPRESS)

    command = commands.add_parser('resume', help='continue a run from its checkpoint')
    add_run_identity(command)
    add_common(command)
    command.add_argument('--checkpoint', type=Path, help='default: the run\'s last.pt')
    command.add_argument('--tag', help='the pilot tag, for a pilot run')
    command.add_argument('--concurrent-jobs', type=int, default=1, help=argparse.SUPPRESS)

    command = commands.add_parser('evaluate', help='score a checkpoint on the test set')
    add_run_identity(command)
    add_common(command)
    command.add_argument('--checkpoint', type=Path, help='default: the selected best.pt')

    command = commands.add_parser('run-all', help='train and evaluate the full plan')
    add_common(command)
    command.add_argument('--datasets', nargs='+', choices=sorted(REGISTRY))
    command.add_argument('--configs', nargs='+')
    command.add_argument('--seeds', nargs='+', type=int)
    command.add_argument('--jobs', type=int, default=1, help='concurrent training processes')

    command = commands.add_parser('verify', help='reload checkpoints and compare with results')
    add_common(command)

    command = commands.add_parser('summarize', help='rebuild tables and figures')
    command.add_argument('--result-root', type=Path, default=runner.DEFAULT_ROOT)
    return parser


def print_registry():
    for dataset in DATASET_ORDER:
        for name, config in REGISTRY[dataset].items():
            print(f'{dataset:6} {name:14} {config.changed_factor:12} {config.description}')


def method_for_diagram(config):
    return runner.METHOD_CLASSES[config.dataset](config, device='cpu')


def run_common(args):
    return {'device': args.device, 'threads': args.threads, 'root': args.result_root,
            'data_dir': args.data_dir}


def finish_run_all(args):
    failures = runner.run_all(Path(__file__).resolve(), datasets=args.datasets,
                              configs=args.configs, seeds=args.seeds, jobs=args.jobs,
                              **run_common(args))
    summarize(args.result_root, method_for_diagram)
    return 1 if failures else 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'list':
        print_registry()
    elif args.command == 'lock-protocol':
        print(runner.lock_protocol(args.result_root))
    elif args.command == 'preflight':
        report = run_preflight(**run_common(args))
        return 0 if report['all_passed'] else 1
    elif args.command == 'train':
        runner.train_run(args.dataset, args.config, args.seed, overrides=args.overrides,
                         tag=args.tag, concurrent_jobs=args.concurrent_jobs,
                         **run_common(args))
    elif args.command == 'resume':
        runner.resume_run(args.dataset, args.config, args.seed, checkpoint=args.checkpoint,
                          tag=args.tag, concurrent_jobs=args.concurrent_jobs,
                          **run_common(args))
    elif args.command == 'evaluate':
        report = runner.evaluate_run(args.dataset, args.config, args.seed,
                                     checkpoint=args.checkpoint, **run_common(args))
        print(json.dumps(report, indent=2))
        return 0 if report.get('passed', True) else 1
    elif args.command == 'run-all':
        return finish_run_all(args)
    elif args.command == 'verify':
        summary = runner.verify_all(**run_common(args))
        print(f"checked {summary['runs_checked']} runs, all passed: {summary['all_passed']}")
        return 0 if summary['all_passed'] else 1
    elif args.command == 'summarize':
        print(summarize(args.result_root, method_for_diagram))
    return 0


def dataset_entry_point(dataset):
    """Shared body of the per-dataset scripts: run that dataset's full panel."""
    parser = argparse.ArgumentParser(description=f'Run every {dataset} configuration')
    add_common(parser)
    parser.add_argument('--configs', nargs='+', help='default: all')
    parser.add_argument('--seeds', nargs='+', type=int, help='default: the formal seeds')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    args.datasets = [dataset]
    return finish_run_all(args)


if __name__ == '__main__':
    sys.exit(main())
