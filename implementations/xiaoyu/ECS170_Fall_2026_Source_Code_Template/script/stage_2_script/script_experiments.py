import argparse
import csv
import json
from pathlib import Path

from script_mlp import PROJECT_ROOT, run_experiment

VARIANTS=[
    ('baseline', [128], 'adam', 'cross_entropy', 0.001),
    ('deeper', [256, 128], 'adam', 'cross_entropy', 0.001),
    ('sgd_lr_001', [128], 'sgd', 'cross_entropy', 0.01),
    ('mse_loss', [128], 'adam', 'mse', 0.001),
    ('lower_lr', [128], 'adam', 'cross_entropy', 0.0003),
]


def write_summary(destination, rows):
    if not rows:
        return
    with destination.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--train-limit', type=int, default=None)
    parser.add_argument('--test-limit', type=int, default=None)
    parser.add_argument('--only', nargs='*', default=None,
                        help='Experiment names; default: all five')
    args=parser.parse_args()
    if args.only is not None:
        unknown=set(args.only)-{name for name, *_ in VARIANTS}
        if unknown:
            parser.error(f'Unknown experiment names: {sorted(unknown)}')
    selected=[v for v in VARIANTS if args.only is None or v[0] in args.only]
    output=PROJECT_ROOT / 'result' / 'stage_2_result'
    output.mkdir(parents=True, exist_ok=True)
    results=[]
    for name, hidden, optimizer, loss, learning_rate in selected:
        params=argparse.Namespace(run_name=name, hidden_dims=hidden, optimizer=optimizer,
                                    loss=loss, learning_rate=learning_rate,
                                    epochs=args.epochs, batch_size=args.batch_size,
                                    device=args.device, seed=2, train_limit=args.train_limit,
                                    test_limit=args.test_limit)
        result_path=output / name
        config_path=result_path / 'run_config.json'
        metrics_path=result_path / 'metrics.json'
        reuse=False
        if config_path.is_file() and metrics_path.is_file():
            old_config=json.loads(config_path.read_text(encoding='utf-8'))
            same=all(old_config.get(k)==v for k,v in {
                'run_name': name, 'epochs': args.epochs, 'hidden_dims': hidden,
                'optimizer': optimizer, 'loss': loss, 'learning_rate': learning_rate,
                'batch_size': args.batch_size, 'seed': 2,
                'train_limit': args.train_limit, 'test_limit': args.test_limit,
            }.items())
            device_ok=args.device=='auto' or old_config.get('device')==args.device
            reuse=same and device_ok
        if reuse:
            print(f'Reusing completed experiment: {name}')
            scores=json.loads(metrics_path.read_text(encoding='utf-8'))
        else:
            scores=run_experiment(params)
        results.append({'experiment': name, 'hidden_dims': str(hidden),
                            'optimizer': optimizer, 'loss': loss, 'learning_rate': learning_rate,
                            **scores})
        write_summary(output / 'experiments_summary.csv', results)
    print(f'Comparison table saved to {output / "experiments_summary.csv"}')


if __name__=='__main__':
    main()
