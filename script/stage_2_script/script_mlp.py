"""Train a baseline, compare configurations, or evaluate the selected model."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Setting import Setting

BASE = dict(hidden=[128], dropout=0.0, loss='cross_entropy', optimizer='adam',
            lr=0.001, weight_decay=0.0, batch_size=128, epochs=25, patience=5, seed=42)
CONFIGS = {
    'baseline': dict(BASE),
    'depth2': dict(BASE, hidden=[256, 128]),
    'depth3': dict(BASE, hidden=[256, 128, 64]),
    'squared_error': dict(BASE, loss='squared_error'),
    'sgd': dict(BASE, optimizer='sgd', lr=0.05),
    'dropout': dict(BASE, hidden=[256, 128], dropout=0.2),
    'weight_decay': dict(BASE, hidden=[256, 128], weight_decay=0.0001),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['baseline', 'compare', 'final'])
    parser.add_argument('--data', type=Path, default=ROOT / 'data/stage_2_data')
    parser.add_argument('--output', type=Path, default=ROOT / 'result/stage_2_result')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    setting = Setting(Dataset_Loader(args.data), args.output, args.device)
    if args.mode == 'baseline':
        record = setting.load_run_save_evaluate('baseline', CONFIGS['baseline'])
        print(json.dumps({'best_epoch': record['best_epoch'],
                          'validation_accuracy': record['validation']['accuracy']}, indent=2))
    elif args.mode == 'compare':
        for name, config in CONFIGS.items():
            path = args.output / name / 'record.json'
            if path.exists():
                record = json.loads(path.read_text())
                if record['config'] != config:
                    raise ValueError(f'Existing {name} uses a different configuration')
                print('Reusing completed run:', name, flush=True)
            else:
                print('Starting independent run:', name, flush=True)
                setting.load_run_save_evaluate(name, config)
        print(json.dumps(setting.select(list(CONFIGS)), indent=2))
    elif args.mode == 'final':
        record = setting.final(args.output / 'selection.json')
        print(json.dumps(record['test'], indent=2))


if __name__ == '__main__':
    main()
