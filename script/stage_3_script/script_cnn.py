"""Compare predefined CNNs and test a validation-selected model on each dataset."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_code.stage_3_code.Configurations import configurations
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Setting import Setting, code_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['baseline', 'compare', 'final'])
    parser.add_argument('--dataset', choices=['MNIST', 'ORL', 'CIFAR', 'all'], default='all')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/stage_3_data')
    parser.add_argument('--output', type=Path, default=ROOT / 'result/stage_3_result')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    for dataset in (['MNIST', 'ORL', 'CIFAR'] if args.dataset == 'all' else [args.dataset]):
        runner = Setting(Dataset_Loader(args.data, dataset), args.output / dataset, args.device)
        configs = configurations(dataset)
        if args.mode == 'final':
            result = runner.final()
            print(dataset, 'TEST', json.dumps(result['test']), flush=True)
            continue
        for name in (['baseline'] if args.mode == 'baseline' else configs):
            path = runner.output / name / 'record.json'
            if path.exists():
                record = json.loads(path.read_text())
                if record['config'] != configs[name] or record['source_sha256'] != code_hash():
                    raise ValueError('Existing run differs; use a new --output directory')
                if record['train_source'] != runner.dataset.load()['source']:
                    raise ValueError('Existing run uses different data')
                print('Reusing', dataset, name, flush=True)
            else:
                print('Starting', dataset, name, flush=True)
                runner.load_run_save_evaluate(name, configs[name])
        if args.mode == 'compare':
            print(json.dumps(runner.select(list(configs)), indent=2), flush=True)


if __name__ == '__main__':
    main()
