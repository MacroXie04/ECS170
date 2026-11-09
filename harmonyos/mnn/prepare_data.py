"""Export eight real PR #6 MNIST training batches for the MNN GPU probe.

Only input preprocessing runs on the host. All model training runs on the device.
These frozen augmented batches are repeated; this is a capability probe, not the
full Stage 3 protocol or an accuracy benchmark.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from local_code.stage_3_code.Dataset_Loader import make_train_loader
from local_code.stage_3_code.Experiment_Config import CNNConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    run = next(r for r in protocol['runs'] if r['id'] == 'mnist_kernel5_seed2026')
    cfg = CNNConfig.from_dict(run['config'])
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())['mnist']
    for filename in ('mnist_train_images.bin', 'mnist_train_labels.bin'):
        digest = hashlib.sha256((ROOT / 'data' / filename).read_bytes()).hexdigest()
        assert digest == manifest['files'][filename]['sha256'], filename
    n = manifest['optimization_samples']
    x = np.fromfile(ROOT / 'data/mnist_train_images.bin', np.uint8).reshape(n, 1, 28, 28)
    y = np.fromfile(ROOT / 'data/mnist_train_labels.bin', np.uint8).astype(np.int64)
    loader = make_train_loader(x, y, manifest['mean'], manifest['std'], cfg,
                              torch.Generator().manual_seed(run['seed']),
                              torch.Generator().manual_seed(run['seed'] + 1))
    xs, ys = [], []
    for images, labels in loader:
        xs.append(images.numpy())
        ys.append(labels.numpy())
        if len(xs) == 8:
            break
    images, labels = np.concatenate(xs), np.concatenate(ys)
    reference = np.fromfile(ROOT / 'data' / run['id'] / 'expected_batch.bin', np.float32)
    assert np.array_equal(images[:128].ravel(), reference), 'PR6 first-batch parity'
    schedule = np.fromfile(ROOT / 'data' / run['id'] / 'epoch_1.bin', np.int32).reshape(-1, 4)[:1024]
    assert len(np.unique(schedule[:, 0])) == 1024, 'Unique training examples'
    assert np.array_equal(labels, y[schedule[:, 0]]), 'PR6 sample order parity'
    images.astype('<f4').tofile(args.output / 'mnn_mnist_images.bin')
    np.eye(10, dtype='<f4')[labels].tofile(args.output / 'mnn_mnist_labels.bin')
    schedule.astype('<i4').tofile(args.output / 'sample_schedule.bin')
    metadata = dict(source_pr=protocol['source_pr'], source_commit=protocol['source_commit'],
                    dataset='instructor Stage 3 MNIST optimization split',
                    source_sha256=manifest['source_sha256'], unique_examples=1024,
                    batch_size=128, frozen_augmented_batches=8, seed=run['seed'],
                    first_batch_exact_pr6_match=True, sample_order_exact_pr6_match=True,
                    class_counts=np.bincount(labels, minlength=10).tolist(),
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in args.output.glob('*.bin')})
    (args.output / 'data_provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
