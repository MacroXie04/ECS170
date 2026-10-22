"""Recreate the instructor CSVs from checksum-verified public MNIST archives."""

import argparse
import gzip
import hashlib
from pathlib import Path
import struct
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np


# The HTTPS mirror used by torchvision.datasets.MNIST:
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py
BASE_URL = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
ARCHIVES = {
    'train-images-idx3-ubyte.gz': '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609',
    'train-labels-idx1-ubyte.gz': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c',
    't10k-images-idx3-ubyte.gz': '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6',
    't10k-labels-idx1-ubyte.gz': 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6',
}
# Verified byte-for-byte against the instructor's original CSV files.
CSV_SHA256 = {
    'train': 'fb60bc58af4dac3554e394af262b3184479833d3cc540ff8783f274b73492d5d',
    'test': '51c292478d94ec3a01461bdfa82eb0885d262eb09e615679b2d69dedb6ad09e7',
}


def download(name):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(BASE_URL + name, timeout=60) as response:
                payload = response.read()
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    if hashlib.sha256(payload).hexdigest() != ARCHIVES[name]:
        raise ValueError(f'Archive checksum mismatch: {name}')
    return gzip.decompress(payload)


def prepare(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, prefix, count in [('train', 'train', 60000), ('test', 't10k', 10000)]:
        target = output_dir / f'{split}.csv'
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != CSV_SHA256[split]:
                raise ValueError(f'Existing file does not match instructor data: {target}')
            print(f'{split}: existing CSV checksum verified', flush=True)
            continue
        images = download(f'{prefix}-images-idx3-ubyte.gz')
        labels = download(f'{prefix}-labels-idx1-ubyte.gz')
        if struct.unpack('>4I', images[:16]) != (2051, count, 28, 28):
            raise ValueError(f'Unexpected {split} image header')
        if struct.unpack('>2I', labels[:8]) != (2049, count):
            raise ValueError(f'Unexpected {split} label header')
        pixels = np.frombuffer(images, dtype=np.uint8, offset=16).reshape(count, 784)
        classes = np.frombuffer(labels, dtype=np.uint8, offset=8)
        rows = np.column_stack((classes, pixels))
        with tempfile.TemporaryDirectory(dir=output_dir) as temp:
            candidate = Path(temp) / target.name
            np.savetxt(candidate, rows, delimiter=',', fmt='%d')
            if hashlib.sha256(candidate.read_bytes()).hexdigest() != CSV_SHA256[split]:
                raise ValueError(f'Generated {split} CSV does not match instructor data')
            candidate.replace(target)
        print(f'{split}: {count:,} rows, instructor CSV checksum verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
