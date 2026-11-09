"""Compare physical-device first-step gradients against the MNN CPU control."""
import argparse
import json
from pathlib import Path

import numpy as np


def metrics(reference, observed):
    if not np.isfinite(reference).all():
        raise ValueError('CPU reference contains nonfinite values')
    nonfinite = int((~np.isfinite(observed)).sum())
    if nonfinite:
        return dict(max_abs_error=None, relative_l2_error=None, nonfinite=nonfinite)
    delta = observed - reference
    return dict(max_abs_error=float(np.max(np.abs(delta))),
                relative_l2_error=float(np.linalg.norm(delta) /
                                        max(np.linalg.norm(reference), 1e-30)),
                nonfinite=0)


def compare(directory, backend):
    cpu = json.loads((directory / 'cpu.json').read_text())
    result = dict(reference='MNN CPU on the same physical HarmonyOS device', backend=backend)
    for suffix in ('initial_weights', 'step1_gradients', 'step1_weights'):
        reference = np.fromfile(directory / f'cpu_{suffix}.bin', '<f4').astype(np.float64)
        observed = np.fromfile(directory / f'{backend}_{suffix}.bin', '<f4').astype(np.float64)
        if reference.size != observed.size or not reference.size:
            raise ValueError('Missing or mismatched numerical-check tensors')
        result[suffix] = metrics(reference, observed)
        if suffix == 'step1_gradients':
            rows, offset = [], 0
            for key in sorted(cpu['learnedTensorUpdates'], key=lambda key: int(key.split(':')[0])):
                shape = [int(value) for value in key.split(':')[1].split('x') if value]
                count = int(np.prod(shape))
                a, b = reference[offset:offset + count], observed[offset:offset + count]
                row = dict(tensor=key, **metrics(a, b),
                           reference_l2=float(np.linalg.norm(a)),
                           observed_l2=float(np.linalg.norm(b)) if np.isfinite(b).all() else None)
                row['passed'] = bool(not row['nonfinite'] and np.allclose(a, b, atol=1e-5, rtol=1e-3)
                                     and row['relative_l2_error'] <= 1e-4)
                rows.append(row)
                offset += count
            if offset != reference.size:
                raise ValueError('Learned-tensor metadata does not cover the gradient array')
            result['gradient_by_tensor'] = rows
    result['passed'] = (result['initial_weights']['max_abs_error'] == 0
                        and all(row['passed'] for row in result['gradient_by_tensor'])
                        and not result['step1_weights']['nonfinite'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--backends', nargs='+', default=['opencl_buffer'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require', nargs='*', default=[])
    args = parser.parse_args()
    results = {backend: compare(args.directory, backend) for backend in args.backends}
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    for backend, result in results.items():
        print(backend, 'PASS' if result['passed'] else 'FAIL', result['step1_gradients'])
    if any(not results[backend]['passed'] for backend in args.require):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
