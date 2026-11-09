"""Probe explicit training/inference backends on a physical HarmonyOS computer.

Success means the requested context ran, not proof of accelerator execution.
Use runtime logs to distinguish delegation, CPU fallback, and unsupported backends.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = 'com.example.mnistprobe'
REMOTE = '/data/storage/el2/base/haps/entry/files/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True)
    parser.add_argument('--run', default='mnist_kernel5_seed2026')
    parser.add_argument('--inference-model', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args()
    if ':' in args.target or not 1 <= args.steps <= 35:
        parser.error('Use a physical USB target and 1-35 steps')
    hdc = os.environ.get('HDC', str(Path.home() /
        'Applications/DevEco-Studio.app/Contents/sdk/default/openharmony/toolchains/hdc'))

    def command(*parts):
        result = subprocess.run([hdc, '-t', args.target, *parts],
                                capture_output=True, text=True, timeout=60)
        if result.returncode or '[Fail]' in result.stdout:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout

    def send(path, name):
        command('file', 'send', '-b', BUNDLE, str(path), REMOTE + name)

    def receive(name, path):
        command('file', 'recv', '-b', BUNDLE, REMOTE + name, str(path))

    device = command('shell', 'param get const.product.devicetype; '
                     'param get const.product.model; param get const.ohos.fullname')
    if '2in1' not in device:
        raise RuntimeError('Physical HarmonyOS computer required')
    run = next(r for r in json.loads((ROOT / 'protocol.json').read_text())['runs']
               if r['id'] == args.run)
    config = run['config']
    dataset = config['dataset']
    data = json.loads((ROOT / 'data/manifest.json').read_text())[dataset]
    channels, height, width = data['input_shape']
    out = args.output or ROOT / 'results' / f'backend_probe_{time.time_ns()}'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'device.txt').write_text(device)
    # Back up the previous launch request before changing any device job.
    receive('job.json', out / 'original-job.json')
    files = [(ROOT / 'models' / args.run / 'train.ms', 'backend_probe_train.ms'),
             (args.inference_model, 'backend_probe_infer.ms'),
             (ROOT / 'models' / args.run / 'variables.json', args.run + '_variables.json'),
             (ROOT / 'data' / args.run / 'epoch_1.bin', args.run + '_epoch_1.bin')]
    files += [(ROOT / 'data' / (dataset + suffix), dataset + suffix)
              for suffix in ('_train_images.bin', '_train_labels.bin')]
    for path, name in files:
        send(path, name)
    (out / 'input_hashes.json').write_text(json.dumps({name:
        hashlib.sha256(path.read_bytes()).hexdigest() for path, name in files}, indent=2))
    sources = [ROOT / 'tests/probe_backends.py',
               ROOT / 'Stage3App/entry/src/main/cpp/training_bridge.inc',
               ROOT / 'Stage3App/entry/src/main/ets/pages/NativeModel.ets',
               ROOT / 'Stage3App/entry/src/main/ets/pages/Index.ets']
    (out / 'source_hashes.json').write_text(json.dumps({str(path.relative_to(ROOT)):
        hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, indent=2))
    base = dict(modelId=args.run, dataset=dataset, height=height, width=width,
                inputChannels=channels, classes=40 if dataset == 'orl' else 10,
                batchSize=config['batch_size'], learningRate=config['learning_rate'],
                mean=data['mean'], std=data['std'],
                replicate=config['aug_padding_mode'] == 'replicate',
                action='probe_backend', probeSteps=args.steps)
    results = []
    try:
        for training in (True, False):
            for backend in ('cpu', 'gpu', 'gpu_cpu', 'nnrt_gpu', 'nnrt_cpu'):
                name = ('train_' if training else 'infer_') + backend
                ident = f'backend_probe_{name}_{time.time_ns()}'
                job = dict(base, id=ident, backend=backend, probeTrain=training,
                           probeModel='backend_probe_train.ms' if training else
                           'backend_probe_infer.ms')
                job_path = out / (name + '_job.json')
                job_path.write_text(json.dumps(job, indent=2))
                send(job_path, 'job.json')
                command('shell', 'aa force-stop ' + BUNDLE)
                command('shell', 'aa start -a EntryAbility -b ' + BUNDLE)
                result = {}
                deadline = time.monotonic() + 60
                destination = out / (name + '.json')
                while time.monotonic() < deadline:
                    time.sleep(1)
                    try:
                        receive(ident + '_backend.json', destination)
                        result = json.loads(destination.read_text())
                        if result.get('state') in ('completed', 'error'):
                            break
                    except RuntimeError:
                        pass
                    if not command('shell', 'pidof ' + BUNDLE).strip():
                        result['state'] = 'app_exited'
                        break
                else:
                    result['state'] = 'timeout'
                pid = command('shell', 'pidof ' + BUNDLE).strip()
                if pid:
                    (out / (name + '.log')).write_text(
                        command('shell', 'hilog -x -P ' + pid))
                result['probe_case'] = name
                destination.write_text(json.dumps(result, indent=2))
                results.append(result)
                (out / 'summary.json').write_text(json.dumps(results, indent=2))
                print(name, result.get('state'), result.get('steps'),
                      result.get('error'), flush=True)
    finally:
        command('shell', 'aa force-stop ' + BUNDLE)
        send(out / 'original-job.json', 'job.json')
    print('Evidence:', out, flush=True)


if __name__ == '__main__':
    main()
