"""Run the optional MNN app on a physical HarmonyOS computer and retain evidence."""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backends', nargs='+', choices=['cpu', 'opencl', 'opencl_buffer', 'vulkan', 'vulkan_batch'], default=['cpu', 'opencl_buffer'])
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--timeout', type=int, default=240)
    p.add_argument('--wait-unlock', type=int, default=0, help='Seconds to wait for a user to unlock the device')
    p.add_argument('--install', action='store_true')
    p.add_argument('--reload-model', help='Saved run ID; verifies inference in this fresh app process')
    args = p.parse_args()
    if ':' in args.target or not 1 <= args.steps <= 800:
        p.error('Use a physical USB target and 1-800 steps')
    hdc = os.environ.get('HDC', str(Path.home() / 'Applications/DevEco-Studio.app/Contents/sdk/default/openharmony/toolchains/hdc'))

    def cmd(*parts):
        r = subprocess.run([hdc, '-t', args.target, *parts], capture_output=True, text=True, timeout=60)
        if r.returncode or '[Fail]' in r.stdout or 'error: failed' in r.stdout.lower():
            raise RuntimeError(r.stdout + r.stderr)
        return r.stdout

    def send(path, name):
        cmd('file', 'send', '-b', BUNDLE, str(path), REMOTE + name)

    def receive(name, path):
        cmd('file', 'recv', '-b', BUNDLE, REMOTE + name, str(path))

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    device = cmd('shell', 'param get const.product.devicetype; param get const.product.model; param get const.ohos.fullname')
    if '2in1' not in device:
        raise RuntimeError('Physical HarmonyOS computer required')
    (out / 'device.txt').write_text(device)
    receive('job.json', out / 'original-job.json')
    if args.install:
        hap = ROOT / 'Stage3App/entry/build/default/outputs/default/entry-default-signed.hap'
        (out / 'hap_sha256.txt').write_text(hashlib.sha256(hap.read_bytes()).hexdigest())
        print(cmd('install', '-r', str(hap)), flush=True)
    for name in ('mnn_mnist_images.bin', 'mnn_mnist_labels.bin'):
        send(args.data / name, name)
    (out / 'data_provenance.json').write_bytes((args.data / 'data_provenance.json').read_bytes())
    sources = [ROOT / 'Stage3App/entry/src/main/cpp/mnn_probe.inc', ROOT / 'mnn/run_probe.py', ROOT / 'mnn/build_probe.sh']
    (out / 'source_hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}, indent=2))
    results = []
    try:
        for backend in args.backends:
            ident = f'mnn_{backend}_{time.time_ns()}'
            job = dict(id=ident, modelId='mnist_kernel5_seed2026', dataset='mnist',
                       action='mnn_reload' if args.reload_model else 'mnn_probe',
                       probeModel=args.reload_model or '', backend=backend, probeSteps=args.steps)
            path = out / (backend + '_job.json')
            path.write_text(json.dumps(job, indent=2))
            send(path, 'job.json')
            cmd('shell', 'aa force-stop ' + BUNDLE)
            unlock_deadline = time.monotonic() + args.wait_unlock
            waiting = False
            while True:
                try:
                    launch = cmd('shell', 'aa start -a EntryAbility -b ' + BUNDLE)
                    break
                except RuntimeError as ex:
                    (out / 'launch_error.txt').write_text(str(ex))
                    if '10106102' not in str(ex) or time.monotonic() >= unlock_deadline:
                        raise
                    if not waiting:
                        print('Waiting for the user to unlock the HarmonyOS computer.', flush=True)
                        waiting = True
                    time.sleep(5)
            (out / (backend + '_launch.txt')).write_text(launch)
            result, last, pid = {}, None, ''
            deadline = time.monotonic() + args.timeout
            dest = out / (backend + '.json')
            while time.monotonic() < deadline:
                time.sleep(1)
                try:
                    receive(ident + '_mnn.json', dest)
                    result = json.loads(dest.read_text())
                    progress = (result.get('state'), result.get('steps'))
                    if progress != last:
                        print(backend, progress, result.get('error'), flush=True)
                        last = progress
                    if result.get('state') in ('completed', 'error'):
                        break
                except (RuntimeError, json.JSONDecodeError):
                    pass
                current_pid = cmd('shell', 'pidof ' + BUNDLE).strip()
                if current_pid:
                    pid = current_pid
                else:
                    result['state'] = 'app_exited'
                    break
            else:
                result['state'] = 'timeout'
            pid = cmd('shell', 'pidof ' + BUNDLE).strip() or pid
            result['device_pid'] = pid
            if pid:
                (out / (backend + '.log')).write_text(cmd('shell', 'hilog -x -P ' + pid))
            if result.get('state') == 'completed' and not args.reload_model:
                for suffix in ('_model.mnn', '_params.mnn', '_expected.bin'):
                    receive(ident + suffix, out / (backend + suffix))
            if not args.reload_model:
                for suffix in ('_initial_weights.bin', '_step1_gradients.bin', '_step1_weights.bin'):
                    try:
                        receive(ident + suffix, out / (backend + suffix))
                    except RuntimeError:
                        pass
            result['probe_case'] = backend
            dest.write_text(json.dumps(result, indent=2))
            results.append(result)
            (out / 'summary.json').write_text(json.dumps(results, indent=2))
            print(backend, result.get('state'), result.get('steps'), result.get('error'), flush=True)
    finally:
        cmd('shell', 'aa force-stop ' + BUNDLE)
        send(out / 'original-job.json', 'job.json')
    print('Evidence:', out, flush=True)


if __name__ == '__main__':
    main()
