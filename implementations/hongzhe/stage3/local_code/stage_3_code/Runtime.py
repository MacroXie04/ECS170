"""Device selection, seeding, environment capture, and atomic file writes."""
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile

import numpy as np
import torch

REPRODUCIBILITY_NOTE = (
    'Seeds, data order, and augmentation are fixed and restored on resume. On CPU, '
    'a rerun matches only with the same PyTorch build, hardware, and thread count, '
    'because floating-point reduction order depends on them. CUDA runs use '
    'deterministic algorithms with TF32 disabled, but can still differ across GPU '
    'models, drivers, and library versions. MPS offers weaker determinism '
    'guarantees. Results from different devices are not expected to match exactly.'
)


def resolve_device(requested='auto'):
    """Return an available device: auto prefers CUDA, then MPS, then CPU."""
    mps_available = torch.backends.mps.is_available()
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('mps' if mps_available else 'cpu')
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    if device.type == 'mps' and not mps_available:
        raise RuntimeError('MPS was requested but is not available')
    return device


def configure_threads(threads):
    if threads:
        torch.set_num_threads(threads)


def seed_everything(seed):
    """Seed every generator and request deterministic kernels."""
    # cuBLAS reads this when it initializes, so it must precede any CUDA call.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def get_rng_state(device):
    state = {'torch_cpu': torch.get_rng_state()}
    if device.type == 'cuda':
        state['torch_cuda'] = torch.cuda.get_rng_state(device)
    elif device.type == 'mps':
        state['torch_mps'] = torch.mps.get_rng_state()
    return state


def set_rng_state(state, device):
    torch.set_rng_state(state['torch_cpu'])
    if device.type == 'cuda' and 'torch_cuda' in state:
        torch.cuda.set_rng_state(state['torch_cuda'], device)
    elif device.type == 'mps' and 'torch_mps' in state:
        torch.mps.set_rng_state(state['torch_mps'])


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _git(repo, *args):
    """Read-only git query; returns None outside a repository."""
    try:
        done = subprocess.run(['git', *args], cwd=repo, capture_output=True,
                              text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def source_fingerprint(repo):
    """Hash the Stage 3 sources, since results can predate the user's commit."""
    repo = Path(repo)
    paths = sorted([*(repo / 'local_code' / 'stage_3_code').glob('*.py'),
                    *(repo / 'script' / 'stage_3_script').glob('*.py')])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def capture_environment(device, repo):
    cuda = device.type == 'cuda'
    return {
        'device': str(device),
        'device_name': torch.cuda.get_device_name(device) if cuda else _cpu_model(),
        'cuda_runtime': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version() if cuda else None,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'torch_threads': torch.get_num_threads(),
        'cpu_model': _cpu_model(),
        'cpu_count': os.cpu_count(),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': {name: _package_version(name) for name in
                     ('torch', 'numpy', 'scikit-learn', 'matplotlib')},
        'git_commit': _git(repo, 'rev-parse', 'HEAD'),
        'git_worktree_dirty': bool(_git(repo, 'status', '--porcelain')),
        'stage3_source_sha256': source_fingerprint(repo),
        'reproducibility_note': REPRODUCIBILITY_NOTE,
    }


@contextmanager
def exclusive_run_lock(run_dir):
    """One process per run directory; the kernel releases this if the process dies."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / '.run.lock', 'w', encoding='utf-8') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'{run_dir} is in use by another process') from None
        yield


def sha256_file(path, chunk_size=1 << 22):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    os.close(handle)
    return path, Path(temporary)


def atomic_write(path, writer):
    """Call writer(temporary_path), then rename it into place.

    An interrupted write therefore never leaves a truncated result or checkpoint.
    """
    path, temporary = _atomic_path(path)
    try:
        writer(temporary)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_text(path, text):
    atomic_write(path, lambda target: target.write_text(text, encoding='utf-8'))


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
