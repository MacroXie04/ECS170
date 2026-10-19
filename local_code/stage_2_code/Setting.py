"""Fixed training-only validation, independent runs, and a separate final test."""
import hashlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path
import numpy as np
import sklearn
import torch
from sklearn.model_selection import train_test_split
from local_code.base_class.setting import setting
from .Method_MLP import Method_MLP
from .Evaluate import Evaluate
from .Result_Saver import Result_Saver


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def code_hash():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / 'local_code/stage_2_code').glob('*.py'))
    paths += sorted((root / 'script/stage_2_script').glob('*.py'))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Setting(setting):
    def __init__(self, loader, output, device='cpu', split_seed=42):
        super().__init__('Stage 2 fixed validation', 'No cross-validation')
        self.dataset, self.output = loader, Path(output)
        self.device, self.split_seed = device, split_seed

    def metadata(self, data, config):
        root = Path(__file__).resolve().parents[2]
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                           text=True).strip()
        return {'config': config, 'input_dim': data['X'].shape[1],
                'classes': data['classes'].tolist(), 'train_source': data['source'],
                'preprocessing': 'fixed divide_by_255; no fitted parameters',
                'environment': {'python': sys.version, 'torch': torch.__version__,
                                'numpy': np.__version__, 'sklearn': sklearn.__version__,
                                'platform': platform.platform(), 'device': self.device,
                                'threads': torch.get_num_threads()},
                'git_head': revision, 'source_sha256': code_hash()}

    def load_run_save_evaluate(self, name, config):
        saver = Result_Saver(self.output / name)
        saver.reserve()
        data = self.dataset.load()
        train_idx, val_idx = train_test_split(np.arange(len(data['y'])), test_size=0.1,
                                             stratify=data['y'], random_state=self.split_seed)
        np.savez_compressed(saver.folder / 'split.npz', train=train_idx, validation=val_idx)
        X, y = data['X'], data['y']
        seed_everything(config['seed'])
        model = Method_MLP(X.shape[1], len(data['classes']), config)
        evaluator = Evaluate(data['classes'])
        initial_loss, initial_accuracy = model.measure(X[val_idx], y[val_idx], model.criterion())
        history, epoch, seconds = model.fit(X[train_idx], y[train_idx],
                                            (X[val_idx], y[val_idx]), self.device)
        train_pred, val_pred = model.test(X[train_idx]), model.test(X[val_idx])
        record = self.metadata(data, config)
        record.update(name=name, phase='validation', split_seed=self.split_seed,
                      training_rows=len(train_idx), validation_rows=len(val_idx),
                      initial_validation={'loss': initial_loss, 'accuracy': initial_accuracy},
                      best_epoch=epoch, completed_epochs=len(history), training_seconds=seconds,
                      train=evaluator.score(y[train_idx], train_pred),
                      validation=evaluator.score(y[val_idx], val_pred))
        saver.data = record, model, history, dict(train_true=y[train_idx], train_pred=train_pred,
                                                validation_true=y[val_idx], validation_pred=val_pred)
        saver.save()
        return record

    def select(self, names):
        records = [json.loads((self.output / name / 'record.json').read_text()) for name in names]
        if any(r['phase'] != 'validation' for r in records):
            raise ValueError('Selection accepts validation runs only')
        if len({(r['train_source']['sha256'], r['split_seed']) for r in records}) != 1:
            raise ValueError('Selection requires identical data and split')
        best = sorted(records, key=lambda r: (-r['validation']['accuracy'],
                      -r['validation']['macro']['f1'], r['name']))[0]
        selection = {'rule': 'max validation accuracy, then macro F1, then name',
                     'candidates': names, 'selected': best['name'],
                     'config': dict(best['config'], epochs=best['best_epoch']),
                     'train_source': best['train_source'], 'classes': best['classes'],
                     'source_sha256': best['source_sha256'],
                     'validation_accuracy': best['validation']['accuracy']}
        target = self.output / 'selection.json'
        with target.open('x') as stream:
            json.dump(selection, stream, indent=2)
        return selection

    def final(self, selection_path):
        selection = json.loads(Path(selection_path).read_text())
        saver = Result_Saver(self.output / 'final')
        saver.reserve()
        data = self.dataset.load()
        if data['source'] != selection['train_source'] or code_hash() != selection['source_sha256']:
            raise ValueError('Training data or experiment source changed after selection')
        config = selection['config']
        # Freeze the selection in final/ BEFORE training or opening test.csv.
        saver.write_json('locked_selection.json', selection)
        seed_everything(config['seed'])
        model = Method_MLP(data['X'].shape[1], len(data['classes']), config)
        history, epoch, seconds = model.fit(data['X'], data['y'], device=self.device)
        train_pred = model.test(data['X'])
        test = self.dataset.load_file('test.csv', data['classes'])
        if test['X'].shape[1] != data['X'].shape[1]:
            raise ValueError('Train/test feature dimensions differ')
        test_pred = model.test(test['X'])
        evaluator = Evaluate(data['classes'])
        record = self.metadata(data, config)
        record.update(name='final', phase='final_test', selected=selection['selected'],
                      test_source=test['source'], best_epoch=epoch, training_seconds=seconds,
                      training_rows=len(data['y']), test_rows=len(test['y']),
                      train=evaluator.score(data['y'], train_pred),
                      test=evaluator.score(test['y'], test_pred))
        saver.data = record, model, history, dict(train_true=data['y'], train_pred=train_pred,
                                                test_true=test['y'], test_pred=test_pred)
        saver.save()
        return record
