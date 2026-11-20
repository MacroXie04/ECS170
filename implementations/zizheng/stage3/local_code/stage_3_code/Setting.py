"""Fixed training-only validation and a separately locked final refit."""
import hashlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from local_code.base_class.setting import setting
from .Evaluate import Evaluate
from .Result_Saver import Result_Saver
from .Method_CNN_MNIST import Method_CNN_MNIST
from .Method_CNN_ORL import Method_CNN_ORL
from .Method_CNN_CIFAR import Method_CNN_CIFAR

MODELS = dict(MNIST=Method_CNN_MNIST, ORL=Method_CNN_ORL, CIFAR=Method_CNN_CIFAR)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def code_hash():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / 'local_code/stage_3_code').glob('*.py'))
    paths += sorted((root / 'script/stage_3_script').glob('*.py'))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Setting(setting):
    def __init__(self, loader, output, device='cpu'):
        super().__init__('Stage 3 fixed validation', 'Instructor test partition is held out')
        self.dataset, self.output, self.device = loader, Path(output), device

    def metadata(self, data, config):
        root = Path(__file__).resolve().parents[2]
        return dict(dataset=self.dataset.name, config=config,
                    input_shape=list(data['X'].shape[1:]), classes=data['classes'].tolist(),
                    train_source=data['source'], source_sha256=code_hash(),
                    git_head=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                    preprocessing='Fixed divide_by_255; NCHW; first ORL channel; no fitted statistics',
                    environment=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__,
                                     platform=platform.platform(), device=self.device, threads=torch.get_num_threads()))

    def load_run_save_evaluate(self, name, config):
        saver = Result_Saver(self.output / name)
        saver.reserve()
        data = self.dataset.load()
        X, y = data['X'], data['y']
        train_idx, val_idx = train_test_split(np.arange(len(y)),
            test_size=40 if self.dataset.name == 'ORL' else 0.1,
            stratify=y.numpy(), random_state=42)
        np.savez_compressed(saver.folder / 'split.npz', train=train_idx, validation=val_idx)
        seed_everything(config['seed'])
        model = MODELS[self.dataset.name](num_classes=len(data['classes']), config=config)
        history, epoch, seconds = model.fit(X[train_idx], y[train_idx], (X[val_idx], y[val_idx]), self.device)
        evaluator = Evaluate(data['classes'])
        record = self.metadata(data, config)
        record.update(name=name, phase='validation', split_seed=42, best_epoch=epoch,
                      training_rows=len(train_idx), validation_rows=len(val_idx),
                      completed_epochs=len(history), training_seconds=seconds,
                      parameters=sum(p.numel() for p in model.parameters()))
        predictions = dict(train_true=y[train_idx].numpy(), train_pred=model.test(X[train_idx]),
                           validation_true=y[val_idx].numpy(), validation_pred=model.test(X[val_idx]))
        record.update(train=evaluator.score(predictions['train_true'], predictions['train_pred']),
                      validation=evaluator.score(predictions['validation_true'], predictions['validation_pred']))
        saver.data = record, model, history, predictions
        saver.save()
        return record

    def select(self, names):
        records = [json.loads((self.output / name / 'record.json').read_text()) for name in names]
        if any(r['phase'] != 'validation' or r['source_sha256'] != code_hash() for r in records):
            raise ValueError('Comparisons require matching validation code')
        if len({(r['train_source']['sha256'], r['split_seed']) for r in records}) != 1:
            raise ValueError('Comparisons require matching data and splits')
        best = sorted(records, key=lambda r: (-r['validation']['accuracy'], -r['validation']['macro']['f1'], r['name']))[0]
        selection = dict(rule='Max validation accuracy, then macro F1, then name; checkpoint by minimum validation loss',
                         candidates=names, selected=best['name'], config=dict(best['config'], epochs=best['best_epoch']),
                         dataset=best['dataset'], train_source=best['train_source'], source_sha256=code_hash(),
                         validation_accuracy=best['validation']['accuracy'])
        with (self.output / 'selection.json').open('x') as stream:
            json.dump(selection, stream, indent=2)
        return selection

    def final(self):
        selection = json.loads((self.output / 'selection.json').read_text())
        data = self.dataset.load()
        if selection['source_sha256'] != code_hash() or selection['train_source'] != data['source']:
            raise ValueError('Data or source changed; use a new output directory')
        saver = Result_Saver(self.output / 'final')
        saver.reserve()
        saver.write_json('locked_selection.json', selection)
        cfg = selection['config']
        seed_everything(cfg['seed'])
        model = MODELS[self.dataset.name](num_classes=len(data['classes']), config=cfg)
        history, epoch, seconds = model.fit(data['X'], data['y'], device=self.device)
        test = self.dataset.load('test', data['classes'])
        predictions = dict(train_true=data['y'].numpy(), train_pred=model.test(data['X']),
                           test_true=test['y'].numpy(), test_pred=model.test(test['X']))
        evaluator = Evaluate(data['classes'])
        record = self.metadata(data, cfg)
        record.update(name='final', phase='final_test', selected=selection['selected'], best_epoch=epoch,
                      training_seconds=seconds, training_rows=len(data['y']), test_rows=len(test['y']),
                      test_source=test['source'], parameters=sum(p.numel() for p in model.parameters()),
                      train=evaluator.score(predictions['train_true'], predictions['train_pred']),
                      test=evaluator.score(predictions['test_true'], predictions['test_pred']))
        saver.data = record, model, history, predictions
        saver.save()
        return record
