"""Train on the fixed split, then test the selected checkpoint.

The two phases are separate so that all configurations can finish training
before any test image is read. Train phase loads the training split only.
"""
import csv
from pathlib import Path

import numpy as np
import torch

from local_code.base_class.setting import setting
from local_code.stage_3_code.Dataset_Loader import (compute_channel_stats, make_eval_loader,
                                                    make_train_loader,
                                                    make_train_validation_split,
                                                    split_summary)
from local_code.stage_3_code.Experiment_Config import SPLIT_SEED, CNNConfig
from local_code.stage_3_code.Method_CNN import load_checkpoint
from local_code.stage_3_code.Runtime import atomic_write, read_json, write_json


def attach_training_data(method, train, validation, normalization, seed, on_epoch_end):
    """Give a method its loaders; each random stream gets its own generator."""
    loader_generator = torch.Generator().manual_seed(seed)
    augment_generator = torch.Generator().manual_seed(seed + 1)
    mean, std = normalization['mean'], normalization['std']
    config = method.config
    method.seed = seed
    method.data = {
        'train_loader': make_train_loader(train['X'], train['y'], mean, std, config,
                                          loader_generator, augment_generator),
        'val_loader': make_eval_loader(validation['X'], validation['y'], mean, std,
                                       config.eval_batch_size),
        'loader_generator': loader_generator,
        'augment_generator': augment_generator,
        'normalization': normalization,
        'on_epoch_end': on_epoch_end,
    }


class Setting_Stage3(setting):
    def __init__(self, sName, sDescription, run_dir, seed, splits_dir=None):
        super().__init__(sName, sDescription)
        self.run_dir = Path(run_dir)
        self.seed = seed
        self.splits_dir = Path(splits_dir) if splits_dir else None

    @property
    def last_checkpoint(self):
        return self.run_dir / 'checkpoints' / 'last.pt'

    @property
    def selected_checkpoint(self):
        return self.run_dir / 'checkpoints' / 'best.pt'

    def _prepare_training_data(self):
        spec = self.dataset.spec
        data = self.dataset.load(splits=('train',))['train']
        images, labels = data['X'], data['y']
        train_idx, val_idx = make_train_validation_split(labels, spec.validation_size,
                                                         SPLIT_SEED)
        if set(train_idx.tolist()) & set(val_idx.tolist()):
            raise AssertionError('Optimization and validation sets overlap')
        # Statistics exclude validation images, like everything else optimized.
        mean, std = compute_channel_stats(images[train_idx])
        normalization = {'mean': mean, 'std': std,
                         'computed_on': 'optimization subset only'}
        summary = split_summary(labels, train_idx, val_idx, spec, SPLIT_SEED)
        summary['normalization'] = normalization
        write_json(self.run_dir / 'split.json', summary)
        if self.splits_dir:
            self._save_split_indices(train_idx, val_idx)
        attach_training_data(self.method,
                             {'X': images[train_idx], 'y': labels[train_idx]},
                             {'X': images[val_idx], 'y': labels[val_idx]},
                             normalization, self.seed, self._on_epoch_end)

    def _save_split_indices(self, train_idx, val_idx):
        path = self.splits_dir / f'{self.dataset.spec.key}_split_seed{SPLIT_SEED}.npz'

        def write(target):
            with open(target, 'wb') as handle:
                np.savez_compressed(handle, optimization_indices=train_idx,
                                    validation_indices=val_idx)
        if not path.exists():
            atomic_write(path, write)

    def _on_epoch_end(self, improved):
        # best.pt first: a stop between the writes then replays this epoch from
        # the older last.pt, which rewrites both files consistently.
        if improved:
            self.method.save_checkpoint(self.selected_checkpoint)
        self.method.save_checkpoint(self.last_checkpoint)
        self._write_history(self.method.history)
        self.update_status('training', epoch=self.method.history[-1]['epoch'],
                           train_seconds=self.method.train_seconds)

    def _write_history(self, history):
        write_json(self.run_dir / 'history.json', history)

        def write(target):
            with open(target, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
        atomic_write(self.run_dir / 'history.csv', write)

    def read_status(self):
        path = self.run_dir / 'status.json'
        return read_json(path) if path.exists() else {}

    def update_status(self, state, **details):
        status = self.read_status()
        status.update(details, state=state)
        write_json(self.run_dir / 'status.json', status)

    def _mark_trained(self, best, train_seconds, epoch):
        self.update_status('trained', selected_epoch=best['epoch'],
                           val_accuracy=best['val_accuracy'],
                           val_cross_entropy=best['val_loss'],
                           train_seconds=train_seconds, epoch=epoch)

    def finish_from_checkpoint(self, checkpoint):
        """Repair records after a stop that followed the final checkpoint."""
        self._write_history(checkpoint['history'])
        if self.read_status().get('state') not in ('trained', 'evaluated'):
            self._mark_trained(checkpoint['best'], checkpoint['train_seconds'],
                               checkpoint['epoch'])

    def train_phase(self, resume_checkpoint=None):
        """Train without touching the test set; returns the selection summary."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_training_data()
        if resume_checkpoint:
            self.method.restore_training_state(load_checkpoint(resume_checkpoint))
            self.method.log(f'resumed from {resume_checkpoint} at epoch '
                            f'{self.method.start_epoch}')
        self.update_status('training', epoch=self.method.start_epoch - 1)
        selected = self.method.run()
        self._mark_trained(selected, self.method.train_seconds,
                           self.method.history[-1]['epoch'])
        return selected

    def test_phase(self, checkpoint_path=None):
        """Reload a checkpoint and score it on the full test set.

        Normalization comes from the checkpoint, so nothing is computed from
        test images. Returns (metrics, predictions); saving is the caller's job.
        """
        checkpoint = load_checkpoint(checkpoint_path or self.selected_checkpoint)
        saved_config = CNNConfig.from_dict(checkpoint['config'])
        if saved_config.fingerprint() != self.method.config.fingerprint():
            raise ValueError('Checkpoint was written by a different configuration')
        self.method.load_state_dict(checkpoint['model_state'])
        spec = self.dataset.spec
        test = self.dataset.load(splits=('test',))['test']
        normalization = checkpoint['normalization']
        loader = make_eval_loader(test['X'], test['y'], normalization['mean'],
                                  normalization['std'], self.method.config.eval_batch_size)
        output = self.method.predict(loader)
        if output['true_y'].size != spec.test_size:
            raise AssertionError('Evaluation did not cover the full test set')
        self.evaluate.data = {'true_y': output['true_y'], 'pred_y': output['pred_y']}
        metrics = self.evaluate.evaluate()
        metrics.update(test_cross_entropy=output['cross_entropy'],
                       selected_epoch=checkpoint['epoch'], seed=checkpoint['seed'],
                       device=str(self.method.device))
        probabilities = output['logits'].softmax(dim=1).max(dim=1).values
        predictions = {
            'true_y': output['true_y'], 'pred_y': output['pred_y'],
            # The dataset's own numbering: 1..40 for ORL, unchanged elsewhere.
            'true_label_original': output['true_y'] + spec.label_offset,
            'pred_label_original': output['pred_y'] + spec.label_offset,
            'confidence': probabilities.numpy().astype(np.float32),
        }
        return metrics, predictions

    def load_run_save_evaluate(self):
        """Template entry point: both phases back to back for a single run."""
        self.train_phase()
        metrics, predictions = self.test_phase()
        self.result.data = {'metrics': metrics, 'predictions': predictions}
        self.result.save()
        self.update_status('evaluated', test_accuracy=metrics['accuracy'])
        return metrics
