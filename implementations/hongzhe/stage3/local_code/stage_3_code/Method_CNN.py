"""Configurable CNN shared by the three dataset-specific methods.

Follows the template's Method_MLP (method + nn.Module) with two deliberate
differences. Training is fit(), not train(), because overriding nn.Module.train
would break model.eval() and with it Dropout and BatchNorm. The network also
returns raw logits, since CrossEntropyLoss applies its own softmax.
"""
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from local_code.base_class.method import method
from local_code.stage_3_code.Experiment_Config import CHECKPOINT_RULE, CNNConfig
from local_code.stage_3_code.Runtime import atomic_write, get_rng_state, set_rng_state

CHECKPOINT_FORMAT = 1


class NonFiniteLossError(RuntimeError):
    """Training diverged; stop rather than record meaningless results."""


def is_better_epoch(candidate, incumbent):
    """CHECKPOINT_RULE. Epochs arrive in order, so an exact tie keeps the earliest."""
    if incumbent is None:
        return True
    if candidate['val_accuracy'] != incumbent['val_accuracy']:
        return candidate['val_accuracy'] > incumbent['val_accuracy']
    return candidate['val_loss'] < incumbent['val_loss']


def load_checkpoint(path):
    """weights_only rejects anything but tensors and plain Python containers."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != CHECKPOINT_FORMAT:
        raise ValueError(f'Unsupported checkpoint format in {path}')
    return checkpoint


class Method_CNN(method, nn.Module):
    data = None
    dataset_key = None
    input_shape = None
    num_classes = None

    def __init__(self, config, device='cpu', mName=None, mDescription=None):
        if config.dataset != self.dataset_key:
            raise ValueError(f'{type(self).__name__} cannot run {config.dataset} configs')
        method.__init__(self, mName or f'CNN {config.dataset}/{config.name}',
                        mDescription or config.description)
        nn.Module.__init__(self)
        self.config = config
        self.max_epoch = config.max_epoch
        self.learning_rate = config.learning_rate
        self.features = self._build_features()
        self.classifier = self._build_classifier(self._flattened_size())
        self._initialize_weights()
        self.device = torch.device(device)
        self.to(self.device)
        self.seed = None
        self.log = print
        self.optimizer = None
        self.scheduler = None
        self.history = []
        self.best = None
        self.start_epoch = 1
        self.train_seconds = 0.0

    def _build_features(self):
        cfg = self.config
        layers = []
        in_channels = self.input_shape[0]
        for out_channels in cfg.channels:
            for index in range(cfg.convs_per_stage):
                last_in_stage = index == cfg.convs_per_stage - 1
                stride = 2 if cfg.downsample == 'stride' and last_in_stage else 1
                # BatchNorm's shift makes a convolution bias redundant.
                layers.append(nn.Conv2d(in_channels, out_channels, cfg.kernel_size,
                                        stride=stride, padding=cfg.padding,
                                        bias=not cfg.batch_norm))
                if cfg.batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            if cfg.downsample == 'maxpool':
                layers.append(nn.MaxPool2d(2))
            elif cfg.downsample == 'avgpool':
                layers.append(nn.AvgPool2d(2))
            elif cfg.downsample != 'stride':
                raise ValueError(f'Unknown downsample: {cfg.downsample}')
        return nn.Sequential(*layers)

    def _flattened_size(self):
        with torch.no_grad():
            return self.features.eval()(torch.zeros(1, *self.input_shape)).numel()

    def _build_classifier(self, width):
        layers = [nn.Flatten()]
        for hidden in self.config.hidden_dims:
            layers += [nn.Linear(width, hidden), nn.ReLU(inplace=True)]
            if self.config.dropout:
                layers.append(nn.Dropout(self.config.dropout))
            width = hidden
        layers.append(nn.Linear(width, self.num_classes))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.classifier(self.features(x))

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def objective(self, logits, targets):
        """The configured training loss."""
        if self.config.loss == 'cross_entropy':
            return F.cross_entropy(logits, targets)
        if self.config.loss == 'mse_softmax':
            # Summed over classes: a mean would shrink gradients as K grows,
            # which would confound this comparison on 40-class ORL.
            one_hot = F.one_hot(targets, self.num_classes).to(logits.dtype)
            return (logits.softmax(dim=1) - one_hot).square().sum(dim=1).mean()
        raise ValueError(f'Unknown loss: {self.config.loss}')

    def configure_optimizer(self):
        cfg = self.config
        if cfg.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.learning_rate,
                                              weight_decay=cfg.weight_decay)
        elif cfg.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=cfg.learning_rate,
                                             momentum=cfg.momentum, nesterov=cfg.nesterov,
                                             weight_decay=cfg.weight_decay)
        else:
            raise ValueError(f'Unknown optimizer: {cfg.optimizer}')
        if cfg.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.max_epoch)
        elif cfg.scheduler != 'constant':
            raise ValueError(f'Unknown scheduler: {cfg.scheduler}')

    def _train_one_epoch(self, epoch):
        self.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for images, labels in self.data['train_loader']:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self(images)
            loss = self.objective(logits, labels)
            value = loss.item()
            if not np.isfinite(value):
                raise NonFiniteLossError(f'Non-finite loss {value} in epoch {epoch}')
            loss.backward()
            self.optimizer.step()
            total_loss += value * labels.numel()
            correct += (logits.detach().argmax(dim=1) == labels).sum().item()
            seen += labels.numel()
        return total_loss / seen, correct / seen

    @torch.no_grad()
    def predict(self, loader):
        """Eval-mode pass over a loader; returns logits, labels, and both losses."""
        self.eval()
        logits, labels = [], []
        for images, batch_labels in loader:
            logits.append(self(images.to(self.device)).cpu())
            labels.append(batch_labels)
        logits = torch.cat(logits)
        labels = torch.cat(labels)
        pred_y = logits.argmax(dim=1)
        return {'logits': logits,
                'true_y': labels.numpy(),
                'pred_y': pred_y.numpy(),
                'accuracy': (pred_y == labels).double().mean().item(),
                'cross_entropy': F.cross_entropy(logits, labels).item(),
                'objective': self.objective(logits, labels).item()}

    def fit(self):
        """Train to max_epoch on self.data, continuing from any restored state.

        The epoch budget is fixed, and only validation data informs the kept
        checkpoint. self.data['train_loader'] and 'val_loader' come from the Setting.
        """
        cfg = self.config
        if self.optimizer is None:
            self.configure_optimizer()
        for epoch in range(self.start_epoch, cfg.max_epoch + 1):
            started = time.perf_counter()
            learning_rate = self.optimizer.param_groups[0]['lr']
            train_loss, train_accuracy = self._train_one_epoch(epoch)
            validation = self.predict(self.data['val_loader'])
            if not np.isfinite(validation['cross_entropy']):
                raise NonFiniteLossError(f'Non-finite validation loss in epoch {epoch}')
            if self.scheduler is not None:
                self.scheduler.step()
            epoch_seconds = time.perf_counter() - started
            self.train_seconds += epoch_seconds
            row = {'epoch': epoch, 'learning_rate': learning_rate,
                   'train_loss': train_loss, 'train_accuracy': train_accuracy,
                   'val_loss': validation['cross_entropy'],
                   'val_objective': validation['objective'],
                   'val_accuracy': validation['accuracy'],
                   'epoch_seconds': epoch_seconds,
                   'elapsed_seconds': self.train_seconds}
            improved = is_better_epoch(row, self.best)
            if improved:
                self.best = {key: row[key] for key in ('epoch', 'val_accuracy', 'val_loss')}
            row['selected_so_far'] = self.best['epoch']
            self.history.append(row)
            self.log(f"epoch {epoch:03d}/{cfg.max_epoch} lr={learning_rate:.6f} "
                     f"train_loss={train_loss:.5f} train_acc={train_accuracy:.4f} "
                     f"val_ce={row['val_loss']:.5f} val_acc={row['val_accuracy']:.4f} "
                     f"best_epoch={self.best['epoch']} seconds={epoch_seconds:.2f}")
            self.data['on_epoch_end'](improved)
        return self.history

    def checkpoint_payload(self):
        """Everything needed to evaluate this model or to continue training it."""
        return {
            'format_version': CHECKPOINT_FORMAT,
            'config': self.config.to_dict(),
            'seed': self.seed,
            'epoch': self.history[-1]['epoch'] if self.history else 0,
            'model_state': self.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'rng_state': get_rng_state(self.device),
            'loader_generator_state': self.data['loader_generator'].get_state(),
            'augment_generator_state': self.data['augment_generator'].get_state(),
            'history': self.history,
            'best': self.best,
            'train_seconds': self.train_seconds,
            'normalization': self.data['normalization'],
            'selection_rule': CHECKPOINT_RULE,
        }

    def save_checkpoint(self, path):
        payload = self.checkpoint_payload()
        atomic_write(path, lambda target: torch.save(payload, target))

    def restore_training_state(self, checkpoint):
        """Continue exactly where the checkpoint's epoch ended."""
        saved = CNNConfig.from_dict(checkpoint['config'])
        if saved.fingerprint() != self.config.fingerprint():
            raise ValueError('Checkpoint was written by a different configuration')
        self.load_state_dict(checkpoint['model_state'])
        self.configure_optimizer()
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.history = list(checkpoint['history'])
        self.best = checkpoint['best']
        self.train_seconds = checkpoint['train_seconds']
        self.start_epoch = checkpoint['epoch'] + 1
        self.data['loader_generator'].set_state(checkpoint['loader_generator_state'])
        self.data['augment_generator'].set_state(checkpoint['augment_generator_state'])
        set_rng_state(checkpoint['rng_state'], self.device)

    def run(self):
        """Template entry point. Testing is Setting_Stage3.test_phase, which runs
        separately so that test data stays out of the training process."""
        self.log('method running...')
        self.log('--start training...')
        self.fit()
        return self.best
