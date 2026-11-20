"""Configurable convolutional classifier with independent mini-batch training."""
import copy
import time
import torch
from torch import nn
from torch.nn import functional as F
from local_code.base_class.method import method


def augment(images):
    # Random crop and horizontal flip are used only for CIFAR training batches.
    n, _, h, w = images.shape
    padded = F.pad(images, (4, 4, 4, 4), mode='reflect')
    offsets = torch.randint(9, (n, 2), device=images.device)
    rows = torch.arange(h, device=images.device)[None, :, None] + offsets[:, 0, None, None]
    cols = torch.arange(w, device=images.device)[None, None, :] + offsets[:, 1, None, None]
    cropped = padded.permute(0, 2, 3, 1)[torch.arange(n, device=images.device)[:, None, None], rows, cols]
    cropped = cropped.permute(0, 3, 1, 2).contiguous()
    flip = torch.rand(n, device=images.device) < 0.5
    return torch.where(flip[:, None, None, None], cropped.flip(-1), cropped)


class Method_CNN(method, nn.Module):
    def __init__(self, input_shape, num_classes, config):
        method.__init__(self, 'Stage 3 CNN', 'Convolution, pooling and fully connected layers')
        nn.Module.__init__(self)
        self.config = dict(config)
        layers, channels = [], input_shape[0]
        for width in config['channels']:
            layers.append(nn.Conv2d(channels, width, config['kernel'], padding=config['kernel'] // 2))
            if config['batch_norm']:
                layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2) if config['pool'] == 'max' else nn.AvgPool2d(2))
            channels = width
        layers.append(nn.AdaptiveAvgPool2d(config['grid']))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(),
            nn.Linear(channels * config['grid'][0] * config['grid'][1], config['hidden']),
            nn.ReLU(), nn.Dropout(config['dropout']), nn.Linear(config['hidden'], num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))

    def criterion(self):
        if self.config['loss'] == 'cross_entropy':
            return F.cross_entropy
        if self.config['loss'] == 'squared_error':
            def squared_error(logits, labels):
                target = F.one_hot(labels, logits.shape[1]).float()
                return ((logits.softmax(1) - target) ** 2).sum(1).mean()
            return squared_error
        raise ValueError('Unknown loss')

    @torch.no_grad()
    def measure(self, X, y):
        self.eval()
        device = next(self.parameters()).device
        predictions, total = [], 0.0
        for start in range(0, len(y), self.config['batch_size']):
            x, target = X[start:start+self.config['batch_size']].to(device), y[start:start+self.config['batch_size']].to(device)
            logits = self(x)
            total += self.criterion()(logits, target).item() * len(target)
            predictions.append(logits.argmax(1).cpu())
        pred = torch.cat(predictions)
        return total / len(y), float((pred == y.cpu()).float().mean()), pred.numpy()

    def fit(self, X, y, validation=None, device='cpu'):
        self.to(device)
        cfg = self.config
        optimizer = torch.optim.Adam(self.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        history, best_state = [], None
        best_loss, best_epoch, stale = float('inf'), 0, 0
        started = time.perf_counter()
        for epoch in range(1, cfg['epochs'] + 1):
            self.train()
            order = torch.randperm(len(y))
            for indices in order.split(cfg['batch_size']):
                x, target = X[indices].to(device), y[indices].to(device)
                if cfg['augment']:
                    x = augment(x)
                optimizer.zero_grad(set_to_none=True)
                loss = self.criterion()(self(x), target)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite loss')
                loss.backward()
                optimizer.step()
            train_loss, train_acc, _ = self.measure(X, y)
            row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc)
            if validation is not None:
                val_loss, val_acc, _ = self.measure(*validation)
                row.update(val_loss=val_loss, val_accuracy=val_acc)
                if val_loss < best_loss - 1e-6:
                    best_loss, best_epoch, stale = val_loss, epoch, 0
                    best_state = copy.deepcopy(self.state_dict())
                else:
                    stale += 1
            else:
                best_epoch = epoch
            history.append(row)
            print(f"epoch={epoch:03d} train_acc={train_acc:.4f}" +
                  (f" val_acc={val_acc:.4f} val_loss={val_loss:.4f}" if validation is not None else ''), flush=True)
            if validation is not None and stale >= cfg['patience']:
                break
        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
        return history, best_epoch, time.perf_counter() - started

    def run(self, trainData, trainLabel, testData):
        self.fit(trainData, trainLabel)
        return self.test(testData)

    @torch.no_grad()
    def test(self, X):
        self.eval()
        device = next(self.parameters()).device
        return torch.cat([self(x.to(device)).argmax(1).cpu()
                          for x in X.split(self.config['batch_size'])]).numpy()
