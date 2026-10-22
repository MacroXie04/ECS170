"""A template-compatible MLP with mini-batch training and explicit modes."""
import copy
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from local_code.base_class.method import method


class Method_MLP(method, nn.Module):
    def __init__(self, input_dim, num_classes, config):
        method.__init__(self, 'Stage 2 MLP', 'Configurable fully connected classifier')
        nn.Module.__init__(self)
        self.config = dict(config)
        layers = []
        previous = input_dim
        for width in config['hidden']:
            layers += [nn.Linear(previous, width), nn.ReLU()]
            if config['dropout']:
                layers.append(nn.Dropout(config['dropout']))
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # CrossEntropyLoss combines log-softmax and NLL internally.
        return self.network(x)

    def criterion(self):
        if self.config['loss'] == 'cross_entropy':
            return nn.CrossEntropyLoss()
        if self.config['loss'] == 'squared_error':
            def squared_error(logits, labels):
                target = nn.functional.one_hot(labels, logits.shape[1]).float()
                return ((logits.softmax(dim=1) - target) ** 2).sum(dim=1).mean()
            return squared_error
        raise ValueError('Unknown loss')

    @torch.no_grad()
    def measure(self, X, y, criterion):
        self.eval()
        device = next(self.parameters()).device
        total_loss, correct = 0.0, 0
        for start in range(0, len(y), 1024):
            x = torch.from_numpy(X[start:start + 1024]).to(device)
            target = torch.from_numpy(y[start:start + 1024]).to(device)
            logits = self(x)
            total_loss += criterion(logits, target).item() * len(target)
            correct += (logits.argmax(1) == target).sum().item()
        return total_loss / len(y), correct / len(y)

    def fit(self, X, y, validation=None, device='cpu'):
        self.to(device)
        cfg = self.config
        if cfg['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=cfg['lr'],
                                         weight_decay=cfg['weight_decay'])
        elif cfg['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=cfg['lr'], momentum=0.9,
                                        weight_decay=cfg['weight_decay'])
        else:
            raise ValueError('Unknown optimizer')
        criterion = self.criterion()
        generator = torch.Generator().manual_seed(cfg['seed'])
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                            batch_size=cfg['batch_size'], shuffle=True,
                            generator=generator, num_workers=0)
        history, best_state = [], None
        best_loss, best_epoch, stale = float('inf'), 0, 0
        started = time.perf_counter()
        for epoch in range(1, cfg['epochs'] + 1):
            self.train()
            for x, target in loader:
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self(x), target)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss')
                loss.backward()
                optimizer.step()
            # Use the same end-of-epoch weights for train and validation metrics.
            train_loss, train_accuracy = self.measure(X, y, criterion)
            row = {'epoch': epoch, 'train_loss': train_loss,
                   'train_accuracy': train_accuracy}
            if validation is not None:
                val_loss, val_accuracy = self.measure(*validation, criterion)
                row.update(val_loss=val_loss, val_accuracy=val_accuracy)
                if val_loss < best_loss - 1e-6:
                    best_loss, best_epoch, stale = val_loss, epoch, 0
                    best_state = copy.deepcopy(self.state_dict())
                else:
                    stale += 1
            else:
                best_epoch = epoch
            history.append(row)
            print(f"epoch={epoch:02d} train_acc={train_accuracy:.4f}" +
                  (f" val_acc={val_accuracy:.4f} val_loss={val_loss:.4f}"
                   if validation is not None else ''), flush=True)
            if validation is not None and stale >= cfg['patience']:
                break
        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
        return history, best_epoch, time.perf_counter() - started

    @torch.no_grad()
    def test(self, X):
        self.eval()
        device = next(self.parameters()).device
        predictions = []
        for start in range(0, len(X), 1024):
            logits = self(torch.from_numpy(X[start:start + 1024]).to(device))
            predictions.append(logits.argmax(1).cpu())
        return torch.cat(predictions).numpy()

    def run(self):
        return self.fit(self.data['train']['X'], self.data['train']['y'],
                        self.data.get('validation'))
