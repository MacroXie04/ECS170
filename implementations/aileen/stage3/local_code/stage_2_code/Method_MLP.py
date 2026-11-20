'''
Configurable MLP for multiclass classification (784 inputs -> 10 classes)
'''

import io
import time
import contextlib
import numpy as np
import torch
from torch import nn
from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Method_MLP(method, nn.Module):
    data = None

    # ---- hyper-parameters (set these from the script before calling run()) ----
    input_size = 784
    num_classes = 10
    hidden_sizes = [128, 64]       # one entry per hidden layer
    activation = 'relu'            # relu | leaky_relu | gelu | tanh
    dropout = 0.0
    batch_norm = False
    optimizer_name = 'adam'        # adam | adamw | sgd | rmsprop
    learning_rate = 1e-3
    weight_decay = 0.0
    loss_name = 'ce'               # ce | ce_ls (label smoothing 0.1) | mse | multimargin
    scheduler = None               # None | 'cosine' (cosine learning-rate decay over max_epoch)
    batch_size = 128
    max_epoch = 20
    seed = 2
    use_gpu = False                # set True to use CUDA / MPS if available 
    verbose = True

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.network = None
        self.loss_history = []     # mean training loss per epoch -> convergence curve
        self.val_history = [] 
        self.train_acc_history = []  # training accuracy per epoch (from the Evaluate_Metrics evaluator)    # validation accuracy per epoch (only if a val set is given)

    def build(self):
        acts = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh}
        layers, prev = [], self.input_size
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(acts[self.activation]())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = h
        # NOTE: no Softmax here. nn.CrossEntropyLoss applies log-softmax itself, so the model outputs raw logits.
        layers.append(nn.Linear(prev, self.num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def make_optimizer(self):
        name = self.optimizer_name.lower()
        p, lr, wd = self.parameters(), self.learning_rate, self.weight_decay
        if name == 'adam':
            return torch.optim.Adam(p, lr=lr, weight_decay=wd)
        if name == 'adamw':
            return torch.optim.AdamW(p, lr=lr, weight_decay=wd)
        if name == 'sgd':
            return torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=wd)
        if name == 'rmsprop':
            return torch.optim.RMSprop(p, lr=lr, weight_decay=wd)
        raise ValueError('unknown optimizer: ' + self.optimizer_name)

    def make_loss(self):
        if self.loss_name == 'ce':
            return nn.CrossEntropyLoss()
        if self.loss_name == 'ce_ls':
            return nn.CrossEntropyLoss(label_smoothing=0.1)
        if self.loss_name == 'multimargin':
            return nn.MultiMarginLoss()
        if self.loss_name == 'mse':
            mse = nn.MSELoss()
            # MSE between the softmax probabilities and the one-hot labels
            return lambda logits, y: mse(torch.softmax(logits, 1), nn.functional.one_hot(y, self.num_classes).float())
        raise ValueError('unknown loss: ' + self.loss_name)
    def predict(self, X):
        self.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                xb = torch.as_tensor(np.asarray(X[i:i + 2048]), dtype=torch.float32, device=self.device)
                preds.append(self.forward(xb).argmax(dim=1).cpu())
        return torch.cat(preds).numpy()

    def train_model(self, X, y, X_val=None, y_val=None):
        optimizer = self.make_optimizer()
        loss_function = self.make_loss()
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch)
                     if self.scheduler == 'cosine' else None)
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.device)
        n = X_t.shape[0]
        y_np = np.asarray(y)
        # for training accuracy investigation purpose (same pattern as the stage 1 template)
        accuracy_evaluator = Evaluate_Metrics('training evaluator', '')
        self.train_acc_history = []

        for epoch in range(self.max_epoch):
            self.train()
            perm = torch.randperm(n, device=self.device)
            total_loss = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                loss = loss_function(self.forward(X_t[idx]), y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)
            if scheduler is not None:
                scheduler.step()
            epoch_loss = total_loss / n
            self.loss_history.append(epoch_loss)

            # training accuracy through the evaluator (measured in eval mode: no dropout, batch-norm running stats)
            accuracy_evaluator.data = {'true_y': y_np, 'pred_y': self.predict(X)}
            with contextlib.redirect_stdout(io.StringIO()):   # evaluate() prints a line on every call; silence it
                train_acc = accuracy_evaluator.evaluate()['accuracy']
            self.train_acc_history.append(train_acc)

            msg = 'Epoch: %d  Loss: %.4f  Train Acc: %.4f' % (epoch + 1, epoch_loss, train_acc)
            if X_val is not None:
                val_acc = float((self.predict(X_val) == np.asarray(y_val)).mean())
                self.val_history.append(val_acc)
                msg += '  Val Acc: %.4f' % val_acc
            if self.verbose:
                print(msg)

    def run(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = get_device() if self.use_gpu else torch.device('cpu')
        self.build()
        self.to(self.device)
        print('method running on', self.device, '| trainable parameters:', self.num_parameters())

        self.loss_history, self.val_history = [], []
        val = self.data.get('val')

        print('--start training...')
        t0 = time.time()
        self.train_model(self.data['train']['X'], self.data['train']['y'],
                         val['X'] if val else None, val['y'] if val else None)
        self.method_training_time = time.time() - t0

        print('--start testing...')
        t0 = time.time()
        # when a validation set exists we are tuning, so "test" there means the validation set
        eval_split = val if (val and self.data.get('eval_on_val')) else self.data['test']
        pred_y = self.predict(eval_split['X'])
        self.method_testing_time = time.time() - t0
        return {'pred_y': pred_y, 'true_y': np.asarray(eval_split['y'])}
