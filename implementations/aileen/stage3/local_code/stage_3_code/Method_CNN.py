'''
Configurable CNN for image classification (shared by the MNIST / ORL / CIFAR-10 methods)

    [Conv2d -> (BatchNorm) -> activation -> (Pool)] x len(conv_channels)
        -> Flatten -> Linear(hidden_dim) -> activation -> (Dropout) -> Linear(num_classes)

The network outputs raw logits (no Softmax): CrossEntropyLoss applies log-softmax itself.
'''
import io
import time
import contextlib
import numpy as np
import torch
from torch import nn
from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Method_CNN(method, nn.Module):
    data = None

    # ---- hyper-parameters (set from the script before calling run()) ----
    in_channels = 1
    image_size = (28, 28)          # (H, W)
    num_classes = 10
    conv_channels = [32, 64]       # one entry per conv layer (depth = len)
    kernel_size = 3
    padding = 1
    stride = 1
    pool = 'max'                   # max | avg | None
    pool_size = 2
    hidden_dim = 128               # fully connected layer before the output (0 = none)
    activation = 'relu'            # relu | leaky_relu | gelu | tanh
    dropout = 0.0
    batch_norm = True
    optimizer_name = 'adam'        # adam | adamw | sgd | rmsprop
    learning_rate = 1e-3
    weight_decay = 0.0
    loss_name = 'ce'               # ce | ce_ls | mse | multimargin
    scheduler = None               # None | 'cosine'
    batch_size = 128
    max_epoch = 10
    seed = 2
    augment = False                # random horizontal flip + random crop (padding 4) on the training batches
    use_gpu = False                # CUDA / MPS if available
    amp = True                     # mixed precision, only used on CUDA
    track_test = True              # record test accuracy each epoch for the learning curve (never used to choose anything)
    verbose = True

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.network = None
        self.loss_history = []        # mean training loss per epoch
        self.train_acc_history = []   # training accuracy per epoch (Evaluate_Metrics, eval mode)
        self.test_acc_history = []    # test accuracy per epoch (learning curve only)

    # ------------------------------------------------------------------ model
    def build(self):
        acts = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh}
        layers, prev = [], self.in_channels
        for c in self.conv_channels:
            layers.append(nn.Conv2d(prev, c, self.kernel_size, stride=self.stride, padding=self.padding))
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(c))
            layers.append(acts[self.activation]())
            if self.pool == 'max':
                layers.append(nn.MaxPool2d(self.pool_size))
            elif self.pool == 'avg':
                layers.append(nn.AvgPool2d(self.pool_size))
            prev = c
        features = nn.Sequential(*layers)
        with torch.no_grad():                         # find the flattened size with a dummy image
            features.eval()
            flat = features(torch.zeros(1, self.in_channels, *self.image_size)).numel()
        head = [nn.Flatten()]
        if self.hidden_dim > 0:
            head += [nn.Linear(flat, self.hidden_dim), acts[self.activation]()]
            if self.dropout > 0:
                head.append(nn.Dropout(self.dropout))
            head.append(nn.Linear(self.hidden_dim, self.num_classes))
        else:
            head.append(nn.Linear(flat, self.num_classes))
        self.network = nn.Sequential(features, *head)

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
            return lambda logits, y: mse(torch.softmax(logits, 1), nn.functional.one_hot(y, self.num_classes).float())
        raise ValueError('unknown loss: ' + self.loss_name)

    # ------------------------------------------------------------------ data helpers
    def _augment(self, xb):
        B, C, H, W = xb.shape
        flip = torch.rand(B, device=xb.device) < 0.5
        xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
        pad = 4
        xp = nn.functional.pad(xb, (pad, pad, pad, pad), mode='reflect')
        oy = torch.randint(0, 2 * pad + 1, (B,), device=xb.device)
        ox = torch.randint(0, 2 * pad + 1, (B,), device=xb.device)
        iy = (oy[:, None] + torch.arange(H, device=xb.device))[:, :, None]
        ix = (ox[:, None] + torch.arange(W, device=xb.device))[:, None, :]
        bi = torch.arange(B, device=xb.device)[:, None, None]
        return xp.permute(0, 2, 3, 1)[bi, iy, ix].permute(0, 3, 1, 2)

    def predict(self, X):
        self.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 1000):
                xb = torch.as_tensor(np.asarray(X[i:i + 1000]), dtype=torch.float32, device=self.device)
                preds.append(self.forward(xb).argmax(dim=1).cpu())
        return torch.cat(preds).numpy()

    # ------------------------------------------------------------------ training
    def train_model(self, X, y, X_test=None, y_test=None):
        optimizer = self.make_optimizer()
        loss_function = self.make_loss()
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch)
                     if self.scheduler == 'cosine' else None)
        use_amp = self.amp and self.device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)   # whole set lives on the GPU
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.device)
        n = X_t.shape[0]
        y_np = np.asarray(y)
        accuracy_evaluator = Evaluate_Metrics('training evaluator', '')
        self.loss_history, self.train_acc_history, self.test_acc_history = [], [], []

        for epoch in range(self.max_epoch):
            self.train()
            perm = torch.randperm(n, device=self.device)
            total_loss = torch.zeros((), device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:                      # BatchNorm cannot train on a single sample
                    continue
                xb = X_t[idx]
                if self.augment:
                    xb = self._augment(xb)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                    logits = self.forward(xb)
                loss = loss_function(logits.float(), y_t[idx])
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.detach() * len(idx)
            if scheduler is not None:
                scheduler.step()
            epoch_loss = total_loss.item() / n
            self.loss_history.append(epoch_loss)

            accuracy_evaluator.data = {'true_y': y_np, 'pred_y': self.predict(X)}
            with contextlib.redirect_stdout(io.StringIO()):
                train_acc = accuracy_evaluator.evaluate()['accuracy']
            self.train_acc_history.append(train_acc)

            msg = 'Epoch: %d  Loss: %.4f  Train Acc: %.4f' % (epoch + 1, epoch_loss, train_acc)
            if self.track_test and X_test is not None:
                test_acc = float((self.predict(X_test) == np.asarray(y_test)).mean())
                self.test_acc_history.append(test_acc)
                msg += '  Test Acc: %.4f' % test_acc
            if self.verbose:
                print(msg)

    def run(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = get_device() if self.use_gpu else torch.device('cpu')
        self.build()
        self.to(self.device)
        print('method running on', self.device, '| trainable parameters:', self.num_parameters())

        test = self.data['test']
        print('--start training...')
        t0 = time.time()
        self.train_model(self.data['train']['X'], self.data['train']['y'], test['X'], test['y'])
        self.method_training_time = time.time() - t0

        print('--start testing...')
        t0 = time.time()
        pred_y = self.predict(test['X'])          # final-epoch model, no best-epoch picking
        self.method_testing_time = time.time() - t0
        return {'pred_y': pred_y, 'true_y': np.asarray(test['y'])}
