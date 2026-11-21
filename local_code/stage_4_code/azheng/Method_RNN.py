'''
Recurrent classifier for text (movie-review sentiment): RNN, LSTM or GRU

    word ids (batch, n)  ->  Embedding (batch, n, d)  ->  recurrent layer(s) (hidden size h)  ->  last real step
    -> Dropout -> Linear -> 2 logits

Padding zeros are skipped with pack_padded_sequence, so the final hidden state is the one after the last real word.
The network outputs raw logits: nn.CrossEntropyLoss applies log-softmax itself.
'''
import io
import time
import copy
import contextlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from local_code.base_class.method import method
from local_code.stage_4_code.azheng.Evaluate_Metrics import Evaluate_Metrics


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Method_RNN(method, nn.Module):
    data = None

    # ---- hyper-parameters (set from the script before calling run()) ----
    num_classes = 2
    embedding_dim = 50             # d (start small)
    hidden_size = 64               # h
    num_layers = 1
    rnn_type = 'rnn'               # rnn | lstm | gru
    bidirectional = False
    pooling = 'last'               # last: final hidden state | mean: average of the hidden states over the real (non-padding) words
    dropout = 0.3                  # before the output layer (and between layers if num_layers > 1)
    embedding_dropout = 0.0        # randomly zero whole embedding values of the input words (regularises the 1M embedding weights)
    glove_path = None              # optional: initialise the embedding from a GloVe file (sets embedding_dim)
    freeze_embedding = False
    optimizer_name = 'adam'        # adam | adamw | sgd | rmsprop
    learning_rate = 1e-3
    weight_decay = 0.0
    grad_clip = 1.0                # recurrent nets can explode: clip the gradient norm
    batch_size = 64
    max_epoch = 10
    restore_best = True            # after training, go back to the epoch with the best VALIDATION accuracy
    seed = 2
    use_gpu = False
    verbose = True

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.embedding = None
        self.rnn = None
        self.classifier = None
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.best_epoch = None

    # ------------------------------------------------------------------ model
    def build(self, vocab_size, embedding_matrix=None):
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(embedding_matrix))
            self.embedding.weight.requires_grad = not self.freeze_embedding
        cell = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[self.rnn_type]
        self.rnn = cell(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, batch_first=True,
                        bidirectional=self.bidirectional, dropout=self.dropout if self.num_layers > 1 else 0.0)
        self.drop = nn.Dropout(self.dropout)
        self.emb_drop = nn.Dropout(self.embedding_dropout)
        self.classifier = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), self.num_classes)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1).clamp(min=1)                       # real words per review
        packed = pack_padded_sequence(self.emb_drop(self.embedding(x)), lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        if self.pooling == 'mean':
            out, _ = pad_packed_sequence(out, batch_first=True)           # (batch, longest review, h * directions)
            mask = (torch.arange(out.size(1), device=out.device)[None, :] < lengths.to(out.device)[:, None]).unsqueeze(2)
            feature = (out * mask).sum(dim=1) / lengths.to(out.device).unsqueeze(1).to(out.dtype)
            return self.classifier(self.drop(feature))
        if self.rnn_type == 'lstm':
            h = h[0]                                                      # (h_n, c_n) -> h_n
        last = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
        return self.classifier(self.drop(last))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def make_optimizer(self):
        name = self.optimizer_name.lower()
        p = [q for q in self.parameters() if q.requires_grad]
        lr, wd = self.learning_rate, self.weight_decay
        if name == 'adam':
            return torch.optim.Adam(p, lr=lr, weight_decay=wd)
        if name == 'adamw':
            return torch.optim.AdamW(p, lr=lr, weight_decay=wd)
        if name == 'sgd':
            return torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=wd)
        if name == 'rmsprop':
            return torch.optim.RMSprop(p, lr=lr, weight_decay=wd)
        raise ValueError('unknown optimizer: ' + self.optimizer_name)

    def predict(self, X):
        self.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 500):
                xb = torch.as_tensor(np.asarray(X[i:i + 500]), dtype=torch.long, device=self.device)
                preds.append(self.forward(xb).argmax(dim=1).cpu())
        return torch.cat(preds).numpy()

    # ------------------------------------------------------------------ training
    def train_model(self, X, y, X_val=None, y_val=None):
        optimizer = self.make_optimizer()
        loss_function = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(torch.as_tensor(np.asarray(X), dtype=torch.long),
                                          torch.as_tensor(np.asarray(y), dtype=torch.long)),
                            batch_size=self.batch_size, shuffle=True)      # mini-batches, shuffled every epoch
        accuracy_evaluator = Evaluate_Metrics('training evaluator', '')
        y_np = np.asarray(y)
        self.loss_history, self.train_acc_history, self.val_acc_history = [], [], []
        best_val, best_state = -1.0, None

        for epoch in range(self.max_epoch):
            self.train()
            total = torch.zeros((), device=self.device)
            n = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = loss_function(self.forward(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                optimizer.step()
                total += loss.detach() * len(xb)
                n += len(xb)
            epoch_loss = total.item() / n
            self.loss_history.append(epoch_loss)

            accuracy_evaluator.data = {'true_y': y_np, 'pred_y': self.predict(X)}
            with contextlib.redirect_stdout(io.StringIO()):
                train_acc = accuracy_evaluator.evaluate()['accuracy']
            self.train_acc_history.append(train_acc)

            msg = 'Epoch: %d  Loss: %.4f  Train Acc: %.4f' % (epoch + 1, epoch_loss, train_acc)
            if X_val is not None:
                val_acc = float((self.predict(X_val) == np.asarray(y_val)).mean())
                self.val_acc_history.append(val_acc)
                msg += '  Val Acc: %.4f' % val_acc
                if val_acc > best_val:
                    best_val, self.best_epoch = val_acc, epoch + 1
                    best_state = copy.deepcopy(self.state_dict())
            if self.verbose:
                print(msg)
        if self.restore_best and best_state is not None:
            self.load_state_dict(best_state)
            if self.verbose:
                print('restored the epoch with the best validation accuracy: epoch %d (%.4f)' % (self.best_epoch, best_val))

    def run(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = get_device() if self.use_gpu else torch.device('cpu')
        vocab = self.data['vocab']
        matrix = None
        if self.glove_path:
            from local_code.stage_4_code.azheng.text_utils import load_glove_matrix
            matrix, found, self.embedding_dim = load_glove_matrix(self.glove_path, vocab, self.seed)
            print('GloVe: %d of %d vocabulary words found (dimension %d)' % (found, len(vocab), self.embedding_dim))
        self.build(len(vocab), matrix)
        self.to(self.device)
        print('method running on', self.device, '|', self.rnn_type.upper(), '| trainable parameters:', self.num_parameters())

        val = self.data.get('val')
        print('--start training...')
        t0 = time.time()
        self.train_model(self.data['train']['X'], self.data['train']['y'],
                         val['X'] if val else None, val['y'] if val else None)
        self.method_training_time = time.time() - t0

        print('--start testing...')
        t0 = time.time()
        pred_y = self.predict(self.data['test']['X'])
        self.method_testing_time = time.time() - t0
        return {'pred_y': pred_y, 'true_y': np.asarray(self.data['test']['y'])}
