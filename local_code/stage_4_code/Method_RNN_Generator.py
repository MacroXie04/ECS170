'''
Word-level recurrent language model for text generation (RNN, LSTM or GRU)

    token ids (batch, T) -> Embedding (d) -> recurrent layer(s) (h) -> Linear -> logits over the vocabulary at every step

Training predicts the next token at every position (teacher forcing). To generate, the three start words are fed in and
the model then keeps choosing the next token until it produces <eos> or reaches max_len.
'''
import time
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from local_code.base_class.method import method
from local_code.stage_4_code.Method_RNN import get_device


class Method_RNN_Generator(method, nn.Module):
    data = None

    embedding_dim = 64
    hidden_size = 128
    num_layers = 1
    rnn_type = 'lstm'              # rnn | lstm | gru
    glove_path = None              # optional: initialise the embedding from a GloVe file (sets embedding_dim); fine-tuned
    freeze_embedding = False
    dropout = 0.3
    learning_rate = 2e-3
    weight_decay = 0.0
    grad_clip = 1.0
    batch_size = 64
    max_epoch = 60
    restore_best = True            # keep the epoch with the lowest validation loss
    seed = 2
    use_gpu = False
    verbose = True

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.loss_history = []
        self.val_loss_history = []
        self.best_epoch = None

    def build(self, vocab_size, embedding_matrix=None):
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(embedding_matrix))
            self.embedding.weight.requires_grad = not self.freeze_embedding
        cell = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[self.rnn_type]
        self.rnn = cell(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, batch_first=True,
                        dropout=self.dropout if self.num_layers > 1 else 0.0)
        self.drop = nn.Dropout(self.dropout)
        self.out = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, x, state=None):
        h, state = self.rnn(self.embedding(x), state)
        return self.out(self.drop(h)), state

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _epoch_loss(self, loader, optimizer=None):
        loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')   # padding is not scored
        self.train(optimizer is not None)
        total, count = 0.0, 0
        with torch.set_grad_enabled(optimizer is not None):
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits, _ = self.forward(xb[:, :-1])
                targets = xb[:, 1:]
                loss = loss_function(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                n_tok = (targets != 0).sum().item()
                if optimizer is not None:
                    optimizer.zero_grad()
                    (loss / max(n_tok, 1)).backward()
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                    optimizer.step()
                total += loss.item()
                count += n_tok
        return total / count

    def train_model(self, X, X_val):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loader = DataLoader(TensorDataset(torch.as_tensor(X, dtype=torch.long)), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.as_tensor(X_val, dtype=torch.long)), batch_size=256)
        best, best_state = float('inf'), None
        self.loss_history, self.val_loss_history = [], []
        for epoch in range(self.max_epoch):
            tr = self._epoch_loss(loader, optimizer)
            va = self._epoch_loss(val_loader)
            self.loss_history.append(tr)
            self.val_loss_history.append(va)
            if va < best:
                best, self.best_epoch, best_state = va, epoch + 1, copy.deepcopy(self.state_dict())
            if self.verbose:
                print('Epoch: %d  Train Loss: %.4f  Val Loss: %.4f  Val Perplexity: %.1f' % (epoch + 1, tr, va, np.exp(va)))
        if self.restore_best and best_state is not None:
            self.load_state_dict(best_state)
            if self.verbose:
                print('restored epoch %d (lowest validation loss %.4f, perplexity %.1f)' % (self.best_epoch, best, np.exp(best)))

    @torch.no_grad()
    def generate(self, start_words, max_len=40, temperature=0.0, top_k=None, seed=None, min_len=0, no_repeat=False):
        '''start_words: list of word tokens. temperature 0 = always the most likely token (greedy);
        temperature > 0 samples (optionally only from the top_k tokens). Returns the full token list.
        min_len: <eos> is not allowed before the joke has this many tokens (the start words count).
        no_repeat: never repeat a word pair that is already in the joke (stops loops like 'a joke and a joke and a joke').'''
        self.eval()
        vocab = self.data['vocab']
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        tokens = list(start_words)
        ids = torch.as_tensor([vocab.encode(tokens, pad=False)], dtype=torch.long, device=self.device)
        logits, state = self.forward(ids)
        while len(tokens) < max_len:
            step = logits[0, -1].float().cpu()
            step[0] = step[1] = -float('inf')                       # never produce <pad> / <unk>
            if tokens[-1] in ('.', ',', '!', '?'):                  # no runs like '.....'
                for ch in ('.', ',', '!', '?'):
                    if ch in vocab.stoi:
                        step[vocab.stoi[ch]] = -float('inf')
            if min_len and len(tokens) < min_len:
                step[2] = -float('inf')                             # too short to stop yet
            if no_repeat:
                for a, b in zip(tokens, tokens[1:]):
                    if a == tokens[-1] and b in vocab.stoi:
                        step[vocab.stoi[b]] = -float('inf')
            if temperature <= 0:
                nxt = int(step.argmax())
            else:
                step = step / temperature
                if top_k:
                    kth = torch.topk(step, top_k).values[-1]
                    step[step < kth] = -float('inf')
                nxt = int(torch.multinomial(torch.softmax(step, 0), 1, generator=gen))
            if nxt == 2:                                            # <eos>
                break
            tokens.append(vocab.itos[nxt])
            logits, state = self.forward(torch.as_tensor([[nxt]], dtype=torch.long, device=self.device), state)
        return tokens

    def run(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = get_device() if self.use_gpu else torch.device('cpu')
        matrix = None
        if self.glove_path:
            from local_code.stage_4_code.text_utils import load_glove_matrix
            matrix, found, self.embedding_dim = load_glove_matrix(self.glove_path, self.data['vocab'], self.seed)
            print('GloVe: %d of %d vocabulary words found (dimension %d)' % (found, len(self.data['vocab']), self.embedding_dim))
        self.build(len(self.data['vocab']), matrix)
        self.to(self.device)
        print('method running on', self.device, '|', self.rnn_type.upper(), '| trainable parameters:', self.num_parameters())
        print('--start training...')
        t0 = time.time()
        self.train_model(self.data['train']['X'], self.data['val']['X'])
        self.method_training_time = time.time() - t0
        return {'val_loss': min(self.val_loss_history) if self.restore_best else self.val_loss_history[-1]}
