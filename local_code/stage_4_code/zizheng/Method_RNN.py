"""Recurrent models for review classification and next-token prediction."""

from typing import TypedDict

import torch
from torch import nn

from ...base_class.method import method
from .Dataset_Loader import EOS, PAD, UNK, tokenize


class EvaluationResults(TypedDict):
    loss: float
    accuracy: float
    count: int
    true_labels: list[int]
    predicted_labels: list[int]


class Method_RNN(method, nn.Module):
    def __init__(self, task, cell, vocab_size, embedding_size, hidden_size, dropout):
        method.__init__(self, cell, "Stage 4 recurrent text model")
        nn.Module.__init__(self)
        self.task = task
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)

        recurrent_layers = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}
        recurrent_layer = recurrent_layers[cell]
        self.recurrent = recurrent_layer(embedding_size, hidden_size, batch_first=True)

        if task == "classify":
            self.output = nn.Linear(hidden_size, 2)
        else:
            self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids):
        embeddings = self.dropout(self.embedding(token_ids))
        hidden_states, _ = self.recurrent(embeddings)

        if self.task == "classify":
            # Average only the hidden states at real tokens. Padding is on the right.
            real_tokens = token_ids.ne(PAD).unsqueeze(-1)
            hidden_sum = (hidden_states * real_tokens).sum(dim=1)
            hidden_states = hidden_sum / real_tokens.sum(dim=1).clamp_min(1)

        return self.output(self.dropout(hidden_states))

    def fit(self, train_batches, val_batches, epochs, patience, learning_rate):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        history = []
        best_loss = float("inf")
        best_epoch = 0
        best_weights = None

        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = self.train_epoch(train_batches, optimizer)
            validation = self.measure(val_batches)
            val_loss = validation["loss"]
            val_accuracy = validation["accuracy"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
            history.append(row)
            print(
                f"{self.method_name} epoch {epoch:02d} | train loss {train_loss:.4f} "
                f"| val loss {val_loss:.4f} | val accuracy {val_accuracy:.4f}",
                flush=True,
            )
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_weights = {}
                for name, weight in self.state_dict().items():
                    best_weights[name] = weight.clone()
            if epoch - best_epoch >= patience:
                break

        if best_weights is None:
            raise ValueError("Training did not produce a checkpoint.")
        self.load_state_dict(best_weights)
        return history, best_epoch

    def train_epoch(self, batches, optimizer):
        self.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for token_ids, targets in batches:
            optimizer.zero_grad()
            logits = self(token_ids)
            loss, correct, count = self.batch_scores(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Training loss became non-finite.")
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * count
            total_correct += correct
            total_count += count

        return total_loss / total_count, total_correct / total_count

    def batch_scores(self, logits, targets):
        predictions = logits.detach().argmax(dim=-1)
        if self.task == "classify":
            loss = nn.functional.cross_entropy(logits, targets)
            correct = int((predictions == targets).sum().item())
            count = targets.numel()
        else:
            loss = nn.functional.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), ignore_index=PAD,
            )
            scored_tokens = targets.ne(PAD)
            correct = int(((predictions == targets) & scored_tokens).sum().item())
            count = int(scored_tokens.sum().item())
        return loss, correct, count

    @torch.no_grad()
    def measure(self, batches, keep_predictions=False) -> EvaluationResults:
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        true_labels = []
        predicted_labels = []

        for token_ids, targets in batches:
            logits = self(token_ids)
            loss, correct, count = self.batch_scores(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Evaluation loss became non-finite.")
            total_loss += loss.item() * count
            total_correct += correct
            total_count += count

            if keep_predictions:
                true_labels.extend(targets.tolist())
                predicted_labels.extend(logits.argmax(dim=-1).tolist())

        return {
            "loss": total_loss / total_count,
            "accuracy": total_correct / total_count,
            "count": total_count,
            "true_labels": true_labels,
            "predicted_labels": predicted_labels,
        }

    @torch.no_grad()
    def generate(self, prompt, vocab, max_new_tokens=60):
        self.eval()
        tokens = tokenize(prompt)
        if len(tokens) != 3:
            raise ValueError("The prompt must contain three tokens.")

        words = sorted(vocab, key=vocab.get)
        prefix_ids = []
        for token in tokens:
            prefix_ids.append(vocab.get(token, UNK))

        token_ids = torch.tensor([prefix_ids], dtype=torch.long)
        state = None
        for _ in range(max_new_tokens):
            embeddings = self.embedding(token_ids)
            hidden_states, state = self.recurrent(embeddings, state)
            scores = self.output(hidden_states[0, -1])
            scores[PAD] = -torch.inf
            scores[UNK] = -torch.inf
            next_id = scores.argmax().item()
            if next_id == EOS:
                return tokens, True
            tokens.append(words[next_id])
            token_ids = torch.tensor([[next_id]], dtype=torch.long)

        return tokens, False
