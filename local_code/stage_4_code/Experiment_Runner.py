"""Reproducible Stage 4 experiments; checkpoints are selected on validation only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader, tokenize
from local_code.stage_4_code.Method_RNN import SequenceModel, generate


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def collate(records):
    lengths = torch.tensor([len(r['tokens']) for r in records], dtype=torch.long)
    tokens = torch.zeros((len(records), int(lengths.max())), dtype=torch.long)
    for i, record in enumerate(records):
        tokens[i, :lengths[i]] = torch.tensor(record['tokens'], dtype=torch.long)
    labels = torch.tensor([r.get('label', -1) for r in records], dtype=torch.long)
    return tokens, lengths, labels, [r['id'] for r in records]


def sequence_objective(logits, targets):
    """Return SUM of non-padding next-token losses, correct count and token count."""
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                       ignore_index=0, reduction='sum')
    mask = targets.ne(0)
    correct = (logits.argmax(-1).eq(targets) & mask).sum()
    return loss, correct, mask.sum()


def run_epoch(model, loader, task, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for tokens, lengths, labels, _ in loader:
            tokens, lengths, labels = tokens.to(device), lengths.to(device), labels.to(device)
            if task == 'classification':
                logits = model(tokens, lengths)
                summed_loss = nn.functional.cross_entropy(logits, labels, reduction='sum')
                correct, count = logits.argmax(-1).eq(labels).sum(), labels.numel()
            else:
                logits = model(tokens[:, :-1], lengths - 1)
                summed_loss, correct, count = sequence_objective(logits, tokens[:, 1:])
            count = int(count)
            loss = summed_loss / count
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; refusing to save a successful result')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            total_loss += float(summed_loss.detach())
            total_correct += int(correct)
            total_count += count
    return {'loss': total_loss / total_count, 'accuracy': total_correct / total_count,
            'count': total_count}


def evaluation(model, loader, task, device, output):
    """Write sufficient held-out evidence to recompute every reported test metric."""
    model.eval()
    rows = []
    with torch.no_grad():
        for tokens, lengths, labels, ids in loader:
            tokens, lengths = tokens.to(device), lengths.to(device)
            if task == 'classification':
                logits = model(tokens, lengths)
                probs = logits.softmax(-1)[:, 1].cpu().tolist()
                pred = logits.argmax(-1).cpu().tolist()
                losses = nn.functional.cross_entropy(logits, labels.to(device), reduction='none').cpu().tolist()
                rows.extend(dict(id=i, label=int(y), prediction=int(p), positive_probability=prob, nll=loss)
                            for i, y, p, prob, loss in zip(ids, labels.tolist(), pred, probs, losses))
            else:
                target = tokens[:, 1:]
                logits = model(tokens[:, :-1], lengths - 1)
                losses = nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten(),
                    ignore_index=0, reduction='none').reshape_as(target)
                mask = target.ne(0)
                counts = mask.sum(-1).cpu().tolist()
                correct = (logits.argmax(-1).eq(target) & mask).sum(-1).cpu().tolist()
                sums = losses.sum(-1).cpu().tolist()
                rows.extend(dict(id=i, token_count=n, correct_tokens=c, nll_sum=s)
                            for i, n, c, s in zip(ids, counts, correct, sums))
    with (output / 'test_predictions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if task == 'classification':
        truth = [r['label'] for r in rows]
        pred = [r['prediction'] for r in rows]
        precision, recall, f1, _ = precision_recall_fscore_support(truth, pred, average='macro', zero_division=0)
        return dict(n_examples=len(rows), loss=float(np.mean([r['nll'] for r in rows])),
                    accuracy=accuracy_score(truth, pred), macro_precision=float(precision),
                    macro_recall=float(recall), macro_f1=float(f1),
                    confusion_matrix=confusion_matrix(truth, pred, labels=[0, 1]).tolist())
    n = sum(r['token_count'] for r in rows)
    loss = sum(r['nll_sum'] for r in rows) / n
    return dict(n_examples=len(rows), n_tokens=n, loss=loss, perplexity=math.exp(loss),
                token_accuracy=sum(r['correct_tokens'] for r in rows) / n)


def generation_examples(model, data, seed):
    """Fixed training-derived 3-WORD prefixes, shared across models; no test selection."""
    vocab = data['vocab']
    selected = []
    # These prefixes are matched against training data, so references are evidence
    # of memorization/reconstruction, not held-out generation quality.
    for prefix in ('what do you', 'why did the', 'what did the'):
        candidates = [r for r in data['train'] if r['text'].strip().lower().startswith(prefix + ' ')]
        if candidates:
            # Shortest reference chosen before seeing model output; fixed tie break.
            selected.append((prefix, min(candidates, key=lambda r: (len(tokenize(r['text'])), r['id']))))
    if not selected:
        raise ValueError('No suitable three-word training prefix found')
    examples = []
    for index, (prefix, reference) in enumerate(selected):
        prefix_ids = vocab.encode(prefix, bos=True)
        greedy = generate(model, prefix_ids, max_new_tokens=40, temperature=0, seed=seed)
        sampled = generate(model, prefix_ids, max_new_tokens=40, temperature=.8, top_k=20, seed=seed + index)
        examples.append(dict(prefix=prefix, source_split='train', reference_id=reference['id'],
            reference=reference['text'], greedy=vocab.decode(greedy), sampled=vocab.decode(sampled),
            sampling={'temperature': .8, 'top_k': 20, 'seed': seed + index}, max_new_tokens=40))
    return examples


def make_curve(history, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for split in ('train', 'validation'):
        ax.plot([h['epoch'] for h in history], [h[f'{split}_loss'] for h in history], label=split.title())
    ax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title=title)
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def experiment(args, task, cell, data, device):
    name = f'{task}_{cell}_{args.tag}'
    output = Path(args.output) / name
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    source_files = ['local_code/stage_4_code/Dataset_Loader.py',
                    'local_code/stage_4_code/Method_RNN.py',
                    'local_code/stage_4_code/Experiment_Runner.py',
                    'script/stage_4_script/run_stage4.py',
                    'script/stage_4_script/prepare_stage4_data.py']
    source_hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in source_files}
    vocab = data['vocab']
    model_kwargs = dict(vocab_size=len(vocab), task=task, cell=cell, embedding_dim=args.embedding_dim,
                        hidden_dim=args.hidden_dim, num_layers=args.layers, dropout=args.dropout)
    epochs = args.max_epochs or (12 if task == 'classification' else 80)
    patience = args.patience or (4 if task == 'classification' else 12)
    if args.smoke:
        epochs = 1
    config = dict(model=model_kwargs, seed=args.seed, max_epochs=epochs, patience=patience,
        batch_size=args.batch_size, learning_rate=args.learning_rate, optimizer='Adam',
        weight_decay=0.0, gradient_clip_norm=1.0, checkpoint_selection='lowest validation loss; earliest tie',
        classification_max_length=args.max_length, max_vocab=args.max_vocab, smoke=args.smoke,
        tag=args.tag, tokenizer='lowercase HTML-cleaned word/punctuation tokenizer',
        data_metadata=data['metadata'], source_sha256=source_hashes)
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False)
    config_hash = hashlib.sha256(canonical.encode()).hexdigest()
    if (output / 'config.json').exists():
        if json.loads((output / 'config.json').read_text()) != config:
            raise ValueError(f'Existing run has different configuration: {output}. Use a new --tag/output.')
    write_json(output / 'config.json', config)
    if (output / 'metrics.json').exists():
        print(f'SKIP completed {name}', flush=True)
        return
    write_json(output / 'environment.json', dict(python=sys.version, torch=str(torch.__version__),
        numpy=np.__version__, platform=platform.platform(), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), command=sys.argv,
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256=source_hashes, parameter_dtype='float32', cudnn_deterministic=True,
        initialization='PyTorch defaults: Embedding N(0,1) with zero PAD; recurrent U(-1/sqrt(H),1/sqrt(H)); Linear Kaiming-uniform default',
        note='Source hashes bind uncommitted code. Seeded stochastic operations; cross-device bitwise identity is not guaranteed.'))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    model = SequenceModel(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(data[split], batch_size=args.batch_size,
                shuffle=(split == 'train'), generator=generator if split == 'train' else None,
                collate_fn=collate, num_workers=0, pin_memory=(device.type == 'cuda'))
               for split in ('train', 'validation', 'test')}
    checkpoint_dir = output / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    history, best_loss, best_epoch, seconds = [], float('inf'), 0, 0.0
    last = checkpoint_dir / 'last.pt'
    if last.exists():
        ckpt = torch.load(last, map_location=device, weights_only=True)
        if ckpt['config_hash'] != config_hash:
            raise ValueError('Checkpoint configuration mismatch')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        history, best_loss, best_epoch, seconds = ckpt['history'], ckpt['best_loss'], ckpt['best_epoch'], ckpt['seconds']
        torch.set_rng_state(ckpt['rng'].cpu())
        generator.set_state(ckpt['loader_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([state.cpu() for state in ckpt['cuda_rng']])
        print(f'RESUME {name} after epoch {len(history)}', flush=True)
    print(f'START {name} parameters={sum(p.numel() for p in model.parameters())} vocab={len(vocab)} device={device}', flush=True)
    for epoch in range(len(history) + 1, epochs + 1):
        if history and len(history) - best_epoch >= patience:
            break
        start = time.monotonic()
        train = run_epoch(model, loaders['train'], task, device, optimizer)
        validation = run_epoch(model, loaders['validation'], task, device)
        elapsed = time.monotonic() - start
        seconds += elapsed
        row = dict(epoch=epoch, train_loss=train['loss'], train_accuracy=train['accuracy'],
            validation_loss=validation['loss'], validation_accuracy=validation['accuracy'], seconds=elapsed)
        history.append(row)
        if validation['loss'] < best_loss:
            best_loss, best_epoch = validation['loss'], epoch
            save_checkpoint(checkpoint_dir / 'best.pt', dict(model=model.state_dict(), model_kwargs=model_kwargs,
                config_hash=config_hash, epoch=epoch, vocab=vocab.to_dict()))
        write_json(output / 'history.json', history)
        save_checkpoint(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            config_hash=config_hash, history=history, best_loss=best_loss, best_epoch=best_epoch, seconds=seconds,
            rng=torch.get_rng_state(), loader_rng=generator.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else []))
        print(json.dumps({'run': name, **row, 'best_epoch': best_epoch}), flush=True)
    best = torch.load(checkpoint_dir / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(best['model'])
    test_metrics = evaluation(model, loaders['test'], task, device, output)
    if task == 'generation':
        write_json(output / 'generation_examples.json', generation_examples(model, data, args.seed))
    make_curve(history, output / 'learning_curve.png', f'{task.title()} - {cell.upper()}')
    metrics = dict(task=task, cell=cell, tag=args.tag, best_epoch=best_epoch, epochs_completed=len(history),
        validation_loss=best_loss, train_seconds=seconds,
        parameters=sum(p.numel() for p in model.parameters()), test=test_metrics, smoke=args.smoke,
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)
    write_json(output / 'metrics.json', metrics)
    print('COMPLETE ' + json.dumps({'run': name, **metrics}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/stage_4_data')
    parser.add_argument('--output', default='result/stage_4_result/runs')
    parser.add_argument('--task', choices=['all', 'classification', 'generation'], default='all')
    parser.add_argument('--cells', default='rnn,lstm,gru')
    parser.add_argument('--tag', default='baseline')
    parser.add_argument('--seed', type=int, default=1704)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=.3)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--max-vocab', type=int, default=20000)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    cells = args.cells.split(',')
    if not cells or any(cell not in ('rnn', 'lstm', 'gru') for cell in cells):
        parser.error('--cells must contain only rnn,lstm,gru')
    if args.batch_size < 1 or args.layers < 1 or args.max_length < 1:
        parser.error('batch size, layers and max length must be positive')
    if args.smoke and args.output == 'result/stage_4_result/runs':
        args.output = 'result/stage_4_result/smoke'
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'environment.json', dict(python=sys.version, torch=str(torch.__version__),
        numpy=np.__version__, platform=platform.platform(), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), command=sys.argv,
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        note='Seeded data/order/dropout; bitwise reproducibility across devices is not guaranteed.'))
    for task in (('classification', 'generation') if args.task == 'all' else (args.task,)):
        loader = Dataset_Loader(args.data_dir, seed=args.seed, max_vocab=args.max_vocab, max_length=args.max_length)
        data = getattr(loader, f'load_{task}')()
        write_json(output / f'{task}_metadata.json', data['metadata'])
        write_json(output / f'{task}_vocabulary.json', data['vocab'].to_dict())
        write_json(output / f'{task}_split_ids.json', {s: [r['id'] for r in data[s]] for s in ('train', 'validation', 'test')})
        if args.smoke:
            data = {**data, **{s: data[s][:32 if s == 'train' else 16] for s in ('train', 'validation', 'test')}}
        for cell in cells:
            experiment(args, task, cell, data, device)


if __name__ == '__main__':
    main()
