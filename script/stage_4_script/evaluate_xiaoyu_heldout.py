"""Reproduce a held-out evaluation of Xiaoyu's Stage 4 joke generator."""

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument(
    "--repo-dir", type=Path, default=Path(__file__).resolve().parent.parent.parent,
    help="Repository root (defaults to the parent of script/stage_4_script).",
)
PARSER.add_argument(
    "--data-dir", type=Path,
    help="Directory containing text_generation/data; defaults to REPO/data/stage_4_data.",
)
PARSER.add_argument(
    "--split-ids", type=Path,
    help="Canonical generation_split_ids.json; defaults to the copy in REPO.",
)
PARSER.add_argument(
    "--output", type=Path,
    help="Fresh output directory; defaults to ignored REPO/result/stage_4_result/smoke/xiaoyu_heldout.",
)
PARSER.add_argument(
    "--device", choices=("auto", "cpu", "cuda"), default="auto",
)
PARSER.add_argument(
    "--protocol-only", action="store_true",
    help="Validate data and split, print protocol JSON, then exit without training or writing files.",
)
ARGS = PARSER.parse_args()
REPO = ARGS.repo_dir.resolve()
DATA = (ARGS.data_dir or REPO / "data/stage_4_data").resolve()
OUTPUT = (ARGS.output or REPO / "result/stage_4_result/smoke/xiaoyu_heldout").resolve()
sys.path.insert(0, str(REPO))

from local_code.stage_4_code.Dataset_Loader import _normalized
from local_code.stage_4_code.xiaoyu.Dataset_Loader import Dataset_Loader, tokenize
from local_code.stage_4_code.xiaoyu.Method_RNN_Generation import Method_RNN_Generation

SPLITS_FILE = (ARGS.split_ids or REPO / "result/stage_4_result/runs/generation_split_ids.json").resolve()
JOKES_FILE = DATA / "text_generation/data"
CELLS = ("rnn", "lstm", "gru")
SEED = 2
MAX_LEN = 70
MAX_VOCAB = 10000
EPOCHS = 15
BATCH_SIZE = 32
EMBEDDING_DIM = 50
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
DEVICE = "cuda" if ARGS.device == "auto" and torch.cuda.is_available() else "cpu" if ARGS.device == "auto" else ARGS.device
if DEVICE == "cuda" and not torch.cuda.is_available():
    PARSER.error("CUDA is unavailable")


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_splits():
    ids = json.loads(SPLITS_FILE.read_text())
    all_ids = [item for name in ("train", "validation", "test") for item in ids[name]]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Split IDs overlap")
    with JOKES_FILE.open(encoding="utf-8-sig", newline="") as source:
        rows = {f"joke/{row['ID']}": row["Joke"] for row in csv.DictReader(source)}
    if set(all_ids) != set(rows):
        raise RuntimeError("Split IDs do not exactly cover source")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left = {_normalized(rows[item]) for item in ids[a]}
        right = {_normalized(rows[item]) for item in ids[b]}
        if left & right:
            raise RuntimeError(f"Normalized joke text overlaps: {a}/{b}")
    return ids, rows


def prepare(ids, rows):
    tokens = {
        name: [tokenize(rows[item])[: MAX_LEN - 1] for item in ids[name]]
        for name in ("train", "validation", "test")
    }
    loader = Dataset_Loader("jokes", "held-out", DATA, "generation", MAX_LEN, MAX_VOCAB)
    vocab = loader._vocab(tokens["train"])
    sequences = {
        name: [[vocab.get(word, 1) for word in joke] + [2] for joke in tokens[name]]
        for name in tokens
    }
    for name in sequences:
        if min(map(len, sequences[name])) < 2:
            raise RuntimeError(f"{name} contains a joke without a predictable token")
    oov = {
        name: {
            "unknown_targets": sum(value == 1 for joke in sequences[name] for value in joke[1:]),
            "predicted_tokens": sum(len(joke) - 1 for joke in sequences[name]),
        }
        for name in sequences
    }
    return vocab, sequences, oov


@torch.no_grad()
def evaluate(model, sequences):
    width = max(map(len, sequences)) - 1
    x = torch.zeros((len(sequences), width), dtype=torch.long)
    y = torch.zeros((len(sequences), width), dtype=torch.long)
    for i, joke in enumerate(sequences):
        x[i, : len(joke) - 1] = torch.tensor(joke[:-1], dtype=torch.long)
        y[i, : len(joke) - 1] = torch.tensor(joke[1:], dtype=torch.long)
    data = DataLoader(TensorDataset(x, y), batch_size=128, shuffle=False)
    model.eval()
    total_loss = 0.0
    correct = 0
    target_count = 0
    for xb, yb in data:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        logits, _ = model.forward(xb)
        total_loss += F.cross_entropy(
            logits.reshape(-1, len(model.vocab)),
            yb.reshape(-1),
            ignore_index=0,
            reduction="sum",
        ).item()
        mask = yb.ne(0)
        correct += ((logits.argmax(-1) == yb) & mask).sum().item()
        target_count += mask.sum().item()
    ce = total_loss / target_count
    return {
        "n_jokes": len(sequences),
        "n_target_tokens": target_count,
        "cross_entropy": ce,
        "perplexity": math.exp(ce),
        "token_accuracy": correct / target_count,
    }


def main():
    ids, rows = load_splits()
    vocab, sequences, oov = prepare(ids, rows)
    protocol = {
        "source": "Xiaoyu Stage 4 generator, originally PR #14 d0797e8; generation code unchanged at PR #14 4e6e007",
        "source_files": {
            str(path.relative_to(REPO)): sha256_file(path)
            for path in (
                REPO / "local_code/stage_4_code/xiaoyu/Dataset_Loader.py",
                REPO / "local_code/stage_4_code/xiaoyu/Method_RNN_Generation.py",
            )
        },
        "data_sha256": sha256_file(JOKES_FILE),
        "split_sha256": sha256_file(SPLITS_FILE),
        "split_policy": "Exact canonical Hongzhe baseline joke IDs, duplicate normalized jokes grouped; source 1,622 jokes",
        "split_counts": {name: len(ids[name]) for name in ids},
        "vocab": "Xiaoyu tokenizer, Xiaoyu vocabulary builder fit on train split only",
        "vocab_size": len(vocab),
        "oov": oov,
        "settings": {
            "seed": SEED,
            "max_len": MAX_LEN,
            "max_vocab": MAX_VOCAB,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_size": HIDDEN_SIZE,
            "learning_rate": LEARNING_RATE,
            "device": DEVICE,
            "selected_epoch": "epoch 15, matching original fixed training schedule; no test-based selection",
        },
    }
    if ARGS.protocol_only:
        print(json.dumps(protocol, indent=2))
        return
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print("Protocol:", json.dumps({"counts": protocol["split_counts"], "vocab": len(vocab), "device": DEVICE}), flush=True)
    for cell in CELLS:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(SEED)
        model = Method_RNN_Generation(
            cell, "", vocab, cell, EMBEDDING_DIM, HIDDEN_SIZE, EPOCHS,
            BATCH_SIZE, LEARNING_RATE, DEVICE,
        )
        model.data = {"sequences": sequences["train"]}
        model.run()
        result = {
            "cell": cell,
            "train": evaluate(model, sequences["train"]),
            "validation": evaluate(model, sequences["validation"]),
            "test": evaluate(model, sequences["test"]),
            "history": model.history,
        }
        (OUTPUT / f"{cell}.json").write_text(json.dumps(result, indent=2))
        torch.save(
            {"weights": model.state_dict(), "vocab": vocab, "protocol": protocol["settings"]},
            OUTPUT / f"{cell}.pt",
        )
        print(
            "RESULT",
            cell,
            "test_ce",
            f"{result['test']['cross_entropy']:.6f}",
            "test_ppl",
            f"{result['test']['perplexity']:.3f}",
            "test_tokens",
            result["test"]["n_target_tokens"],
            flush=True,
        )


if __name__ == "__main__":
    main()
