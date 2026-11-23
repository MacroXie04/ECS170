"""Reproduce an isolated held-out evaluation of Azheng's joke generators.

Uses the frozen Stage 4 joke split IDs, but Azheng's tokenizer, vocabulary and
model. Outputs are deliberately separate from the team's original PR results.
Run from the repository root or pass --repo, --data-dir and --output.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(repo, data_dir, split_file, tokenize_joke, Vocabulary):
    raw = data_dir / "text_generation" / "data"
    splits = json.loads(split_file.read_text())
    members = {name: set(ids) for name, ids in splits.items()}
    if set(members) != {"train", "validation", "test"}:
        raise ValueError("expected train, validation and test split IDs")
    if len(set.union(*members.values())) != sum(map(len, members.values())):
        raise ValueError("overlapping split IDs")

    docs = {name: [] for name in members}
    dropped = []
    with raw.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            identifier = "joke/" + row["ID"]
            tokens = tokenize_joke(row["Joke"])[:39] + ["<eos>"]
            if len(tokens) <= 4:
                dropped.append(identifier)
                continue
            for name, ids in members.items():
                if identifier in ids:
                    docs[name].append(tokens)
                    break
            else:
                raise ValueError("row absent from frozen split: " + identifier)

    vocab = Vocabulary(specials=("<pad>", "<unk>", "<eos>")).build(
        docs["train"], max_size=100000, min_freq=2
    )
    arrays = {
        name: np.asarray([vocab.encode(tokens, 40) for tokens in rows], dtype=np.int64)
        for name, rows in docs.items()
    }
    sets = {name: {tuple(tokens) for tokens in rows} for name, rows in docs.items()}
    overlaps = {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_test": len(sets["train"] & sets["test"]),
        "validation_test": len(sets["validation"] & sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError("tokenized joke duplicates cross split boundaries")
    test_targets = int((arrays["test"][:, 1:] != 0).sum())
    info = {
        "counts": {name: len(rows) for name, rows in docs.items()},
        "dropped_ids": dropped,
        "vocab_size": len(vocab),
        "test_targets": test_targets,
        "unknown_test_target_rate": float(
            (arrays["test"][:, 1:] == 1).sum() / test_targets
        ),
        "tokenized_duplicate_overlap": overlaps,
        "raw_sha256": digest(raw),
        "split_sha256": digest(split_file),
        "model_source_sha256": digest(
            repo / "local_code/stage_4_code/azheng/Method_RNN_Generator.py"
        ),
        "loader_source_sha256": digest(
            repo / "local_code/stage_4_code/azheng/Dataset_Loader_Jokes.py"
        ),
    }
    return arrays, vocab, info


def score(model, test):
    loader = DataLoader(
        TensorDataset(torch.as_tensor(test, dtype=torch.long)), batch_size=256
    )
    loss = model._epoch_loss(loader)
    return float(loss), float(math.exp(loss))


def train(cell, arrays, vocab, info, output, Method_RNN_Generator, device):
    json_path = output / (cell + ".json")
    checkpoint_path = output / (cell + ".pt")
    if json_path.exists() or checkpoint_path.exists():
        raise FileExistsError("use a fresh output directory for training")

    model = Method_RNN_Generator("held-out evaluation", "")
    model.rnn_type = cell
    model.use_gpu = device == "cuda"
    model.data = {
        "vocab": vocab,
        "train": {"X": arrays["train"]},
        "val": {"X": arrays["validation"]},
    }
    model.run()
    test_loss, test_ppl = score(model, arrays["test"])
    result = {
        "cell": cell,
        "device": str(model.device),
        "seed": model.seed,
        "max_epoch": model.max_epoch,
        "best_epoch": model.best_epoch,
        "epochs_completed": len(model.loss_history),
        "test_cross_entropy": test_loss,
        "test_perplexity": test_ppl,
        "validation_cross_entropy": float(min(model.val_loss_history)),
        "validation_perplexity": float(math.exp(min(model.val_loss_history))),
        "test_examples": len(arrays["test"]),
        "test_targets": info["test_targets"],
        "config": {
            "embedding_dim": model.embedding_dim,
            "hidden_size": model.hidden_size,
            "dropout": model.dropout,
            "batch_size": model.batch_size,
            "learning_rate": model.learning_rate,
            "max_len": 40,
            "min_freq": 2,
        },
        "data": info,
    }
    output.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"state_dict": model.state_dict(), "vocab": vocab.itos}, checkpoint_path)
    return result


def verify(cell, arrays, vocab, info, output, Method_RNN_Generator, device):
    result = json.loads((output / (cell + ".json")).read_text())
    for key in ("raw_sha256", "split_sha256", "model_source_sha256",
                "loader_source_sha256"):
        if result["data"][key] != info[key]:
            raise ValueError(key + " changed")
    saved = torch.load(output / (cell + ".pt"), map_location="cpu", weights_only=False)
    if saved["vocab"] != vocab.itos:
        raise ValueError("vocabulary changed")
    model = Method_RNN_Generator("checkpoint replay", "")
    model.rnn_type = cell
    model.device = torch.device(device)
    model.data = {"vocab": vocab}
    model.build(len(vocab))
    model.load_state_dict(saved["state_dict"])
    model.to(model.device)
    loss, ppl = score(model, arrays["test"])
    if abs(loss - result["test_cross_entropy"]) > 1e-6:
        raise AssertionError("checkpoint replay changed test loss")
    return {"cell": cell, "test_cross_entropy": loss, "test_perplexity": ppl,
            "checkpoint_replay_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split-ids", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cell", choices=("rnn", "lstm", "gru", "all"), default="all")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true", help="train and evaluate")
    action.add_argument("--verify", action="store_true", help="replay saved checkpoints")
    args = parser.parse_args()
    repo = args.repo.resolve()
    data_dir = (args.data_dir or repo / "data/stage_4_data").resolve()
    split_file = (args.split_ids or repo / "result/stage_4_result/runs/generation_split_ids.json").resolve()
    if args.device == "cuda" and (args.run or args.verify) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu")
    sys.path.insert(0, str(repo))
    from local_code.stage_4_code.azheng.Dataset_Loader_Jokes import tokenize_joke
    from local_code.stage_4_code.azheng.Method_RNN_Generator import Method_RNN_Generator
    from local_code.stage_4_code.azheng.text_utils import Vocabulary

    torch.set_num_threads(2)
    arrays, vocab, info = load(repo, data_dir, split_file, tokenize_joke, Vocabulary)
    print(json.dumps(info, indent=2), flush=True)
    cells = ("rnn", "lstm", "gru") if args.cell == "all" else (args.cell,)
    if args.run:
        for cell in cells:
            result = train(cell, arrays, vocab, info, args.output, Method_RNN_Generator, args.device)
            print(json.dumps({key: result[key] for key in (
                "cell", "best_epoch", "test_cross_entropy", "test_perplexity",
                "test_examples", "test_targets"
            )}), flush=True)
    elif args.verify:
        for cell in cells:
            print(json.dumps(verify(cell, arrays, vocab, info, args.output,
                                    Method_RNN_Generator, args.device)), flush=True)


if __name__ == "__main__":
    main()
