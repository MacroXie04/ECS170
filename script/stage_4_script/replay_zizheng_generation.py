"""Recompute Zizheng Stage 4 generation test metrics from saved checkpoints.

Run with the instructor data and a checkout containing the namespaced code.
The JSON contains hashes and aggregate scores, never joke text.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))

    import torch
    from torch.utils.data import DataLoader
    from local_code.stage_4_code.zizheng.Dataset_Loader import Dataset_Loader, generate_batch
    from local_code.stage_4_code.zizheng.Method_RNN import Method_RNN

    torch.set_num_threads(1)
    splits, vocab, _ = Dataset_Loader(args.data_root.resolve()).load("generate", smoke=False)
    loader = DataLoader(splits["test"], batch_size=32, shuffle=False, collate_fn=generate_batch)
    source_paths = [
        "local_code/stage_4_code/zizheng/Dataset_Loader.py",
        "local_code/stage_4_code/zizheng/Method_RNN.py",
        "local_code/stage_4_code/zizheng/Setting.py",
    ]
    report = {
        "repo_head": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "torch_version_replay": str(torch.__version__),
        "data_sha256": sha256_file(args.data_root / "text_generation" / "data"),
        "source_sha256": {name: sha256_file(repo / name) for name in source_paths},
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "encoded_split_sha256": {
            name: sha256_json(rows) for name, rows in splits.items()
        },
        "vocab_size": len(vocab),
        "vocab_sha256": sha256_json(vocab),
        "cross_split_overlap": {
            f"{a}/{b}": len({tuple(row) for row in splits[a]} & {tuple(row) for row in splits[b]})
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        },
        "models": {},
    }

    for cell in ("RNN", "LSTM", "GRU"):
        folder = args.result_root / cell
        record_path = folder / "record.json"
        checkpoint_path = folder / "model.pt"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        vocab_matches = checkpoint["vocab"] == vocab
        if not vocab_matches:
            raise AssertionError(f"{cell}: checkpoint vocabulary differs from recomputed vocabulary")
        model = Method_RNN(**checkpoint["settings"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        score = model.measure(loader)
        observed = {
            "loss": float(score["loss"]),
            "perplexity": float(math.exp(score["loss"])),
            "token_accuracy": float(score["accuracy"]),
            "scored_tokens": int(score["count"]),
        }
        saved = {
            "loss": float(record["test"]["loss"]),
            "perplexity": float(record["test"]["perplexity"]),
            "token_accuracy": float(record["test"]["accuracy"]),
            "scored_tokens": int(record["test"]["count"]),
        }
        deltas = {
            name: observed[name] - saved[name]
            for name in ("loss", "perplexity", "token_accuracy")
        }
        consistent = (
            observed["scored_tokens"] == saved["scored_tokens"]
            and all(abs(delta) <= 1e-4 for delta in deltas.values())
        )
        report["models"][cell] = {
            "record_sha256": sha256_file(record_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "record_torch_version": record["torch_version"],
            "best_epoch": record["best_epoch"],
            "settings": checkpoint["settings"],
            "vocab_matches": vocab_matches,
            "observed": observed,
            "saved": saved,
            "delta": deltas,
            "consistent_with_saved_record": consistent,
        }
        if not consistent:
            raise AssertionError(f"{cell}: replay differs from saved record")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}; all three checkpoint replays match saved records.")


if __name__ == "__main__":
    main()
