"""Validation-only IMDb classifier search and explicitly locked final evaluation.

This entry point never reads the official test split during training. A final
evaluation requires a selection manifest written after comparing validation
runs. The original Stage 4 runner and generation code are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader, truncate_tokens
from local_code.stage_4_code.Experiment_Runner import collate, make_curve, run_epoch, save_checkpoint, write_json
from local_code.stage_4_code.Method_Classifier import ClassifierModel

DEFAULT_ROOT = REPO / "result/stage_4_result/classification_optimization_20261117"
SPLIT_SEED = 1704


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    names = [
        "local_code/stage_4_code/Dataset_Loader.py",
        "local_code/stage_4_code/Method_Classifier.py",
        "local_code/stage_4_code/Experiment_Runner.py",
        "script/stage_4_script/train_classification.py",
    ]
    return {name: digest_file(REPO / name) for name in names}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def data_loader(data_dir: str, config: dict, include_test: bool) -> dict:
    loader = Dataset_Loader(
        data_dir, seed=SPLIT_SEED, max_vocab=config["max_vocab"],
        max_length=config["max_length"],
    )
    return loader.load_classification(
        include_test=include_test, truncation_mode=config["truncation_mode"],
        archive_path=config.get("archive_path")
    )


def metrics_and_rows(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, list[dict]]:
    model.eval()
    rows = []
    with torch.no_grad():
        for tokens, lengths, labels, ids in loader:
            logits = model(tokens.to(device), lengths.to(device))
            losses = nn.functional.cross_entropy(logits, labels.to(device), reduction="none")
            probabilities = logits.softmax(-1)[:, 1]
            predictions = logits.argmax(-1)
            rows.extend(
                {
                    "id": identifier,
                    "label": int(label),
                    "prediction": int(prediction),
                    "positive_probability": float(probability),
                    "nll": float(loss),
                }
                for identifier, label, prediction, probability, loss in zip(
                    ids, labels.tolist(), predictions.cpu().tolist(),
                    probabilities.cpu().tolist(), losses.cpu().tolist()
                )
            )
    truth = [row["label"] for row in rows]
    predictions = [row["prediction"] for row in rows]
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, average="macro", zero_division=0
    )
    metrics = {
        "n_examples": len(rows),
        "loss": float(np.mean([row["nll"] for row in rows])),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
    }
    return metrics, rows


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No prediction rows")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def config_for_train(args: argparse.Namespace) -> dict:
    model = {
        "vocab_size": None,
        "cell": args.cell,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.layers,
        "dropout": args.dropout,
        "bidirectional": args.bidirectional,
        "pooling": args.pooling,
        "pad_id": 0,
    }
    return {
        "tag": args.tag,
        "data_dir": str((REPO / args.data_dir).resolve()),
        "archive_path": str((REPO / args.archive).resolve()) if args.archive else None,
        "archive_sha256": digest_file((REPO / args.archive).resolve()) if args.archive else None,
        "split_seed": SPLIT_SEED,
        "model_seed": args.model_seed,
        "model": model,
        "max_vocab": args.max_vocab,
        "max_length": args.max_length,
        "truncation_mode": args.truncation,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "gradient_clip_norm": 1.0,
        "checkpoint_selection": "lowest validation cross-entropy; earliest tie",
        "source_sha256": source_hashes(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
    }


def train(args: argparse.Namespace) -> None:
    start_total = time.monotonic()
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    config = config_for_train(args)
    data = data_loader(args.data_dir, config, include_test=False)
    if "test" in data or "test" in data["metadata"]["counts"]:
        raise AssertionError("Training loader must not include the official test split")
    config["model"]["vocab_size"] = len(data["vocab"])
    config["data_fingerprint"] = {
        "split_id_sha256": data["metadata"]["split_id_sha256"],
        "vocabulary_sha256": data["metadata"]["vocabulary_sha256"],
        "source_sha256_train_only": data["metadata"]["source_sha256"],
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output = Path(args.root) / args.tag
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError(f"Existing run configuration differs: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    if (output / "summary.json").exists():
        print(f"SKIP completed {args.tag}", flush=True)
        return
    write_json(output / "data_metadata.json", data["metadata"])
    write_json(output / "environment.json", {
        "python": sys.version,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device": str(device),
        "command": sys.argv,
        "source_sha256": config["source_sha256"],
        "git_head": config["git_head"],
    })
    set_seed(args.model_seed)
    model = ClassifierModel(**config["model"]).to(device)
    optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[args.optimizer]
    optimizer = optimizer_type(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
        )
        if args.scheduler else None
    )
    generator = torch.Generator().manual_seed(args.model_seed)
    loaders = {
        split: DataLoader(
            data[split], batch_size=args.batch_size, shuffle=(split == "train"),
            generator=generator if split == "train" else None, collate_fn=collate,
            num_workers=0, pin_memory=(device.type == "cuda")
        )
        for split in ("train", "validation")
    }
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history, best_loss, best_epoch, seconds = [], float("inf"), 0, 0.0
    last_path = checkpoint_dir / "last.pt"
    if (checkpoint_dir / "best.pt").exists() and not last_path.exists():
        raise RuntimeError("Incomplete run has best.pt but no last.pt; inspect it before retrying")
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=True)
        if state["config_hash"] != config_hash:
            raise ValueError("Resume checkpoint configuration mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        history, best_loss, best_epoch, seconds = (
            state["history"], state["best_loss"], state["best_epoch"], state["seconds"]
        )
        torch.set_rng_state(state["rng"].cpu())
        generator.set_state(state["loader_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        print(f"RESUME {args.tag} epoch={len(history)}", flush=True)
    print(f"START {args.tag} params={model.parameter_count()} device={device}", flush=True)
    for epoch in range(len(history) + 1, args.max_epochs + 1):
        if history and len(history) - best_epoch >= args.patience:
            break
        epoch_start = time.monotonic()
        train_stats = run_epoch(model, loaders["train"], "classification", device, optimizer)
        val_stats = run_epoch(model, loaders["validation"], "classification", device)
        elapsed = time.monotonic() - epoch_start
        seconds += elapsed
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "validation_loss": val_stats["loss"],
            "validation_accuracy": val_stats["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
        }
        history.append(row)
        if val_stats["loss"] < best_loss:
            best_loss, best_epoch = val_stats["loss"], epoch
            save_checkpoint(checkpoint_dir / "best.pt", {
                "model": model.state_dict(),
                "model_kwargs": config["model"],
                "config_hash": config_hash,
                "epoch": epoch,
                "vocab": data["vocab"].to_dict(),
                "validation_loss": best_loss,
                "validation_accuracy": val_stats["accuracy"],
            })
        if scheduler is not None:
            scheduler.step(val_stats["loss"])
        write_json(output / "history.json", history)
        save_checkpoint(last_path, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else {},
            "config_hash": config_hash, "history": history, "best_loss": best_loss,
            "best_epoch": best_epoch, "seconds": seconds,
            "rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        })
        print(json.dumps({"tag": args.tag, **row, "best_epoch": best_epoch}), flush=True)
    best = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model"])
    val_metrics, val_rows = metrics_and_rows(model, loaders["validation"], device)
    if abs(val_metrics["loss"] - best_loss) > 1e-5:
        raise AssertionError("Best checkpoint did not restore the selected validation loss")
    write_rows(output / "validation_predictions.csv", val_rows)
    make_curve(history, output / "learning_curve.png", args.tag)
    summary = {
        "tag": args.tag,
        "config_hash": config_hash,
        "best_checkpoint_sha256": digest_file(checkpoint_dir / "best.pt"),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "parameters": model.parameter_count(),
        "validation": val_metrics,
        "train_seconds": seconds,
        "total_seconds": time.monotonic() - start_total,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "split_id_sha256": data["metadata"]["split_id_sha256"],
        "vocabulary_sha256": data["metadata"]["vocabulary_sha256"],
        "truncated_counts": data["metadata"]["truncated_counts"],
    }
    write_json(output / "summary.json", summary)
    print("COMPLETE " + json.dumps(summary), flush=True)


def load_locked(selection_path: Path) -> tuple[dict, dict, Path]:
    selection = json.loads(selection_path.read_text())
    required = ["selected_run", "config_hash", "best_checkpoint_sha256", "selection_rule", "locked_before_test"]
    if any(not selection.get(field) for field in required):
        raise ValueError(f"Selection manifest needs {required}")
    run = selection_path.parent / selection["selected_run"]
    if run.resolve().parent != selection_path.parent.resolve():
        raise ValueError("Selected run must be in the manifest's result directory")
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    if summary["config_hash"] != selection["config_hash"]:
        raise ValueError("Locked configuration hash mismatch")
    if digest_file(run / "checkpoints/best.pt") != selection["best_checkpoint_sha256"]:
        raise ValueError("Locked checkpoint hash mismatch")
    current_sources = source_hashes()
    for name in ("local_code/stage_4_code/Dataset_Loader.py",
                 "local_code/stage_4_code/Method_Classifier.py",
                 "local_code/stage_4_code/Experiment_Runner.py"):
        if current_sources[name] != config["source_sha256"][name]:
            raise ValueError(f"Source changed since training: {name}")
    return config, summary, run


def locked_model(config: dict, run: Path, device: torch.device) -> tuple[ClassifierModel, dict]:
    state = torch.load(run / "checkpoints/best.pt", map_location=device, weights_only=True)
    model = ClassifierModel(**config["model"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state["vocab"]


def final_eval(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    selection_path = Path(args.selection).resolve()
    config, summary, run = load_locked(selection_path)
    output = selection_path.parent / "final_test"
    if (output / "metrics.json").exists():
        existing = json.loads((output / "metrics.json").read_text())
        if (existing.get("selected_run") != run.name or
            existing.get("config_hash") != summary["config_hash"] or
            existing.get("checkpoint_sha256") != summary["best_checkpoint_sha256"]):
            raise ValueError("Existing test result belongs to a different selection; refusing to overwrite")
        print(f"SKIP existing locked test evaluation {output}", flush=True)
        return
    device = choose_device(args.device)
    data = data_loader(config["data_dir"], config, include_test=True)
    for split in ("train", "validation"):
        if data["metadata"]["split_id_sha256"][split] != summary["split_id_sha256"][split]:
            raise ValueError(f"{split} split differs from locked run")
    if data["metadata"]["vocabulary_sha256"] != summary["vocabulary_sha256"]:
        raise ValueError("Vocabulary differs from locked run")
    baseline = json.loads((REPO / "result/stage_4_result/runs/classification_metadata.json").read_text())
    if len(data["test"]) != 25000 or data["metadata"]["label_counts"]["test"] != {"0": 12500, "1": 12500}:
        raise ValueError("Official test is incomplete or unbalanced")
    if (data["metadata"]["split_id_sha256"]["test"] != baseline["split_id_sha256"]["test"] or
        data["metadata"]["source_sha256"] != baseline["source_sha256"]):
        raise ValueError("Official test IDs or source contents differ from saved baseline")
    model, vocab_data = locked_model(config, run, device)
    if vocab_data != data["vocab"].to_dict():
        raise ValueError("Checkpoint vocabulary differs from official data loader")
    test_loader = DataLoader(
        data["test"], batch_size=config["batch_size"], shuffle=False,
        collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda")
    )
    started = time.monotonic()
    test_metrics, rows = metrics_and_rows(model, test_loader, device)
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "predictions.csv", rows)
    write_json(output / "metrics.json", {
        "selected_run": run.name,
        "config_hash": summary["config_hash"],
        "checkpoint_sha256": summary["best_checkpoint_sha256"],
        "official_test": test_metrics,
        "delta_accuracy_percentage_points_vs_gru_87_184": (
            test_metrics["accuracy"] * 100 - 87.184
        ),
        "evaluation_seconds": time.monotonic() - started,
        "test_split_id_sha256": data["metadata"]["split_id_sha256"]["test"],
        "n_test_examples": len(data["test"]),
    })
    print("FINAL_TEST " + json.dumps(test_metrics), flush=True)


def predict(args: argparse.Namespace) -> None:
    from local_code.stage_4_code.Dataset_Loader import Vocabulary
    config, _, run = load_locked(Path(args.selection).resolve())
    device = choose_device(args.device)
    model, vocab_data = locked_model(config, run, device)
    vocab = Vocabulary.from_dict(vocab_data)
    ids = truncate_tokens(vocab.encode(args.text), config["max_length"], config["truncation_mode"]) or [vocab.UNK]
    tokens = torch.tensor([ids], device=device, dtype=torch.long)
    lengths = torch.tensor([len(ids)], device=device)
    with torch.no_grad():
        probability = float(model(tokens, lengths).softmax(-1)[0, 1])
    print(json.dumps({
        "label": "positive" if probability >= 0.5 else "negative",
        "positive_probability": probability,
        "selected_run": run.name,
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--tag", required=True)
    p_train.add_argument("--root", default=str(DEFAULT_ROOT))
    p_train.add_argument("--data-dir", default="data/stage_4_data")
    p_train.add_argument("--archive", default="data/stage_4_data.zip",
                         help="Instructor archive; empty string uses extracted files")
    p_train.add_argument("--model-seed", type=int, default=1704)
    p_train.add_argument("--cell", choices=["rnn", "lstm", "gru"], default="gru")
    p_train.add_argument("--bidirectional", action="store_true")
    p_train.add_argument("--pooling", choices=["mean", "max", "meanmax"], default="mean")
    p_train.add_argument("--embedding-dim", type=int, default=128)
    p_train.add_argument("--hidden-dim", type=int, default=128)
    p_train.add_argument("--layers", type=int, default=1)
    p_train.add_argument("--dropout", type=float, default=0.3)
    p_train.add_argument("--max-vocab", type=int, default=20000)
    p_train.add_argument("--max-length", type=int, default=256)
    p_train.add_argument("--truncation", choices=["head", "head_tail"], default="head")
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--max-epochs", type=int, default=30)
    p_train.add_argument("--patience", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=0.001)
    p_train.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    p_train.add_argument("--weight-decay", type=float, default=0.0)
    p_train.add_argument("--scheduler", action="store_true")
    p_train.add_argument("--device", default="auto")
    p_train.add_argument("--threads", type=int, default=8)
    p_final = sub.add_parser("final-eval")
    p_final.add_argument("--selection", default=str(DEFAULT_ROOT / "selection.json"))
    p_final.add_argument("--device", default="auto")
    p_final.add_argument("--threads", type=int, default=8)
    p_predict = sub.add_parser("predict")
    p_predict.add_argument("--selection", default=str(DEFAULT_ROOT / "selection.json"))
    p_predict.add_argument("--text", required=True)
    p_predict.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "train":
        if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1:
            parser.error("Epochs, patience, and batch size must be positive")
        train(args)
    elif args.command == "final-eval":
        final_eval(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
