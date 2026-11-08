"""Validate fresh Zizheng experiment artifacts without executing checkpoints."""
import csv
import json
import math
from pathlib import Path
import re

import numpy as np


BASE = dict(hidden=[128], dropout=0.0, loss="cross_entropy", optimizer="adam",
            lr=0.001, weight_decay=0.0, batch_size=128, epochs=25, patience=5, seed=42)
CONFIGS = {
    "baseline": dict(BASE),
    "depth2": dict(BASE, hidden=[256, 128]),
    "depth3": dict(BASE, hidden=[256, 128, 64]),
    "squared_error": dict(BASE, loss="squared_error"),
    "sgd": dict(BASE, optimizer="sgd", lr=0.05),
    "dropout": dict(BASE, hidden=[256, 128], dropout=0.2),
    "weight_decay": dict(BASE, hidden=[256, 128], weight_decay=0.0001),
}
RULE = "max validation accuracy, then macro F1, then name"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def close(actual, expected, description):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
            f"Mismatch in {description}")


def labels(values, count, description):
    values = np.asarray(values)
    require(values.shape == (count,) and np.issubdtype(values.dtype, np.integer),
            f"Invalid shape or dtype for {description}")
    require(bool(((values >= 0) & (values < 10)).all()), f"Invalid class in {description}")
    return values.astype(np.int64, copy=False)


def metrics(true_y, pred_y):
    cm = np.bincount(true_y * 10 + pred_y, minlength=100).reshape(10, 10)
    diagonal = cm.diagonal().astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    precision = np.divide(diagonal, predicted, out=np.zeros(10), where=predicted != 0)
    recall = np.divide(diagonal, support, out=np.zeros(10), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(10), where=precision + recall != 0)
    accuracy = float(diagonal.sum() / len(true_y))
    result = {"accuracy": accuracy, "confusion_matrix": cm, "macro": {}, "weighted": {}, "micro": {}}
    for name, values in (("precision", precision), ("recall", recall), ("f1", f1)):
        result["macro"][name] = float(values.mean())
        result["weighted"][name] = float(np.dot(values, support) / len(true_y))
        result["micro"][name] = accuracy
    return result


def verify_metrics(saved, calculated, description):
    close(saved["accuracy"], calculated["accuracy"], description + " accuracy")
    for average in ("macro", "weighted", "micro"):
        for name in ("precision", "recall", "f1"):
            close(saved[average][name], calculated[average][name], f"{description} {average} {name}")
    cm = np.asarray(saved["confusion_matrix"])
    require(cm.shape == (10, 10) and np.issubdtype(cm.dtype, np.integer)
            and np.array_equal(cm, calculated["confusion_matrix"]), f"Mismatch in {description} confusion matrix")


def load_predictions(folder, record, split, train_count, eval_count):
    with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
        train_true = labels(archive["train_true"], train_count, f"{folder.name} train labels")
        train_pred = labels(archive["train_pred"], train_count, f"{folder.name} train predictions")
        eval_true = labels(archive[split + "_true"], eval_count, f"{folder.name} {split} labels")
        eval_pred = labels(archive[split + "_pred"], eval_count, f"{folder.name} {split} predictions")
    train_metrics, eval_metrics = metrics(train_true, train_pred), metrics(eval_true, eval_pred)
    verify_metrics(record["train"], train_metrics, folder.name + " train")
    verify_metrics(record[split], eval_metrics, folder.name + " " + split)
    return train_true, eval_true, train_metrics, eval_metrics


def history(folder, record, train_metrics, eval_metrics=None):
    validation = eval_metrics is not None
    fields = ["epoch", "train_loss", "train_accuracy"] + (["val_loss", "val_accuracy"] if validation else [])
    with (folder / "history.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == fields, f"Invalid history columns: {folder.name}")
        rows = list(reader)
    require(1 <= len(rows) <= 25, f"Invalid history length: {folder.name}")
    best_loss, best_epoch, stale = float("inf"), 0, 0
    for epoch, row in enumerate(rows, 1):
        require(row["epoch"] == str(epoch) and None not in row, f"Invalid epoch sequence: {folder.name}")
        for name in fields[1:]:
            value = float(row[name])
            require(math.isfinite(value) and value >= 0 and ("accuracy" not in name or value <= 1),
                    f"Invalid {name}: {folder.name} epoch {epoch}")
        if validation:
            value = float(row["val_loss"])
            if value < best_loss - 1e-6:
                best_loss, best_epoch, stale = value, epoch, 0
            else:
                stale += 1
            require(stale < 5 or epoch == len(rows), f"Training continued beyond early stopping: {folder.name}")
        else:
            best_epoch = epoch
    require(type(record["best_epoch"]) is int and record["best_epoch"] == best_epoch,
            f"Selected epoch differs from history: {folder.name}")
    if validation:
        require(type(record["completed_epochs"]) is int and record["completed_epochs"] == len(rows),
                f"Incomplete history: {folder.name}")
        require(len(rows) == 25 or stale == 5, f"Run ended before its stopping rule: {folder.name}")
        close(float(rows[best_epoch - 1]["val_accuracy"]), eval_metrics["accuracy"], folder.name + " selected validation accuracy")
    else:
        require(len(rows) == record["config"]["epochs"], "Final training duration differs from locked selection")
    close(float(rows[best_epoch - 1]["train_accuracy"]), train_metrics["accuracy"], folder.name + " selected train accuracy")


def check_record(folder, record, name, phase, train_rows):
    require(record["name"] == name and record["phase"] == phase, f"Wrong run identity: {name}")
    require(record["input_dim"] == 784 and record["classes"] == list(range(10)), f"Wrong feature/classes: {name}")
    require(record["training_rows"] == train_rows, f"Wrong training row count: {name}")
    require(record["train_source"]["rows"] == 60000 and record["train_source"]["features"] == 784,
            f"Wrong training source shape: {name}")
    require(isinstance(record["source_sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", record["source_sha256"]),
            f"Invalid source hash: {name}")
    checkpoint = folder / "model.pt"
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0, f"Missing checkpoint: {name}")


def make_row(name, split, calculated, record):
    return {"name": name, "split": split, "accuracy": calculated["accuracy"],
            "macro_f1": calculated["macro"]["f1"], "precision_macro": calculated["macro"]["precision"],
            "recall_macro": calculated["macro"]["recall"], "epochs": record["best_epoch"],
            "seed": record["config"]["seed"], "batch_size": record["config"]["batch_size"]}


def summarize(directory, true_y, log_path=None):
    """Return final-test then seven validation rows; logs are not metric evidence."""
    try:
        return _summarize(Path(directory), labels(true_y, 10000, "official test labels"))
    except (OSError, KeyError, TypeError, OverflowError) as error:
        raise ValueError(f"Invalid or incomplete Zizheng results: {error}") from error


def _summarize(directory, official_test):
    selection = read_json(directory / "selection.json")
    candidates = selection["candidates"]
    require(isinstance(candidates, list) and len(candidates) == 7 and set(candidates) == set(CONFIGS),
            "Expected all seven unique candidate configurations")
    require(selection["rule"] == RULE, "Unexpected selection rule")
    records, evaluations, rows = {}, {}, []
    shared_split = None
    training_labels = None
    for name, config in CONFIGS.items():
        folder = directory / name
        record = read_json(folder / "record.json")
        check_record(folder, record, name, "validation", 54000)
        require(record["config"] == config, f"Unexpected configuration: {name}")
        require(record["validation_rows"] == 6000 and record["split_seed"] == 42, f"Wrong validation split: {name}")
        require(record["train_source"] == selection["train_source"]
                and record["source_sha256"] == selection["source_sha256"], f"Data or code sources differ: {name}")
        with np.load(folder / "split.npz", allow_pickle=False) as archive:
            train_idx, val_idx = archive["train"], archive["validation"]
        require(train_idx.shape == (54000,) and val_idx.shape == (6000,)
                and np.issubdtype(train_idx.dtype, np.integer) and np.issubdtype(val_idx.dtype, np.integer),
                f"Invalid split indices: {name}")
        require(np.array_equal(np.sort(np.concatenate((train_idx, val_idx))), np.arange(60000)),
                f"Split is overlapping or incomplete: {name}")
        if shared_split is None:
            shared_split = train_idx, val_idx
        else:
            require(np.array_equal(train_idx, shared_split[0]) and np.array_equal(val_idx, shared_split[1]),
                    f"Candidates used different splits: {name}")
        train_true, val_true, train_metrics, val_metrics = load_predictions(folder, record, "validation", 54000, 6000)
        reconstructed = np.empty(60000, dtype=np.int64)
        reconstructed[train_idx], reconstructed[val_idx] = train_true, val_true
        if training_labels is None:
            training_labels = reconstructed
        else:
            require(np.array_equal(training_labels, reconstructed), f"Candidates used different true labels: {name}")
        history(folder, record, train_metrics, val_metrics)
        records[name], evaluations[name] = record, val_metrics
        rows.append(make_row(name, "validation", val_metrics, record))
    selected = sorted(CONFIGS, key=lambda name: (-evaluations[name]["accuracy"], -evaluations[name]["macro"]["f1"], name))[0]
    require(selection["selected"] == selected, "Selected configuration is not the validation winner")
    expected_config = dict(CONFIGS[selected], epochs=records[selected]["best_epoch"])
    require(selection["config"] == expected_config and selection["classes"] == list(range(10)),
            "Selection does not match the winning configuration and epoch")
    close(selection["validation_accuracy"], evaluations[selected]["accuracy"], "selection validation accuracy")
    folder = directory / "final"
    require(read_json(folder / "locked_selection.json") == selection, "Final locked selection differs from selection")
    final = read_json(folder / "record.json")
    check_record(folder, final, "final", "final_test", 60000)
    require(final["selected"] == selected and final["config"] == expected_config, "Final configuration differs from selection")
    require(final["test_rows"] == 10000 and final["test_source"]["rows"] == 10000
            and final["test_source"]["features"] == 784, "Wrong final test shape")
    require(final["train_source"] == selection["train_source"] and final["source_sha256"] == selection["source_sha256"],
            "Final data or code differs from selection")
    train_true, test_true, train_metrics, test_metrics = load_predictions(folder, final, "test", 60000, 10000)
    require(np.array_equal(train_true, training_labels), "Final model did not use the same complete training labels")
    require(np.array_equal(test_true, official_test), "Final true labels differ from the official test set")
    history(folder, final, train_metrics)
    return [make_row("final", "test", test_metrics, final), *rows]
