"""Validate Xiaoyu's five freshly generated, full-dataset experiment outputs."""
import csv
import io
import json
import math
from pathlib import Path
import pickle
import re

import numpy as np


CONFIGS = {
    "baseline": ([128], "adam", "cross_entropy", 0.001),
    "deeper": ([256, 128], "adam", "cross_entropy", 0.001),
    "sgd_lr_001": ([128], "sgd", "cross_entropy", 0.01),
    "mse_loss": ([128], "adam", "mse", 0.001),
    "lower_lr": ([128], "adam", "cross_entropy", 0.0003),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class _ArrayUnpickler(pickle.Unpickler):
    """Allow only NumPy array reconstruction, never arbitrary pickle globals."""

    def find_class(self, module, name):
        if (module, name) == ("numpy", "ndarray"):
            return np.ndarray
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "_reconstruct":
            return np._core.multiarray._reconstruct
        if module in ("numpy.core.numeric", "numpy._core.numeric") and name == "_frombuffer":
            return np._core.numeric._frombuffer
        raise pickle.UnpicklingError(f"Forbidden pickle global: {module}.{name}")


def _labels(value, description):
    require(isinstance(value, np.ndarray), f"{description}: expected a NumPy array")
    require(value.shape == (10000,) and np.issubdtype(value.dtype, np.integer),
            f"{description}: expected 10,000 integer labels")
    require(bool(((value >= 0) & (value <= 9)).all()), f"{description}: labels must be in [0, 9]")
    return value


def _predictions(path):
    require(0 < path.stat().st_size <= 1024 * 1024, "Invalid prediction archive size")
    stream = io.BytesIO(path.read_bytes())
    try:
        result = _ArrayUnpickler(stream).load()
    except (pickle.UnpicklingError, EOFError, TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"Invalid prediction archive: {error}") from error
    require(stream.read(1) == b"", "Trailing data in prediction archive")
    require(type(result) is dict and set(result) == {"true_y", "pred_y"}, "Invalid prediction keys")
    return _labels(result["true_y"], "Saved true labels"), _labels(result["pred_y"], "Predicted labels")


def _metrics(cm):
    diagonal = cm.diagonal().astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    precision = np.divide(diagonal, predicted, out=np.zeros(10), where=predicted != 0)
    recall = np.divide(diagonal, support, out=np.zeros(10), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(10), where=precision + recall != 0)
    accuracy = float(diagonal.sum() / 10000)
    scores = {"accuracy": accuracy}
    for key, values in (("precision", precision), ("recall", recall), ("f1", f1)):
        scores[f"{key}_macro"] = float(values.mean())
        scores[f"{key}_micro"] = accuracy
        scores[f"{key}_weighted"] = float(np.dot(values, support) / 10000)
    return scores


def _check_metric(value, expected, description):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            f"{description}: invalid metric")
    require(math.isclose(value, expected, rel_tol=0, abs_tol=1e-12), f"{description}: metric mismatch")


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _check_log(path):
    log = Path(path).read_text(encoding="utf-8")
    require("Reusing completed experiment:" not in log, "Cached experiments are not fresh training")
    starts = "************ Stage 2 Start ************"
    finishes = "************ Stage 2 Finish ************"
    require(log.count(starts) == 5 and log.count(finishes) == 5, "Incomplete experiment log")
    require(log.count("Training set: 60000 instances, 784 features") == 5,
            "Missing full training dataset records")
    require(log.count("Testing set:  10000 instances, 784 features") == 5,
            "Missing full test dataset records")
    epochs = [(int(a), int(b)) for a, b in re.findall(r"^Epoch\s+(\d+)/(\d+)\s+\|", log, re.MULTILINE)]
    require(epochs == [(epoch, 10) for _ in CONFIGS for epoch in range(1, 11)],
            "Expected five complete 10-epoch training runs")


def summarize(directory, true_y, log_path=None):
    """Return validated test metrics; the caller guarantees a new output directory."""
    directory = Path(directory)
    official_y = _labels(np.asarray(true_y), "Official test labels")
    if log_path is not None:
        _check_log(log_path)
    summaries = _read_csv(directory / "experiments_summary.csv")
    require(len(summaries) == len(CONFIGS), "Expected all five summary rows")
    names = [row.get("experiment") for row in summaries]
    require(len(set(names)) == 5 and set(names) == set(CONFIGS), "Invalid or duplicate experiment names")
    summaries = {row["experiment"]: row for row in summaries}
    output = []
    for name, (hidden, optimizer, loss, learning_rate) in CONFIGS.items():
        folder = directory / name
        expected = dict(run_name=name, epochs=10, hidden_dims=hidden, batch_size=256,
                        optimizer=optimizer, loss=loss, learning_rate=learning_rate,
                        device="cpu", seed=2, train_limit=None, test_limit=None)
        config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
        require(config == expected, f"Unexpected configuration for {name}")
        history = _read_csv(folder / "training_history.csv")
        require(len(history) == 10, f"Incomplete history for {name}")
        for epoch, record in enumerate(history, 1):
            require(set(record) == {"epoch", "train_loss", "train_accuracy"}, f"Invalid history columns for {name}")
            require(int(record["epoch"]) == epoch, f"Invalid epoch sequence for {name}")
            train_loss, train_accuracy = float(record["train_loss"]), float(record["train_accuracy"])
            require(math.isfinite(train_loss) and train_loss >= 0, f"Invalid training loss for {name}")
            require(math.isfinite(train_accuracy) and 0 <= train_accuracy <= 1,
                    f"Invalid training accuracy for {name}")
        for filename in ("mlp_weights.pt", "learning_curve.png", "classification_report.txt"):
            path = folder / filename
            require(path.is_file() and path.stat().st_size > 0, f"Missing or empty {name}/{filename}")
        with (folder / "learning_curve.png").open("rb") as handle:
            require(handle.read(8) == b"\x89PNG\r\n\x1a\n", f"Invalid plot format for {name}")
        saved_y, pred_y = _predictions(folder / "predictions.pkl")
        require(np.array_equal(saved_y, official_y), f"Official test labels differ for {name}")
        cm = np.bincount(saved_y.astype(np.int64) * 10 + pred_y.astype(np.int64), minlength=100).reshape(10, 10)
        with (folder / "confusion_matrix.csv").open(newline="", encoding="utf-8") as handle:
            saved_cm = np.array([[int(value) for value in row] for row in csv.reader(handle)])
        require(saved_cm.shape == (10, 10) and np.array_equal(saved_cm, cm), f"Confusion matrix differs for {name}")
        calculated = _metrics(cm)
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        require(set(metrics) == set(calculated), f"Incomplete or unexpected metric keys for {name}")
        summary = summaries[name]
        require(json.loads(summary["hidden_dims"]) == hidden and summary["optimizer"] == optimizer
                and summary["loss"] == loss and float(summary["learning_rate"]) == learning_rate,
                f"Summary configuration differs for {name}")
        for key, value in calculated.items():
            _check_metric(metrics[key], value, f"{name}/{key}")
            _check_metric(float(summary[key]), value, f"Summary {name}/{key}")
        output.append(dict(name=name, split="test", accuracy=calculated["accuracy"],
                           macro_f1=calculated["f1_macro"], precision_macro=calculated["precision_macro"],
                           recall_macro=calculated["recall_macro"], epochs=10, seed=2, batch_size=256))
    return output
