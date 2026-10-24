"""Validate a fresh, complete Stage 2 run and emit a compact JSON summary."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np


DATA_HASHES = {
    "train.csv": "fb60bc58af4dac3554e394af262b3184479833d3cc540ff8783f274b73492d5d",
    "test.csv": "51c292478d94ec3a01461bdfa82eb0885d262eb09e615679b2d69dedb6ad09e7",
}
CONFIGS = {
    "shallow_adam_ce": ([128], "Adam", .001, "CE"),
    "deep_adam_ce": ([256, 128], "Adam", .001, "CE"),
    "deep_sgd_ce": ([256, 128], "SGD", .05, "CE"),
    "deep_adam_mse": ([256, 128], "Adam", .001, "MSE"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def summarize(directory):
    directory = Path(directory)
    document = json.loads((directory / "experiments.json").read_text())
    environment = document["environment"]
    require(environment["train_shape"] == [60000, 784], "Unexpected training shape")
    require(environment["test_shape"] == [10000, 784], "Unexpected test shape")
    require(environment["data_sha256"] == DATA_HASHES, "Dataset hashes do not match the course CSVs")
    require(environment["device"] == "cpu", "Expected CPU execution")
    for key in ("python", "pytorch"):
        require(isinstance(environment[key], str) and re.fullmatch(r"[0-9][0-9A-Za-z.+-]{0,63}", environment[key]), "Invalid environment version")
    experiments = document["experiments"]
    require(isinstance(experiments, list) and len(experiments) == 4, "Expected all four experiments")
    names = [experiment["config"]["name"] for experiment in experiments]
    require(len(set(names)) == 4 and set(names) == set(CONFIGS), "Expected four unique configurations")
    by_name = dict(zip(names, experiments))
    reference_labels = None
    rows = []
    for name, (hidden, optimizer, lr, loss) in CONFIGS.items():
        experiment = by_name[name]
        expected = dict(name=name, hidden=hidden, optimizer=optimizer, lr=lr, loss=loss,
                        seed=42, epochs=15, batch_size=256)
        require(experiment["config"] == expected, f"Unexpected configuration for {name}")
        history = experiment["history"]
        require(isinstance(history, list) and len(history) == 15, f"Incomplete training history for {name}")
        for epoch, item in enumerate(history, 1):
            require(type(item["epoch"]) is int and item["epoch"] == epoch, f"Invalid epoch sequence for {name}")
            require(finite_number(item["train_loss"]) and item["train_loss"] >= 0, f"Invalid training loss for {name}")
        checkpoint = directory / f"{name}.pt"
        require(checkpoint.is_file() and checkpoint.stat().st_size > 0, f"Missing checkpoint for {name}")
        with np.load(directory / f"{name}_predictions.npz", allow_pickle=False) as archive:
            true_y, pred_y = archive["true_y"], archive["pred_y"]
        for labels in (true_y, pred_y):
            require(labels.shape == (10000,) and np.issubdtype(labels.dtype, np.integer), f"Invalid prediction shape or type for {name}")
            require(bool(((labels >= 0) & (labels < 10)).all()), f"Invalid label for {name}")
        if reference_labels is None:
            reference_labels = true_y
        else:
            require(np.array_equal(true_y, reference_labels), "Experiments use different test labels")
        cm = np.bincount(true_y.astype(np.int64) * 10 + pred_y.astype(np.int64), minlength=100).reshape(10, 10)
        metrics = experiment["metrics"]
        saved_cm = np.asarray(metrics["confusion_matrix"])
        require(saved_cm.shape == (10, 10) and np.issubdtype(saved_cm.dtype, np.integer), f"Invalid confusion matrix for {name}")
        require(np.array_equal(cm, saved_cm), f"Predictions and confusion matrix differ for {name}")
        diagonal = cm.diagonal().astype(float)
        support, predicted = cm.sum(axis=1), cm.sum(axis=0)
        precision = np.divide(diagonal, predicted, out=np.zeros(10), where=predicted != 0)
        recall = np.divide(diagonal, support, out=np.zeros(10), where=support != 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(10), where=precision + recall != 0)
        calculated = {"accuracy": float(diagonal.sum() / 10000)}
        for metric, values in (("precision", precision), ("recall", recall), ("f1", f1)):
            calculated[f"{metric}_macro"] = float(values.mean())
            calculated[f"{metric}_weighted"] = float(np.dot(values, support) / 10000)
        for key, value in calculated.items():
            require(finite_number(metrics[key]) and 0 <= metrics[key] <= 1, f"Invalid {key} for {name}")
            require(math.isclose(metrics[key], value, rel_tol=0, abs_tol=1e-12), f"Saved {key} differs from predictions for {name}")
        rows.append({"name": name, "accuracy": calculated["accuracy"], "macro_f1": calculated["f1_macro"]})
    return {"rows": rows, "environment": {key: environment[key] for key in ("python", "pytorch", "device")},
            "train_count": 60000, "test_count": 10000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    try:
        report = summarize(args.results_dir)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Result validation failed: {error}\n")
    print(json.dumps(report, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
