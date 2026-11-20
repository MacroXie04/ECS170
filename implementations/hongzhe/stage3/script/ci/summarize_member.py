"""Validate one fresh, full-data member run and emit JSON for its CI comment."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from script.ci.summarize_aileen import summarize as summarize_aileen
from script.ci.summarize_xiaoyu import summarize as summarize_xiaoyu
from script.ci.summarize_zizheng import summarize as summarize_zizheng

DATA_HASHES = {
    "train.csv": "fb60bc58af4dac3554e394af262b3184479833d3cc540ff8783f274b73492d5d",
    "test.csv": "51c292478d94ec3a01461bdfa82eb0885d262eb09e615679b2d69dedb6ad09e7",
}
EXPECTED = {
    "hongzhe": {"shallow_adam_ce", "deep_adam_ce", "deep_sgd_ce", "deep_adam_mse"},
    "aileen": {"tuned_mlp"},
    "zizheng": {"final", "baseline", "depth2", "depth3", "squared_error", "sgd", "dropout", "weight_decay"},
    "xiaoyu": {"baseline", "deeper", "sgd_lr_001", "mse_loss", "lower_lr"},
}


def verify_data(directory):
    for name, expected in DATA_HASHES.items():
        digest = hashlib.sha256()
        with (Path(directory) / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"{name} does not match the instructor dataset")
    labels = np.loadtxt(Path(directory) / "test.csv", delimiter=",", usecols=0, dtype=np.int64)
    if labels.shape != (10000,):
        raise ValueError("Expected 10,000 official test labels")
    return labels


def summarize_hongzhe(directory, true_y, log_path=None):
    path = ROOT / "implementations/hongzhe/script/ci/summarize_results.py"
    spec = importlib.util.spec_from_file_location("hongzhe_result_validation", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    validated = original.summarize(directory)
    document = json.loads((Path(directory) / "experiments.json").read_text())
    by_name = {entry["config"]["name"]: entry for entry in document["experiments"]}
    rows = []
    for row in validated["rows"]:
        name = row["name"]
        with np.load(Path(directory) / f"{name}_predictions.npz", allow_pickle=False) as predictions:
            if not np.array_equal(predictions["true_y"], true_y):
                raise ValueError(f"{name}: labels differ from the official test CSV")
        metrics = by_name[name]["metrics"]
        rows.append(dict(row, split="test", precision_macro=metrics["precision_macro"],
                         recall_macro=metrics["recall_macro"], epochs=15, seed=42, batch_size=256))
    return rows


def validate_rows(member, rows):
    if not isinstance(rows, list) or len(rows) != len(EXPECTED[member]):
        raise ValueError("Incomplete member results")
    if {row["name"] for row in rows} != EXPECTED[member]:
        raise ValueError("Unexpected or duplicate experiment names")
    for row in rows:
        split = "validation" if member == "zizheng" and row["name"] != "final" else "test"
        if row["split"] != split:
            raise ValueError("Validation and test results must be distinguished")
        for key in ("accuracy", "macro_f1", "precision_macro", "recall_macro"):
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid metric: {key}")
        expected_seed = 42 if member in ("hongzhe", "zizheng") else 2
        expected_batch = 256 if member in ("hongzhe", "xiaoyu") else 128
        if row["seed"] != expected_seed or row["batch_size"] != expected_batch:
            raise ValueError("Unexpected training protocol")
        epochs = row["epochs"]
        if type(epochs) is not int or not 1 <= epochs <= 25:
            raise ValueError("Invalid selected training duration")
        fixed_epochs = {"hongzhe": 15, "aileen": 25, "xiaoyu": 10}.get(member)
        if fixed_epochs is not None and epochs != fixed_epochs:
            raise ValueError("Incomplete training duration")


def summarize(member, directory, data_dir, log_path):
    if member not in EXPECTED:
        raise ValueError("Unknown implementation")
    if not Path(log_path).is_file() or Path(log_path).stat().st_size == 0:
        raise ValueError("Missing fresh training log")
    labels = verify_data(data_dir)
    readers = {"hongzhe": summarize_hongzhe, "aileen": summarize_aileen,
               "zizheng": summarize_zizheng, "xiaoyu": summarize_xiaoyu}
    rows = readers[member](Path(directory), labels, Path(log_path))
    validate_rows(member, rows)
    return {"implementation": member, "rows": rows, "train_count": 60000,
            "test_count": 10000, "data_sha256": DATA_HASHES,
            "environment": {"python": platform.python_version(), "pytorch": torch.__version__, "device": "cpu"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=EXPECTED, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = summarize(args.implementation, args.results_dir, args.data_dir, args.log)
        print(json.dumps(report, separators=(",", ":"), allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, EOFError) as error:
        parser.exit(1, f"Result validation failed: {error}\n")


if __name__ == "__main__":
    main()
