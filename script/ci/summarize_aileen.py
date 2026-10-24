"""Validate a fresh Aileen run and derive its test metrics from saved labels.

The native implementation saves predictions and a plot, but no checkpoint or
machine-readable epoch history. This validator checks the available predictions
and printed 25-epoch history; it does not claim to reproduce model inference.
"""
import io
from pathlib import Path
import pickle
import re

import numpy as np


PREDICTIONS = "MLP_tuned_prediction_result_None"
PLOT = "convergence_curve_tuned.png"
TEST_ROWS = 10000
EPOCHS = 25
MAX_PICKLE_BYTES = 1024 * 1024
MAX_LOG_BYTES = 10 * 1024 * 1024
EPOCH_PATTERN = re.compile(
    r"Epoch: ([0-9]+)  Loss: (\S+)  Train Acc: (\S+)"
)


class _NumpyUnpickler(pickle.Unpickler):
    """Allow only the NumPy array constructors used by native numeric arrays."""

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


def _load_predictions(path):
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_PICKLE_BYTES:
        raise ValueError("Missing, empty, or oversized prediction file")
    data = path.read_bytes()
    stream = io.BytesIO(data)
    try:
        result = _NumpyUnpickler(stream).load()
    except (pickle.UnpicklingError, EOFError, ValueError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("Invalid or unsupported prediction pickle") from error
    if stream.read(1):
        raise ValueError("Unexpected trailing data in prediction file")
    if type(result) is not dict or set(result) != {"true_y", "pred_y"}:
        raise ValueError("Prediction file must contain exactly true_y and pred_y")
    return result


def _labels(values, name, require_array=False):
    if require_array and type(values) is not np.ndarray:
        raise ValueError(f"{name} must be a NumPy array")
    array = np.asarray(values)
    if array.shape != (TEST_ROWS,) or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain exactly 10,000 integer labels")
    if np.any(array < 0) or np.any(array > 9):
        raise ValueError(f"{name} contains invalid class labels")
    return array.astype(np.int64, copy=False)


def _validate_log(path):
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_LOG_BYTES:
        raise ValueError("Missing, empty, or oversized training log")
    text = path.read_text(encoding="utf-8")
    epochs = []
    for line in text.splitlines():
        if not line.startswith("Epoch:"):
            continue
        match = EPOCH_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError("Malformed epoch entry or unexpected validation mode")
        epoch, loss, accuracy = match.groups()
        try:
            loss, accuracy = float(loss), float(accuracy)
        except ValueError as error:
            raise ValueError("Non-numeric training history") from error
        if not np.isfinite([loss, accuracy]).all() or loss < 0 or not 0 <= accuracy <= 1:
            raise ValueError("Non-finite or invalid training history")
        epochs.append(int(epoch))
    if epochs != list(range(1, EPOCHS + 1)):
        raise ValueError("Expected exactly one complete, ordered 25-epoch training run")
    if "method running on cpu | trainable parameters: 537354" not in text:
        raise ValueError("Missing expected tuned-model CPU/parameter confirmation")
    if "************ Finish ************" not in text:
        raise ValueError("Native training script did not reach its completion marker")


def summarize(directory, true_y, log_path):
    """Return one validated test-result row; true_y must be official test labels.

    The caller is responsible for verifying the official CSV checksum and that
    directory/log_path come from this fresh workflow run. The fixed epoch, seed,
    and batch-size metadata correspond to the preserved native tuned script.
    """
    directory, log_path = Path(directory), Path(log_path)
    expected = _labels(true_y, "Official test labels")
    saved = _load_predictions(directory / PREDICTIONS)
    actual = _labels(saved["true_y"], "Saved true_y", require_array=True)
    predicted = _labels(saved["pred_y"], "Saved pred_y", require_array=True)
    if not np.array_equal(actual, expected):
        raise ValueError("Saved true_y does not match the official test labels in order")
    _validate_log(log_path)
    plot = directory / PLOT
    if not plot.is_file() or plot.stat().st_size <= 8:
        raise ValueError("Missing or empty convergence plot")
    with plot.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Convergence plot is not a PNG")

    confusion = np.bincount(10 * actual + predicted, minlength=100).reshape(10, 10)
    correct = confusion.diagonal()
    support, predicted_counts = confusion.sum(axis=1), confusion.sum(axis=0)
    precision = np.divide(correct, predicted_counts, out=np.zeros(10), where=predicted_counts != 0)
    recall = np.divide(correct, support, out=np.zeros(10), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(10), where=(precision + recall) != 0)
    return [{"name": "tuned_mlp", "split": "test",
             "accuracy": float(correct.sum() / TEST_ROWS),
             "macro_f1": float(f1.mean()), "precision_macro": float(precision.mean()),
             "recall_macro": float(recall.mean()), "epochs": EPOCHS,
             "seed": 2, "batch_size": 128}]
