"""Reject partial, stale, tampered, and unsafe Xiaoyu result fixtures."""
import csv
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from script.ci.summarize_xiaoyu import CONFIGS, summarize


class XiaoyuSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.labels = np.tile(np.arange(10, dtype=np.int64), 1000)
        summary = []
        for name, (hidden, optimizer, loss, lr) in CONFIGS.items():
            folder = self.root / name
            folder.mkdir()
            self.write_json(folder / "run_config.json", dict(run_name=name, epochs=10, hidden_dims=hidden,
                batch_size=256, optimizer=optimizer, loss=loss, learning_rate=lr,
                device="cpu", seed=2, train_limit=None, test_limit=None))
            metrics = {"accuracy": 1.0}
            metrics.update({f"{metric}_{average}": 1.0 for metric in ("precision", "recall", "f1")
                            for average in ("macro", "micro", "weighted")})
            self.write_json(folder / "metrics.json", metrics)
            with (folder / "predictions.pkl").open("wb") as handle:
                pickle.dump(dict(true_y=self.labels, pred_y=self.labels.copy()), handle)
            np.savetxt(folder / "confusion_matrix.csv", np.eye(10, dtype=int) * 1000, fmt="%d", delimiter=",")
            self.write_csv(folder / "training_history.csv", [dict(epoch=i, train_loss=1/i, train_accuracy=.9)
                                                           for i in range(1, 11)])
            (folder / "mlp_weights.pt").write_bytes(b"nonempty checkpoint fixture")
            (folder / "learning_curve.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
            (folder / "classification_report.txt").write_text("classification report fixture")
            summary.append(dict(experiment=name, hidden_dims=json.dumps(hidden), optimizer=optimizer,
                                loss=loss, learning_rate=lr, **metrics))
        self.write_csv(self.root / "experiments_summary.csv", summary)

    def write_json(self, path, value):
        path.write_text(json.dumps(value))

    def write_csv(self, path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def invalid(self):
        with self.assertRaises((ValueError, OSError, KeyError)):
            summarize(self.root, self.labels)

    def test_complete_results_return_five_verified_rows(self):
        rows = summarize(self.root, self.labels)
        self.assertEqual([r["name"] for r in rows], list(CONFIGS))
        for row in rows:
            self.assertEqual(row["accuracy"], 1)
            self.assertEqual(row["macro_f1"], 1)
            self.assertEqual((row["split"], row["epochs"], row["seed"], row["batch_size"]), ("test", 10, 2, 256))

    def test_missing_result_is_rejected(self):
        (self.root / "lower_lr" / "metrics.json").unlink()
        self.invalid()

    def test_unpredicted_classes_use_zero_precision_and_macro_f1(self):
        folder = self.root / "baseline"
        with (folder / "predictions.pkl").open("wb") as handle:
            pickle.dump(dict(true_y=self.labels, pred_y=np.zeros(10000, dtype=np.int64)), handle)
        cm = np.zeros((10, 10), dtype=int)
        cm[:, 0] = 1000
        np.savetxt(folder / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
        # Only class zero is predicted: precision=.1, recall=1, F1=2/11.
        metrics = dict(accuracy=.1, precision_macro=.01, recall_macro=.1, f1_macro=1/55,
                       precision_micro=.1, recall_micro=.1, f1_micro=.1,
                       precision_weighted=.01, recall_weighted=.1, f1_weighted=1/55)
        self.write_json(folder / "metrics.json", metrics)
        path = self.root / "experiments_summary.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0].update(metrics)
        self.write_csv(path, rows)
        row = summarize(self.root, self.labels)[0]
        self.assertAlmostEqual(row["accuracy"], .1)
        self.assertAlmostEqual(row["precision_macro"], .01)
        self.assertAlmostEqual(row["recall_macro"], .1)
        self.assertAlmostEqual(row["macro_f1"], 1/55)

    def test_partial_or_duplicate_summary_is_rejected(self):
        path = self.root / "experiments_summary.csv"
        original = path.read_text()
        path.write_text("\n".join(original.splitlines()[:-1]) + "\n")
        self.invalid()
        path.write_text(original.replace("lower_lr,", "baseline,"))
        self.invalid()

    def test_changed_configuration_is_rejected(self):
        path = self.root / "baseline" / "run_config.json"
        config = json.loads(path.read_text())
        for key, value in (("seed", 42), ("epochs", 1), ("train_limit", 100), ("device", "cuda")):
            with self.subTest(key=key):
                self.write_json(path, dict(config, **{key: value}))
                self.invalid()

    def test_incomplete_or_nonfinite_history_is_rejected(self):
        path = self.root / "baseline" / "training_history.csv"
        original = path.read_text()
        for content in ("\n".join(original.splitlines()[:-1]), original.replace("10,0.1,", "1,0.1,"),
                        original.replace("1,1.0,", "1,nan,"), original.replace(",0.9", ",inf")):
            with self.subTest(content=content[-30:]):
                path.write_text(content)
                self.invalid()

    def test_fake_macro_and_weighted_metrics_are_rejected(self):
        path = self.root / "baseline" / "metrics.json"
        metrics = json.loads(path.read_text())
        for key, value in (("accuracy", .99), ("f1_macro", .9), ("recall_weighted", .5), ("precision_micro", float("nan"))):
            with self.subTest(key=key):
                self.write_json(path, dict(metrics, **{key: value}))
                self.invalid()

    def test_changed_predictions_and_wrong_official_labels_are_rejected(self):
        path = self.root / "baseline" / "predictions.pkl"
        changed = self.labels.copy()
        changed[0] = 1
        for true, pred in ((self.labels, changed), (changed, self.labels)):
            with path.open("wb") as handle:
                pickle.dump(dict(true_y=true, pred_y=pred), handle)
            self.invalid()
        with self.assertRaises(ValueError):
            summarize(self.root, self.labels[:-1])

    def test_short_float_object_and_out_of_range_predictions_are_rejected(self):
        path = self.root / "baseline" / "predictions.pkl"
        for pred in (self.labels[:-1], self.labels.astype(float), self.labels.astype(object), self.labels + 1):
            with self.subTest(dtype=pred.dtype, count=len(pred)):
                with path.open("wb") as handle:
                    pickle.dump(dict(true_y=self.labels, pred_y=pred), handle)
                self.invalid()

    def test_tampered_confusion_or_summary_metrics_are_rejected(self):
        path = self.root / "baseline" / "confusion_matrix.csv"
        original = path.read_text()
        path.write_text(original.replace("1000", "999", 1))
        self.invalid()
        path.write_text(original)
        summary = self.root / "experiments_summary.csv"
        summary.write_text(summary.read_text().replace(",1.0,", ",0.5,", 1))
        self.invalid()

    def test_empty_checkpoint_or_missing_plot_is_rejected(self):
        path = self.root / "baseline" / "mlp_weights.pt"
        path.write_bytes(b"")
        self.invalid()
        path.write_bytes(b"fixture")
        (self.root / "baseline" / "learning_curve.png").unlink()
        self.invalid()

    def test_unsafe_pickle_and_trailing_data_are_rejected(self):
        path = self.root / "baseline" / "predictions.pkl"
        original = path.read_bytes()
        path.write_bytes(pickle.dumps(str))
        self.invalid()
        path.write_bytes(original + b"unexpected")
        self.invalid()

    def test_protocol_five_numpy_arrays_are_supported(self):
        for name in CONFIGS:
            with (self.root / name / "predictions.pkl").open("wb") as handle:
                pickle.dump(dict(true_y=self.labels, pred_y=self.labels.copy()), handle, protocol=5)
        self.assertEqual(len(summarize(self.root, self.labels)), 5)

    def test_log_rejects_reuse_missing_runs_and_wrong_epoch_counts(self):
        run = ("************ Stage 2 Start ************\n"
               "Training set: 60000 instances, 784 features\n"
               "Testing set:  10000 instances, 784 features\n")
        run += "".join(f"Epoch {i:>3}/10 | Loss 0.1 | Training accuracy 0.9\n" for i in range(1, 11))
        run += "************ Stage 2 Finish ************\n"
        path = self.root / "training.log"
        path.write_text(run * 5)
        self.assertEqual(len(summarize(self.root, self.labels, path)), 5)
        for text in (run * 4, run * 5 + "Reusing completed experiment: baseline\n", (run * 5).replace("/10 |", "/1 |", 1)):
            path.write_text(text)
            with self.assertRaises(ValueError):
                summarize(self.root, self.labels, path)


if __name__ == "__main__":
    unittest.main()
