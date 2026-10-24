"""Ensure malformed or partial runs cannot produce a successful PR report."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from script.ci.summarize_results import CONFIGS, DATA_HASHES, summarize


class SummarizeResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.labels = np.tile(np.arange(10, dtype=np.int64), 1000)
        self.document = {
            "environment": {"python": "3.13.12", "pytorch": "2.14.1+cpu", "device": "cpu",
                            "train_shape": [60000, 784], "test_shape": [10000, 784],
                            "data_sha256": copy.deepcopy(DATA_HASHES)},
            "experiments": [],
        }
        for name, (hidden, optimizer, lr, loss) in CONFIGS.items():
            self.document["experiments"].append({
                "config": dict(name=name, hidden=hidden, optimizer=optimizer, lr=lr, loss=loss,
                               seed=42, epochs=15, batch_size=256),
                "history": [{"epoch": epoch, "train_loss": .5} for epoch in range(1, 16)],
                "metrics": {"accuracy": 1., "precision_macro": 1., "recall_macro": 1., "f1_macro": 1.,
                            "precision_weighted": 1., "recall_weighted": 1., "f1_weighted": 1.,
                            "confusion_matrix": (np.eye(10, dtype=int) * 1000).tolist()},
            })
            (self.directory / f"{name}.pt").write_bytes(b"checkpoint fixture")
            np.savez_compressed(self.directory / f"{name}_predictions.npz", true_y=self.labels, pred_y=self.labels)
        self.write_document()

    def write_document(self):
        (self.directory / "experiments.json").write_text(json.dumps(self.document))

    def assert_invalid(self):
        self.write_document()
        with self.assertRaises((ValueError, KeyError, OSError)):
            summarize(self.directory)

    def test_complete_run_reports_fixed_names_and_recomputed_metrics(self):
        report = summarize(self.directory)
        self.assertEqual([row["name"] for row in report["rows"]], list(CONFIGS))
        self.assertTrue(all(row["accuracy"] == row["macro_f1"] == 1 for row in report["rows"]))
        self.assertEqual((report["train_count"], report["test_count"]), (60000, 10000))
        json.dumps(report, allow_nan=False)

    def test_partial_run_is_rejected(self):
        self.document["experiments"].pop()
        self.assert_invalid()

    def test_duplicate_configuration_is_rejected(self):
        self.document["experiments"][1] = copy.deepcopy(self.document["experiments"][0])
        self.assert_invalid()

    def test_wrong_dataset_is_rejected(self):
        self.document["environment"]["data_sha256"]["test.csv"] = "0" * 64
        self.assert_invalid()

    def test_changed_hyperparameters_are_rejected(self):
        self.document["experiments"][0]["config"]["epochs"] = 1
        self.assert_invalid()

    def test_incomplete_or_nonfinite_history_is_rejected(self):
        original = copy.deepcopy(self.document["experiments"][0]["history"])
        cases = [original[:-1], [dict(item, epoch=1) for item in original],
                 [dict(item, train_loss=float("nan")) for item in original]]
        for history in cases:
            with self.subTest(history=history[-1]):
                self.document["experiments"][0]["history"] = history
                self.assert_invalid()

    def test_saved_metrics_must_match_predictions(self):
        self.document["experiments"][0]["metrics"]["f1_macro"] = .9
        self.assert_invalid()

    def test_nonfinite_metrics_are_rejected(self):
        self.document["experiments"][0]["metrics"]["accuracy"] = float("nan")
        self.assert_invalid()

    def test_changed_predictions_cannot_reuse_old_metrics(self):
        predictions = self.labels.copy()
        predictions[0] = 1
        np.savez_compressed(self.directory / "shallow_adam_ce_predictions.npz", true_y=self.labels, pred_y=predictions)
        self.assert_invalid()

    def test_short_prediction_archive_is_rejected(self):
        np.savez_compressed(self.directory / "shallow_adam_ce_predictions.npz", true_y=self.labels[:-1], pred_y=self.labels[:-1])
        self.assert_invalid()

    def test_empty_checkpoint_is_rejected(self):
        (self.directory / "shallow_adam_ce.pt").write_bytes(b"")
        self.assert_invalid()


if __name__ == "__main__":
    unittest.main()
