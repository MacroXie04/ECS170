"""Validation failures tested against immutable, saved experiment fixtures."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("summarize_zizheng", ROOT / "script/ci/summarize_zizheng.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
FIXTURE = ROOT / "implementations/zizheng/result/stage_2_result"


class ZizhengSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        # Structured fixture data is copied; no saved code, pickle, or PT is executed.
        for name in [*summary.CONFIGS, "final"]:
            folder = self.output / name
            folder.mkdir()
            for path in (FIXTURE / name).iterdir():
                if path.suffix in (".json", ".npz", ".csv"):
                    shutil.copyfile(path, folder / path.name)
            (folder / "model.pt").write_bytes(b"nonempty checkpoint fixture; never loaded")
        shutil.copyfile(FIXTURE / "selection.json", self.output / "selection.json")
        with np.load(self.output / "final/predictions.npz", allow_pickle=False) as archive:
            self.true_y = archive["test_true"]

    def read_json(self, relative):
        return json.loads((self.output / relative).read_text())

    def write_json(self, relative, value):
        (self.output / relative).write_text(json.dumps(value))

    def change_archive(self, relative, change):
        path = self.output / relative
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        change(arrays)
        np.savez_compressed(path, **arrays)

    def test_complete_fixture_returns_final_then_seven_validation_rows(self):
        rows = summary.summarize(self.output, self.true_y)
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["split"], "test")
        self.assertEqual(rows[0]["accuracy"], 0.9826)
        self.assertAlmostEqual(rows[0]["macro_f1"], 0.9824514021706579)
        self.assertEqual(rows[0]["epochs"], 8)
        self.assertEqual({row["name"] for row in rows[1:]}, set(summary.CONFIGS))
        self.assertTrue(all(row["split"] == "validation" for row in rows[1:]))
        dropout = next(row for row in rows if row["name"] == "dropout")
        self.assertEqual(dropout["epochs"], 8)  # selected epoch, not 13 completed epochs
        self.assertEqual(dropout["accuracy"], 0.9795)
        self.assertEqual(dropout["seed"], 42)
        self.assertEqual(dropout["batch_size"], 128)

    def test_rejects_incorrect_saved_metrics(self):
        record = self.read_json("baseline/record.json")
        record["validation"]["macro"]["f1"] = 0.5
        self.write_json("baseline/record.json", record)
        with self.assertRaisesRegex(ValueError, "macro f1"):
            summary.summarize(self.output, self.true_y)

    def test_rejects_wrong_confusion_matrix(self):
        record = self.read_json("final/record.json")
        record["test"]["confusion_matrix"][0][0] += 1
        self.write_json("final/record.json", record)
        with self.assertRaisesRegex(ValueError, "confusion matrix"):
            summary.summarize(self.output, self.true_y)

    def test_rejects_missing_candidate(self):
        shutil.rmtree(self.output / "weight_decay")
        with self.assertRaises(ValueError):
            summary.summarize(self.output, self.true_y)

    def test_rejects_duplicate_candidate_list(self):
        selection = self.read_json("selection.json")
        selection["candidates"][-1] = "baseline"
        self.write_json("selection.json", selection)
        with self.assertRaisesRegex(ValueError, "seven unique"):
            summary.summarize(self.output, self.true_y)

    def test_rejects_wrong_official_labels(self):
        wrong = self.true_y.copy()
        wrong[0] = (wrong[0] + 1) % 10
        with self.assertRaisesRegex(ValueError, "official test set"):
            summary.summarize(self.output, wrong)

    def test_rejects_wrong_prediction_size(self):
        self.change_archive("final/predictions.npz", lambda arrays: arrays.update(test_pred=arrays["test_pred"][:-1]))
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            summary.summarize(self.output, self.true_y)

    def test_rejects_noninteger_prediction_dtype(self):
        self.change_archive("baseline/predictions.npz", lambda arrays: arrays.update(validation_pred=arrays["validation_pred"].astype(float)))
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            summary.summarize(self.output, self.true_y)

    def test_rejects_empty_checkpoint(self):
        (self.output / "sgd/model.pt").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            summary.summarize(self.output, self.true_y)

    def test_rejects_partial_history_even_when_completed_count_is_changed(self):
        path = self.output / "dropout/history.csv"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n")
        record = self.read_json("dropout/record.json")
        record["completed_epochs"] -= 1
        self.write_json("dropout/record.json", record)
        with self.assertRaisesRegex(ValueError, "stopping rule"):
            summary.summarize(self.output, self.true_y)

    def test_rejects_misreported_selected_epoch(self):
        record = self.read_json("dropout/record.json")
        record["best_epoch"] = 9
        self.write_json("dropout/record.json", record)
        with self.assertRaisesRegex(ValueError, "Selected epoch"):
            summary.summarize(self.output, self.true_y)

    def test_rejects_nan_history_loss(self):
        path = self.output / "baseline/history.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        rows[0]["train_loss"] = "nan"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "Invalid train_loss"):
            summary.summarize(self.output, self.true_y)

    def test_rejects_overlap_in_split(self):
        def overlap(arrays):
            arrays["validation"][0] = arrays["train"][0]
        self.change_archive("baseline/split.npz", overlap)
        with self.assertRaisesRegex(ValueError, "overlapping or incomplete"):
            summary.summarize(self.output, self.true_y)

    def test_rejects_wrong_config(self):
        record = self.read_json("depth2/record.json")
        record["config"]["batch_size"] = 256
        self.write_json("depth2/record.json", record)
        with self.assertRaisesRegex(ValueError, "Unexpected configuration"):
            summary.summarize(self.output, self.true_y)

    def test_rejects_selection_that_is_not_validation_winner(self):
        selection = self.read_json("selection.json")
        selection["selected"] = "baseline"
        self.write_json("selection.json", selection)
        with self.assertRaisesRegex(ValueError, "validation winner"):
            summary.summarize(self.output, self.true_y)

    def test_rejects_changed_locked_selection(self):
        selection = self.read_json("final/locked_selection.json")
        selection["config"]["epochs"] = 9
        self.write_json("final/locked_selection.json", selection)
        with self.assertRaisesRegex(ValueError, "locked selection"):
            summary.summarize(self.output, self.true_y)

    def test_rejects_final_incomplete_refit(self):
        record = self.read_json("final/record.json")
        record["training_rows"] = 54000
        self.write_json("final/record.json", record)
        with self.assertRaisesRegex(ValueError, "training row count"):
            summary.summarize(self.output, self.true_y)


if __name__ == "__main__":
    unittest.main()
