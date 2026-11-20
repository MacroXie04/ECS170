"""Synthetic fixtures for Aileen CI validation; never load archived pickles."""
import base64
import importlib.util
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("summarize_aileen", REPO / "script/ci/summarize_aileen.py")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class AileenSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.true = np.tile(np.arange(10, dtype=np.int64), 1000)
        self.pred = self.true.copy()
        # Exactly 100 errors from class 0 into class 1, allowing analytic PRF.
        self.pred[np.flatnonzero(self.true == 0)[:100]] = 1
        self.predictions = self.directory / validator.PREDICTIONS
        self.write_predictions()
        self.log = self.directory / "training.log"
        self.log.write_text(self.valid_log(), encoding="utf-8")
        self.plot = self.directory / validator.PLOT
        self.plot.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aKXYAAAAASUVORK5CYII="))

    def valid_log(self):
        return ("method running on cpu | trainable parameters: 537354\n"
                + "".join(f"Epoch: {epoch}  Loss: 0.5100  Train Acc: 0.9990\n"
                          for epoch in range(1, 26))
                + "************ Finish ************\n")

    def write_predictions(self, true=None, pred=None, protocol=pickle.DEFAULT_PROTOCOL):
        self.predictions.write_bytes(pickle.dumps({
            "true_y": self.true if true is None else true,
            "pred_y": self.pred if pred is None else pred,
        }, protocol=protocol))

    def summarize(self):
        return validator.summarize(self.directory, self.true, self.log)

    def test_derives_metrics_from_arrays_for_numpy_pickle_protocols(self):
        for protocol in (4, 5):
            with self.subTest(protocol=protocol):
                self.write_predictions(protocol=protocol)
                row, = self.summarize()
                self.assertEqual(row["accuracy"], .99)
                self.assertAlmostEqual(row["precision_macro"], (9 + 1000 / 1100) / 10)
                self.assertAlmostEqual(row["recall_macro"], .99)
                self.assertAlmostEqual(row["macro_f1"], (8 + 1800 / 1900 + 2000 / 2100) / 10)
                self.assertEqual((row["split"], row["epochs"], row["seed"], row["batch_size"]),
                                 ("test", 25, 2, 128))
        self.write_predictions(pred=self.true)
        self.assertEqual(self.summarize()[0]["accuracy"], 1)

    def test_rejects_wrong_label_order_and_shape(self):
        for true in (np.roll(self.true, 1), self.true[:-1], self.true.reshape(1000, 10)):
            with self.subTest(shape=true.shape):
                self.write_predictions(true=true)
                with self.assertRaises(ValueError):
                    self.summarize()

    def test_rejects_invalid_predictions(self):
        for pred in (self.pred[:-1], self.pred.astype(float),
                     np.full(10000, np.nan), np.full(10000, 10),
                     np.full(10000, -1), np.full(10000, True),
                     self.pred.astype(object)):
            with self.subTest(dtype=pred.dtype, shape=pred.shape):
                self.write_predictions(pred=pred)
                with self.assertRaises(ValueError):
                    self.summarize()

    def test_rejects_missing_and_truncated_prediction_files(self):
        self.predictions.unlink()
        with self.assertRaises(ValueError):
            self.summarize()
        for invalid in (b"", b"not a pickle", pickle.dumps({"pred_y": self.pred}),
                        pickle.dumps({"true_y": self.true, "pred_y": self.pred})[:-5]):
            self.predictions.write_bytes(invalid)
            with self.assertRaises(ValueError):
                self.summarize()

    def test_rejects_trailing_pickle_and_unauthorized_global(self):
        self.predictions.write_bytes(self.predictions.read_bytes() + pickle.dumps(1))
        with self.assertRaises(ValueError):
            self.summarize()

        class Forbidden:
            def __reduce__(self):
                # A harmless callable used to ensure arbitrary globals are denied.
                return str, ("not a NumPy array",)

        self.predictions.write_bytes(pickle.dumps(Forbidden()))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.summarize()

    def test_rejects_partial_duplicate_out_of_order_and_nonfinite_epochs(self):
        text = self.valid_log()
        cases = [
            text.replace("Epoch: 25  Loss: 0.5100  Train Acc: 0.9990\n", ""),
            text + "Epoch: 25  Loss: 0.5100  Train Acc: 0.9990\n",
            text.replace("Epoch: 1 ", "Epoch: 2 ", 1),
            text.replace("Loss: 0.5100", "Loss: nan", 1),
            text.replace("Loss: 0.5100", "Loss: inf", 1),
            text.replace("Loss: 0.5100", "Loss: -0.1", 1),
            text.replace("Train Acc: 0.9990", "Train Acc: nan", 1),
            text.replace("Train Acc: 0.9990", "Train Acc: 1.1", 1),
            text.replace("************ Finish ************", ""),
            text.replace("parameters: 537354", "parameters: 123"),
        ]
        for invalid in cases:
            with self.subTest(log=invalid[:80]):
                self.log.write_text(invalid, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.summarize()

    def test_rejects_missing_log_and_missing_or_empty_plot(self):
        self.log.unlink()
        with self.assertRaises(ValueError):
            self.summarize()
        self.log.write_text(self.valid_log(), encoding="utf-8")
        self.plot.unlink()
        with self.assertRaises(ValueError):
            self.summarize()
        for content in (b"", b"not a png at all"):
            self.plot.write_bytes(content)
            with self.assertRaises(ValueError):
                self.summarize()


if __name__ == "__main__":
    unittest.main()
