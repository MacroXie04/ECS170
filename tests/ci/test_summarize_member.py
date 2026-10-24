from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.ci import summarize_member as module


class MemberSummaryTests(unittest.TestCase):
    def rows(self, member):
        return [dict(name=name, split="validation" if member == "zizheng" and name != "final" else "test",
                     accuracy=.9, macro_f1=.89, precision_macro=.9, recall_macro=.89,
                     epochs={"hongzhe": 15, "aileen": 25, "xiaoyu": 10, "zizheng": 8}[member],
                     seed=42 if member in ("hongzhe", "zizheng") else 2,
                     batch_size=256 if member in ("hongzhe", "xiaoyu") else 128)
                for name in module.EXPECTED[member]]

    def test_complete_member_reports_keep_validation_distinct_from_test(self):
        for member in module.EXPECTED:
            module.validate_rows(member, self.rows(member))
        rows = self.rows("zizheng")
        next(row for row in rows if row["name"] != "final")["split"] = "test"
        with self.assertRaises(ValueError):
            module.validate_rows("zizheng", rows)

    def test_partial_duplicate_and_nonfinite_reports_are_rejected(self):
        cases = [self.rows("hongzhe")[:-1], [self.rows("hongzhe")[0]] * 4]
        for value in (float("nan"), float("inf"), 1.1, -1, True):
            rows = self.rows("hongzhe")
            rows[0]["accuracy"] = value
            cases.append(rows)
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                module.validate_rows("hongzhe", rows)

    def test_changed_protocol_is_rejected(self):
        for key, value in (("epochs", 14), ("seed", 2), ("batch_size", 128)):
            rows = self.rows("hongzhe")
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.validate_rows("hongzhe", rows)

    def test_wrong_dataset_fails_before_results_are_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "train.csv").write_text("not the official dataset\n")
            (path / "training.log").write_text("training completed\n")
            with patch.object(module, "summarize_aileen", side_effect=AssertionError("must not read")):
                with self.assertRaisesRegex(ValueError, "instructor dataset"):
                    module.summarize("aileen", path, path, path / "training.log")


if __name__ == "__main__":
    unittest.main()
