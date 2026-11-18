"""Focused classification loader checks for validation-only optimization."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader


class ClassificationOptimizationLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for split, count in (("train", 20), ("test", 3)):
            for label in ("neg", "pos"):
                directory = self.root / "text_classification" / split / label
                directory.mkdir(parents=True)
                for index in range(count):
                    text = (f"opening {label} {split} {index} one two three four five "
                            f"six seven eight closing {label}")
                    (directory / f"{index}.txt").write_text(text, encoding="utf-8")

    def test_train_only_keeps_split_and_vocabulary_without_test_files(self):
        full = Dataset_Loader(self.root, seed=1704, max_length=6).load_classification()
        # Removing test files makes accidental test access fail, while the same
        # official-training data must produce the original split and vocabulary.
        shutil.rmtree(self.root / "text_classification" / "test")
        train_only = Dataset_Loader(self.root, seed=1704, max_length=6).load_classification(
            include_test=False)
        self.assertNotIn("test", train_only)
        self.assertEqual(full["metadata"]["split_id_sha256"]["train"],
                         train_only["metadata"]["split_id_sha256"]["train"])
        self.assertEqual(full["metadata"]["split_id_sha256"]["validation"],
                         train_only["metadata"]["split_id_sha256"]["validation"])
        self.assertEqual(full["metadata"]["vocabulary_sha256"],
                         train_only["metadata"]["vocabulary_sha256"])
        self.assertEqual(full["vocab"].tokens, train_only["vocab"].tokens)
        self.assertEqual(train_only["metadata"]["id_overlap_counts"],
                         {"train_validation": 0})
        self.assertFalse(train_only["metadata"]["official_test_loaded"])
        self.assertEqual(train_only["metadata"]["source_hash_scope"], "official_train_only")

    def test_head_tail_preserves_ends_and_default_head(self):
        loader = Dataset_Loader(self.root, seed=1704, max_length=5)
        default = loader.load_classification()
        head = loader.load_classification(include_test=True, truncation_mode="head")
        tail = loader.load_classification(include_test=False, truncation_mode="head_tail")
        self.assertEqual(default["metadata"], head["metadata"])
        self.assertEqual(default["metadata"]["split_id_sha256"]["train"],
                         tail["metadata"]["split_id_sha256"]["train"])
        self.assertEqual(default["metadata"]["split_id_sha256"]["validation"],
                         tail["metadata"]["split_id_sha256"]["validation"])
        self.assertEqual(default["metadata"]["vocabulary_sha256"],
                         tail["metadata"]["vocabulary_sha256"])
        for split in ("train", "validation"):
            head_rows = {row["id"]: row for row in head[split]}
            for row in tail[split]:
                ids = tail["vocab"].encode(row["text"])
                self.assertGreater(len(ids), 5)
                self.assertEqual(head_rows[row["id"]]["tokens"], ids[:5])
                self.assertEqual(row["tokens"], ids[:3] + ids[-2:])
        self.assertEqual(tail["metadata"]["truncation_mode"], "head_tail")

    def test_zip_matches_extracted_data_and_skips_test_members(self):
        extracted = Dataset_Loader(self.root, seed=1704, max_length=6).load_classification()
        archive_path = self.root / "course.zip"
        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
            for path in sorted((self.root / "text_classification").rglob("*.txt")):
                archive.write(path, "stage_4_data/" + path.relative_to(self.root).as_posix())
        zipped = Dataset_Loader(self.root, seed=1704, max_length=6).load_classification(
            archive_path=archive_path)
        for field in ("source_sha256", "split_id_sha256", "vocabulary_sha256",
                      "truncated_counts"):
            self.assertEqual(extracted["metadata"][field], zipped["metadata"][field])
        for split in ("train", "validation", "test"):
            self.assertEqual([(r["id"], r["tokens"], r["label"]) for r in extracted[split]],
                             [(r["id"], r["tokens"], r["label"]) for r in zipped[split]])
        original_read = ZipFile.read
        def guard_read(archive, name, *args, **kwargs):
            self.assertNotIn("/test/", str(name))
            return original_read(archive, name, *args, **kwargs)
        with patch.object(ZipFile, "read", guard_read):
            train_only = Dataset_Loader(self.root, seed=1704, max_length=6).load_classification(
                include_test=False, archive_path=archive_path)
        self.assertNotIn("test", train_only)
        self.assertEqual(extracted["metadata"]["split_id_sha256"]["train"],
                         train_only["metadata"]["split_id_sha256"]["train"])
        self.assertEqual(extracted["metadata"]["vocabulary_sha256"],
                         train_only["metadata"]["vocabulary_sha256"])

    def test_invalid_truncation_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "truncation_mode"):
            Dataset_Loader(self.root).load_classification(include_test=False,
                                                          truncation_mode="middle")


if __name__ == "__main__":
    unittest.main()
