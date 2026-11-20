"""Data-contract and leakage tests using small synthetic datasets."""
import csv
import tempfile
import unittest
from pathlib import Path

from local_code.stage_4_code.Dataset_Loader import (
    BOS, EOS, PAD, UNK, Dataset_Loader, Vocabulary, detokenize, tokenize,
)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for split in ("train", "test"):
            for label in ("neg", "pos"):
                directory = self.root / "text_classification" / split / label
                directory.mkdir(parents=True)
                for index in range(20 if split == "train" else 3):
                    text = f"A film is {label}. {split}{label}{index}"
                    (directory / f"{index}_8.txt").write_text(text, encoding="utf-8")
        gen = self.root / "text_generation"
        gen.mkdir()
        with (gen / "data").open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=["ID", "Joke"])
            writer.writeheader()
            for index in range(30):
                writer.writerow({"ID": index, "Joke": f"A small creature number{index} builds a house."})
            writer.writerow({"ID": 30, "Joke": "A SMALL CREATURE NUMBER0 BUILDS A HOUSE."})
            writer.writerow({"ID": 31, "Joke": "A small creature number0 builds a house."})

    def tearDown(self):
        self.tmp.cleanup()

    def test_cleaning_retains_negation_and_contractions(self):
        tokens = tokenize("I CAN’T say it’s good.<br />Not good &amp; never dull!")
        self.assertIn("can't", tokens)
        self.assertIn("not", tokens)
        self.assertIn("never", tokens)
        self.assertNotIn("br", tokens)
        self.assertEqual(detokenize(tokenize("What did the bird say? Nothing!")),
                         "what did the bird say? nothing!")

    def test_vocabulary_specials_serialization_and_oov(self):
        vocab = Vocabulary.build(["word word another"], max_vocab=6)
        self.assertEqual([vocab.stoi[t] for t in ("<pad>", "<unk>", "<bos>", "<eos>")],
                         [PAD, UNK, BOS, EOS])
        self.assertEqual(vocab.encode("unseen", bos=True, eos=True), [BOS, UNK, EOS])
        restored = Vocabulary.from_dict(vocab.to_dict())
        self.assertEqual(restored.tokens, vocab.tokens)
        self.assertEqual(restored.encode("another word"), vocab.encode("another word"))
        self.assertEqual(vocab.decode([BOS, vocab.stoi["word"], EOS, PAD]), "word")
        with self.assertRaises(ValueError):
            vocab.decode([len(vocab)])

    def test_imdb_preserves_official_test_and_stratifies_validation(self):
        data = Dataset_Loader(self.root).load_classification()
        self.assertEqual(data["metadata"]["counts"], {"train": 36, "validation": 4, "test": 6})
        self.assertEqual(data["metadata"]["label_counts"]["validation"], {"0": 2, "1": 2})
        self.assertTrue(all(r["id"].startswith("train/") for r in data["train"] + data["validation"]))
        self.assertTrue(all(r["id"].startswith("test/") for r in data["test"]))
        expected = {p.relative_to(self.root / "text_classification").as_posix()
                    for p in (self.root / "text_classification" / "test").rglob("*.txt")}
        self.assertEqual({r["id"] for r in data["test"]}, expected)
        self.assertEqual(set(data["metadata"]["id_overlap_counts"].values()), {0})

    def test_imdb_vocabulary_fitted_to_training_only(self):
        data = Dataset_Loader(self.root).load_classification()
        train_words = {word for row in data["train"] for word in tokenize(row["text"])}
        self.assertEqual(set(data["vocab"].tokens[4:]), train_words)
        for row in data["validation"] + data["test"]:
            unique = tokenize(row["text"])[-1]
            self.assertNotIn(unique, data["vocab"].stoi)
            self.assertIn(UNK, row["tokens"])

    def test_classification_truncation_and_empty_text(self):
        (self.root / "text_classification" / "test" / "neg" / "0_8.txt").write_text("<br />")
        data = Dataset_Loader(self.root, max_length=2).load_classification()
        self.assertTrue(all(1 <= len(r["tokens"]) <= 2 for name in ("train", "validation", "test") for r in data[name]))
        empty = next(r for r in data["test"] if r["id"] == "test/neg/0_8.txt")
        self.assertEqual(empty["tokens"], [UNK])

    def test_classification_duplicate_groups_never_cross_validation(self):
        # A duplicate group larger than a class budget must stay wholly in
        # training; row-level stratification would inevitably leak this group.
        for path in (self.root / "text_classification" / "train" / "pos").glob("*.txt"):
            path.write_text("A repeated positive movie review.", encoding="utf-8")
        data = Dataset_Loader(self.root).load_classification()
        train_texts = {tuple(tokenize(r["text"])) for r in data["train"]}
        validation_texts = {tuple(tokenize(r["text"])) for r in data["validation"]}
        self.assertFalse(train_texts & validation_texts)
        self.assertEqual(sum(len(data[name]) for name in ("train", "validation")), 40)
        self.assertEqual(data["metadata"]["normalized_text_overlap_counts"]["train_validation"], 0)
        self.assertEqual(data["metadata"]["official_train_duplicate_normalized_rows"], 19)

    def test_generation_duplicates_stay_together_and_all_rows_retained(self):
        data = Dataset_Loader(self.root).load_generation()
        self.assertEqual(sum(len(data[name]) for name in ("train", "validation", "test")), 32)
        self.assertEqual(data["metadata"]["normalized_group_count"], 30)
        self.assertEqual(data["metadata"]["duplicate_normalized_rows"], 2)
        self.assertEqual(set(data["metadata"]["normalized_text_overlap_counts"].values()), {0})
        owners = {row["id"]: name for name in ("train", "validation", "test") for row in data[name]}
        self.assertEqual(owners["joke/0"], owners["joke/30"])
        self.assertEqual(owners["joke/0"], owners["joke/31"])

    def test_generation_training_vocab_bos_eos_and_no_truncation(self):
        data = Dataset_Loader(self.root, max_length=2).load_generation()
        train_words = {word for row in data["train"] for word in tokenize(row["text"])}
        self.assertEqual(set(data["vocab"].tokens[4:]), train_words)
        for name in ("train", "validation", "test"):
            for row in data[name]:
                self.assertEqual(row["tokens"][0], BOS)
                self.assertEqual(row["tokens"][-1], EOS)
                self.assertEqual(len(row["tokens"]), len(tokenize(row["text"])) + 2)
        for row in data["validation"] + data["test"]:
            novel = next(w for w in tokenize(row["text"]) if w.startswith("number"))
            self.assertNotIn(novel, data["vocab"].stoi)
            self.assertIn(UNK, row["tokens"])

    def test_deterministic_splits_and_changed_seed(self):
        first = Dataset_Loader(self.root, seed=17).load_generation()
        repeat = Dataset_Loader(self.root, seed=17).load_generation()
        other = Dataset_Loader(self.root, seed=18).load_generation()
        self.assertEqual(first["metadata"]["split_ids"], repeat["metadata"]["split_ids"])
        self.assertEqual(first["vocab"].tokens, repeat["vocab"].tokens)
        self.assertNotEqual(first["metadata"]["split_ids"], other["metadata"]["split_ids"])

    def test_invalid_configuration_rejected(self):
        for kwargs in ({"max_length": 0}, {"max_vocab": 3}, {"validation_fraction": 0}):
            with self.assertRaises(ValueError):
                Dataset_Loader(self.root, **kwargs)


if __name__ == "__main__":
    unittest.main()
