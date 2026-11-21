"""Saved-checkpoint inference contracts; tiny random models, CPU only."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from local_code.stage_4_code.Dataset_Loader import EOS, Vocabulary
from local_code.stage_4_code.Method_RNN import SequenceModel
from script.stage_4_script.predict_stage4 import main, predict


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.vocab = Vocabulary(["what", "did", "the", "?", "bird", "say", ",", "a", "good", "movie", "bad", "."])

    def tearDown(self):
        self.tmp.cleanup()

    def checkpoint(self, task, cell="rnn", force_token=None):
        run = self.root / f"{task}_{cell}"
        (run / "checkpoints").mkdir(parents=True, exist_ok=True)
        kwargs = dict(vocab_size=len(self.vocab), task=task, cell=cell,
                      embedding_dim=4, hidden_dim=5, num_layers=1, dropout=0.0)
        model = SequenceModel(**kwargs)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            if task == "classification":
                model.output.bias.copy_(torch.tensor([-1.0, 1.0]))
            elif force_token is not None:
                model.output.bias[force_token] = 5
        config = {"model": kwargs, "classification_max_length": 4}
        canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False)
        config_hash = hashlib.sha256(canonical.encode()).hexdigest()
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        checkpoint = run / "checkpoints" / "best.pt"
        torch.save({"model": model.state_dict(), "model_kwargs": kwargs, "config_hash": config_hash,
                    "epoch": 7, "vocab": self.vocab.to_dict()}, checkpoint)
        return checkpoint

    def test_classification_saved_weights_probabilities_and_truncation_all_cells(self):
        for cell in ("rnn", "lstm", "gru"):
            with self.subTest(cell=cell):
                path = self.checkpoint("classification", cell)
                result = predict(path, "a good movie a good movie", device="cpu")
                self.assertEqual(result["label"], "positive")
                self.assertAlmostEqual(result["probabilities"]["positive"], torch.sigmoid(torch.tensor(2.0)).item(), places=6)
                self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=6)
                self.assertEqual(result["used_token_count"], 4)
                self.assertTrue(result["truncated"])
                self.assertEqual(result["checkpoint_epoch"], 7)

    def test_classification_oov_is_explicit(self):
        result = predict(self.checkpoint("classification"), "a strange movie", device="cpu")
        self.assertEqual(result["unknown_tokens"], ["strange"])
        self.assertEqual(result["unknown_token_count"], 1)

    def test_generation_counts_words_not_punctuation_and_preserves_prefix(self):
        result = predict(self.checkpoint("generation", force_token=EOS), "What, did the?", device="cpu")
        self.assertEqual(result["prompt_word_count"], 3)
        self.assertEqual(result["prompt_token_count"], 5)
        self.assertEqual(result["text"], "What, did the?")
        self.assertTrue(result["ended_with_eos"])

    def test_generation_appends_punctuation_without_changing_prefix(self):
        result = predict(self.checkpoint("generation", force_token=self.vocab.stoi["?"]),
                         "What did the", device="cpu", max_new_tokens=2)
        self.assertEqual(result["text"], "What did the??")
        self.assertEqual(result["generated_token_count"], 2)
        self.assertFalse(result["ended_with_eos"])

    def test_generation_invalid_word_count_or_unknown_prompt_rejected(self):
        path = self.checkpoint("generation")
        for text in ("what did", "what did the bird"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "exactly three words"):
                predict(path, text, device="cpu")
        with self.assertRaisesRegex(ValueError, "spaceship"):
            predict(path, "what did spaceship", device="cpu")

    def test_generation_seed_reproducibility_and_rng_isolation(self):
        path = self.checkpoint("generation", cell="gru")
        rng = torch.get_rng_state().clone()
        one = predict(path, "what did the", device="cpu", temperature=0.8, top_k=8, seed=123, max_new_tokens=8)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        two = predict(path, "what did the", device="cpu", temperature=0.8, top_k=8, seed=123, max_new_tokens=8)
        self.assertEqual(one, two)

    def test_modified_or_missing_classification_config_is_rejected(self):
        path = self.checkpoint("classification")
        config_path = path.parent.parent / "config.json"
        config = json.loads(config_path.read_text())
        config["classification_max_length"] = 3
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "does not match"):
            predict(path, "a good movie", device="cpu")
        config_path.unlink()
        with self.assertRaisesRegex(ValueError, "original config.json"):
            predict(path, "a good movie", device="cpu")

    def test_text_file_cli_returns_valid_json(self):
        path = self.checkpoint("classification")
        text_file = self.root / "review.txt"
        text_file.write_text("a good movie", encoding="utf-8")
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(main([str(path), "--text-file", str(text_file), "--device", "cpu"]), 0)
        self.assertEqual(json.loads(stream.getvalue())["label"], "positive")

    def test_empty_text_and_unavailable_cuda_have_clear_errors(self):
        path = self.checkpoint("classification")
        for text in ("", "   ", "<br />"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                predict(path, text, device="cpu")
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA was requested"):
                predict(path, "a good movie", device="cuda")


if __name__ == "__main__":
    unittest.main()
