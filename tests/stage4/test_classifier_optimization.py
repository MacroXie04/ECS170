"""Focused checks for the classification-only recurrent model."""
import unittest

import torch

from local_code.stage_4_code.Method_Classifier import ClassifierModel


class ClassifierModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1704)

    def test_right_padding_does_not_change_logits(self):
        tokens = torch.tensor(
            [[3, 4, 5, 6, 7], [8, 9, 10, 0, 0], [11, 12, 0, 0, 0]]
        )
        lengths = torch.tensor([5, 3, 2])
        extra_padding = torch.nn.functional.pad(tokens, (0, 7), value=0)
        for cell, bidirectional in (
            ("rnn", False), ("gru", False), ("gru", True), ("lstm", True)
        ):
            for pooling in ("mean", "max", "meanmax"):
                with self.subTest(cell=cell, bidirectional=bidirectional, pooling=pooling):
                    model = ClassifierModel(
                        vocab_size=32,
                        cell=cell,
                        embedding_dim=8,
                        hidden_dim=6,
                        dropout=0,
                        bidirectional=bidirectional,
                        pooling=pooling,
                    ).eval()
                    with torch.no_grad():
                        original = model(tokens, lengths)
                        extended = model(extra_padding, lengths)
                        separate = torch.cat(
                            [model(tokens[i : i + 1, : int(length)], lengths[i : i + 1])
                             for i, length in enumerate(lengths)]
                        )
                    self.assertEqual(tuple(original.shape), (3, 2))
                    torch.testing.assert_close(original, extended, rtol=1e-6, atol=1e-6)
                    torch.testing.assert_close(original, separate, rtol=1e-6, atol=1e-6)

    def test_max_pool_ignores_zero_states_from_padding(self):
        tokens = torch.tensor([[3, 4, 0, 0]])
        lengths = torch.tensor([2])
        for pooling in ("max", "meanmax"):
            with self.subTest(pooling=pooling):
                model = ClassifierModel(
                    vocab_size=8,
                    cell="rnn",
                    embedding_dim=3,
                    hidden_dim=2,
                    dropout=0,
                    pooling=pooling,
                ).eval()
                with torch.no_grad():
                    model.recurrent.weight_ih_l0.zero_()
                    model.recurrent.weight_hh_l0.zero_()
                    model.recurrent.bias_ih_l0.fill_(-1)
                    model.recurrent.bias_hh_l0.zero_()
                    model.output.weight.fill_(1)
                    model.output.bias.zero_()
                    logits = model(tokens, lengths)
                self.assertTrue(torch.all(logits < -1), logits)

    def test_invalid_lengths_rejected(self):
        model = ClassifierModel(vocab_size=16, dropout=0)
        tokens = torch.tensor([[2, 3, 0]])
        for lengths in ([0], [4], [1.5], [True], [1, 2]):
            with self.subTest(lengths=lengths):
                with self.assertRaises(ValueError):
                    model(tokens, lengths)


if __name__ == "__main__":
    unittest.main()
