"""Behavioral checks for padding, recurrent state, causality, and decoding."""
import unittest

import torch
from torch.nn import functional as F

from local_code.stage_4_code.Method_RNN import SequenceModel, generate


class SequenceModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1704)
        torch.set_num_threads(1)

    def model(self, task="classification", cell="rnn", **kwargs):
        return SequenceModel(17, task=task, cell=cell, embedding_dim=8,
                             hidden_dim=11, **kwargs)

    def test_padding_does_not_change_classification(self):
        for cell in ("rnn", "lstm", "gru"):
            with self.subTest(cell=cell):
                model = self.model(cell=cell).eval()
                single = model(torch.tensor([[4, 5, 6]]))
                batch = model(torch.tensor([[4, 5, 6, 0, 0], [7, 8, 9, 10, 11]]))
                torch.testing.assert_close(single[0], batch[0], atol=1e-6, rtol=1e-5)
                # Explicit lengths must also mask non-padding values after the end.
                trailing = model(torch.tensor([[4, 5, 6, 14, 15]]), torch.tensor([3]))
                torch.testing.assert_close(single, trailing, atol=1e-6, rtol=1e-5)

    def test_generation_is_causal_and_step_matches_forward(self):
        for cell in ("rnn", "lstm", "gru"):
            with self.subTest(cell=cell):
                model = self.model(task="generation", cell=cell).eval()
                first = torch.tensor([[2, 4, 5, 6, 7]])
                second = torch.tensor([[2, 4, 5, 14, 15]])
                full = model(first)
                torch.testing.assert_close(full[:, :3], model(second)[:, :3])
                prefix, state = model.step(first[:, :3])
                suffix, _ = model.step(first[:, 3:], state)
                torch.testing.assert_close(full, torch.cat([prefix, suffix], dim=1))

    def test_all_cells_have_finite_gradients_for_both_tasks(self):
        for cell in ("rnn", "lstm", "gru"):
            for task in ("classification", "generation"):
                with self.subTest(cell=cell, task=task):
                    model = self.model(task=task, cell=cell, num_layers=2)
                    inputs = torch.tensor([[2, 4, 5], [2, 7, 8]])
                    logits = model(inputs)
                    if task == "classification":
                        self.assertEqual(logits.shape, (2, 2))
                        loss = F.cross_entropy(logits, torch.tensor([0, 1]))
                    else:
                        self.assertEqual(logits.shape, (2, 3, 17))
                        loss = F.cross_entropy(logits.reshape(-1, 17),
                                               torch.tensor([4, 5, 3, 7, 8, 3]))
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(model.embedding.weight.grad[4].abs().sum().item(), 0)
                    self.assertEqual(model.embedding.weight.grad[0].abs().sum().item(), 0)
                    self.assertGreater(model.parameter_count(), 0)

    def test_generation_preserves_prefix_and_stops_at_eos(self):
        model = self.model(task="generation")
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.fill_(-10)
            # Illegal tokens are more likely than EOS before suppression.
            model.output.bias[:3] = 100
            model.output.bias[3] = 10
        prefix = [2, 4, 5, 6]
        self.assertEqual(generate(model, prefix), prefix + [3])
        self.assertTrue(model.training)
        model.eval()
        self.assertEqual(generate(model, prefix + [3]), prefix + [3])
        self.assertFalse(model.training)

    def test_sampling_reproducible_without_changing_global_rng(self):
        model = self.model(task="generation", cell="gru")
        prefix = [2, 4, 5, 6]
        rng = torch.get_rng_state().clone()
        first = generate(model, prefix, temperature=0.8, top_k=4, seed=9)
        second = generate(model, prefix, temperature=0.8, top_k=4, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(first[:len(prefix)], prefix)
        self.assertFalse(set(first[len(prefix):]) & {0, 1, 2})
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(model.training)

    def test_invalid_inputs_fail_clearly(self):
        model = self.model()
        for lengths in (torch.tensor([0]), torch.tensor([5]), torch.tensor([1.5])):
            with self.assertRaises(ValueError):
                model(torch.tensor([[4, 5]]), lengths)
        with self.assertRaises(ValueError):
            model.step(torch.tensor([[4]]))
        with self.assertRaises(ValueError):
            generate(model, [2, 4, 5, 6])
        with self.assertRaises(ValueError):
            generate(self.model(task="generation"), [])


if __name__ == "__main__":
    unittest.main()
