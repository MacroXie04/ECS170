import tempfile
import unittest
from pathlib import Path

import torch

from local_code.stage_4_code.Experiment_Runner import collate, sequence_objective, evaluation
from local_code.stage_4_code.Method_RNN import SequenceModel


class RunnerTests(unittest.TestCase):
    def test_shift_and_padding_loss(self):
        records = [{'id': 'a', 'tokens': [2, 4, 5, 3]}, {'id': 'b', 'tokens': [2, 6, 3]}]
        tokens, lengths, _, _ = collate(records)
        targets = tokens[:, 1:]
        self.assertEqual(targets.tolist(), [[4, 5, 3], [6, 3, 0]])
        logits = torch.zeros(2, 3, 7, requires_grad=True)
        loss, correct, n = sequence_objective(logits, targets)
        self.assertEqual(int(n), 5)
        self.assertAlmostEqual(float(loss.detach()), 5 * torch.log(torch.tensor(7.)).item(), places=5)
        loss.backward()
        self.assertEqual(float(logits.grad[1, 2].abs().sum()), 0)

    def test_checkpoint_reload_reproduces_logits(self):
        torch.manual_seed(4)
        model = SequenceModel(10, task='classification', cell='lstm', embedding_dim=4, hidden_dim=6).eval()
        tokens = torch.tensor([[4, 5, 0], [6, 7, 8]])
        lengths = torch.tensor([2, 3])
        expected = model(tokens, lengths)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'checkpoint.pt'
            torch.save(model.state_dict(), path)
            restored = SequenceModel(10, task='classification', cell='lstm', embedding_dim=4, hidden_dim=6).eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(expected, restored(tokens, lengths), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
