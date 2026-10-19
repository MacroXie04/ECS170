"""Targeted checks for label position, optimization, modes, and checkpoints."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Evaluate import Evaluate
from local_code.stage_2_code.Setting import seed_everything

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('runner', ROOT / 'script/stage_2_script/script_mlp.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Stage2Tests(unittest.TestCase):
    def test_csv_first_column_no_header_and_unknown_class(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p / 'train.csv').write_text('3,0,255\n7,255,0\n')
            loader = Dataset_Loader(folder)
            data = loader.load()  # Works even when no test.csv exists.
            np.testing.assert_array_equal(data['y'], [0, 1])
            np.testing.assert_array_equal(data['X'], [[0, 1], [1, 0]])
            (p / 'test.csv').write_text('8,0,0\n')
            with self.assertRaises(ValueError):
                loader.load_file('test.csv', data['classes'])

    def test_gradient_reduces_loss_and_checkpoint_preserves_predictions(self):
        seed_everything(42)
        config = dict(runner.BASE, hidden=[16])
        model = Method_MLP(4, 2, config)
        X = torch.eye(4).repeat(8, 1)
        y = torch.tensor([0, 1, 0, 1]).repeat(8)
        criterion = model.criterion()
        initial = criterion(model(X), y).item()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        for _ in range(50):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            optimizer.step()
        self.assertLess(criterion(model(X), y).item(), initial / 10)
        self.assertEqual((model(X).argmax(1) == y).sum().item(), len(y))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pt'
            torch.save(model.state_dict(), path)
            restored = Method_MLP(4, 2, config)
            restored.load_state_dict(torch.load(path, weights_only=True))
            np.testing.assert_array_equal(model.test(X.numpy()), restored.test(X.numpy()))

    def test_original_train_eval_modes_and_independent_initialization(self):
        config = dict(runner.BASE, dropout=0.5)
        seed_everything(42)
        a = Method_MLP(4, 2, config)
        seed_everything(42)
        b = Method_MLP(4, 2, config)
        for left, right in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(left, right))
            self.assertNotEqual(left.data_ptr(), right.data_ptr())
        a.train()
        self.assertTrue(a.training)
        a.eval()
        self.assertTrue(all(not module.training for module in a.modules()))
        x = torch.ones(8, 4)
        torch.testing.assert_close(a(x), a(x))

    def test_multiclass_metrics_include_unpredicted_classes(self):
        metrics = Evaluate([0, 1, 2]).score([0, 1, 2], [0, 0, 0])
        self.assertAlmostEqual(metrics['accuracy'], 1/3)
        self.assertAlmostEqual(metrics['macro']['recall'], 1/3)
        self.assertAlmostEqual(metrics['macro']['f1'], 1/6)
        self.assertEqual(np.array(metrics['confusion_matrix']).shape, (3, 3))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
