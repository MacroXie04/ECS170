"""Stage 3 unit tests. Most use synthetic tensors; ones needing the instructor's
files skip when data/stage_3_data is absent.

    python -m unittest discover -s tests/stage3 -p 'test_stage3.py' -v
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from local_code.stage_3_code import Experiment_Runner as runner  # noqa: E402
from local_code.stage_3_code.Dataset_Loader import (DATASET_SPECS, BatchAugmentation,  # noqa: E402
                                                    Dataset_Loader, ImageTensorDataset,
                                                    _NumpyOnlyUnpickler,
                                                    compute_channel_stats, make_eval_loader,
                                                    make_train_loader,
                                                    make_train_validation_split)
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics  # noqa: E402
from local_code.stage_3_code.Experiment_Config import (FORMAL_SEEDS, REGISTRY,  # noqa: E402
                                                       apply_overrides, get_config)
from local_code.stage_3_code.Method_CNN import (NonFiniteLossError, is_better_epoch,  # noqa: E402
                                                load_checkpoint)
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST  # noqa: E402
from local_code.stage_3_code.Report_Artifacts import (aggregate_configs,  # noqa: E402
                                                      select_best_config, select_final_seed,
                                                      summarize)
from local_code.stage_3_code.Result_Saver import Result_Saver  # noqa: E402
from local_code.stage_3_code.Runtime import (atomic_write, exclusive_run_lock,  # noqa: E402
                                             resolve_device, seed_everything)
from local_code.stage_3_code.Setting_Stage3 import Setting_Stage3, attach_training_data  # noqa: E402

DATA = REPO / 'data' / 'stage_3_data'
HAS_ORL = (DATA / 'ORL').is_file()
NORMALIZATION = {'mean': [0.5], 'std': [0.25]}
SEED = FORMAL_SEEDS[0]


def tiny_config(**changes):
    values = dict(max_epoch=4, batch_size=16, aug_shift=2, channels=(4,), hidden_dims=(8,))
    values.update(changes)
    return replace(get_config('mnist', 'baseline'), **values)


def synthetic_digits(count=64, seed=0):
    generator = np.random.default_rng(seed)
    return {'X': generator.integers(0, 256, (count, 1, 28, 28), dtype=np.uint8),
            'y': np.arange(count, dtype=np.int64) % 10}


def tiny_method(config=None, seed=1, on_epoch_end=None):
    seed_everything(seed)
    method = Method_CNN_MNIST(config or tiny_config())
    method.log = lambda _message: None
    attach_training_data(method, synthetic_digits(), synthetic_digits(32, 1), NORMALIZATION,
                         seed, on_epoch_end or (lambda _improved: None))
    return method


class DataTests(unittest.TestCase):
    def test_unpickler_blocks_code_execution(self):
        class Attack:
            def __reduce__(self):
                return os.system, ('echo attacked',)
        path = Path(tempfile.mkdtemp()) / 'bad'
        path.write_bytes(pickle.dumps(Attack()))
        with open(path, 'rb') as handle, self.assertRaises(pickle.UnpicklingError):
            _NumpyOnlyUnpickler(handle).load()

    def test_unpickler_allows_numpy_arrays(self):
        path = Path(tempfile.mkdtemp()) / 'good'
        path.write_bytes(pickle.dumps({'train': [{'image': np.ones((2, 2)), 'label': 1}]}))
        with open(path, 'rb') as handle:
            self.assertEqual(_NumpyOnlyUnpickler(handle).load()['train'][0]['label'], 1)

    def test_orl_labels_become_zero_based(self):
        loader = Dataset_Loader('orl')
        shifted = loader._to_zero_based(np.arange(1, 41))
        np.testing.assert_array_equal(shifted, np.arange(40))
        with self.assertRaises(ValueError):
            loader._to_zero_based(np.arange(0, 40))

    def test_orl_keeps_one_identical_channel(self):
        loader = Dataset_Loader('orl')
        gray = np.random.default_rng(0).integers(0, 256, (2, 112, 92), dtype=np.uint8)
        converted = loader._to_uint8_nchw(np.stack([gray] * 3, axis=-1))
        self.assertEqual(converted.shape, (2, 1, 112, 92))
        np.testing.assert_array_equal(converted[:, 0], gray)
        color = np.stack([gray, gray, gray + 1], axis=-1)
        with self.assertRaises(ValueError):
            loader._to_uint8_nchw(color)

    def test_cifar_channels_move_first_without_mixing(self):
        images = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        images[0, 5, 7] = (10, 20, 30)
        converted = Dataset_Loader('cifar')._to_uint8_nchw(images)
        self.assertEqual(converted.shape, (1, 3, 32, 32))
        self.assertEqual(converted[0, :, 5, 7].tolist(), [10, 20, 30])

    def test_mnist_float_pixels_convert_only_when_lossless(self):
        loader = Dataset_Loader('mnist')
        pixels = np.full((1, 28, 28), 255.0)
        self.assertEqual(loader._to_uint8_nchw(pixels).dtype, np.uint8)
        with self.assertRaises(ValueError):
            loader._to_uint8_nchw(pixels - 0.5)

    def test_validation_split_is_disjoint_stratified_and_stable(self):
        labels = np.repeat(np.arange(40), 9)
        first = make_train_validation_split(labels, 40, 170)
        second = make_train_validation_split(labels, 40, 170)
        np.testing.assert_array_equal(first[1], second[1])
        self.assertFalse(set(first[0]) & set(first[1]))
        self.assertEqual(len(first[0]) + len(first[1]), 360)
        np.testing.assert_array_equal(np.bincount(labels[first[1]]), np.ones(40))
        np.testing.assert_array_equal(np.bincount(labels[first[0]]), np.full(40, 8))

    def test_normalization_matches_formula_and_uses_given_images(self):
        images = np.array([[[[0, 255]]], [[[255, 255]]]], dtype=np.uint8)
        mean, std = compute_channel_stats(images)
        self.assertAlmostEqual(mean[0], 0.75)
        data = ImageTensorDataset(images, np.array([0, 1]), mean, std)
        batch, _ = data.__getitems__([0])
        expected = (0.0 - 0.75) / std[0]
        self.assertAlmostEqual(batch[0, 0, 0, 0].item(), expected, places=5)

    def test_batched_and_single_access_agree(self):
        digits = synthetic_digits(8)
        data = ImageTensorDataset(digits['X'], digits['y'], [0.5], [0.25])
        batch, labels = data.__getitems__([3, 1])
        single, label = data[3]
        self.assertTrue(torch.equal(batch[0], single))
        self.assertEqual(labels[0], label)

    def test_shift_moves_pixels_and_keeps_shape(self):
        image = torch.zeros(1, 1, 9, 9)
        image[0, 0, 4, 4] = 1
        augmentation = BatchAugmentation(shift=2)
        moved = {tuple(torch.nonzero(augmentation(image, torch.Generator().manual_seed(seed)))[0].tolist())
                 for seed in range(40)}
        self.assertGreater(len(moved), 3)
        for _, _, row, column in moved:
            self.assertLessEqual(abs(row - 4), 2)
            self.assertLessEqual(abs(column - 4), 2)

    def test_flip_affects_about_half_of_a_batch(self):
        images = torch.arange(4.0).view(1, 1, 1, 4).repeat(1000, 1, 1, 1)
        flipped = BatchAugmentation(hflip=True)(images, torch.Generator().manual_seed(0))
        rate = (flipped[:, 0, 0, 0] == 3).float().mean().item()
        self.assertTrue(0.4 < rate < 0.6)

    def test_evaluation_loader_has_no_augmentation_or_shuffle(self):
        digits = synthetic_digits(32)
        loader = make_eval_loader(digits['X'], digits['y'], [0.5], [0.25], 16)
        self.assertIsNone(loader.dataset.augmentation)
        first = torch.cat([batch[0] for batch in loader])
        second = torch.cat([batch[0] for batch in loader])
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(torch.cat([batch[1] for batch in loader]).tolist(), digits['y'].tolist())

    def test_training_loader_augments(self):
        digits = synthetic_digits(32)
        generators = [torch.Generator().manual_seed(seed) for seed in (1, 2)]
        loader = make_train_loader(digits['X'], digits['y'], [0.5], [0.25], tiny_config(),
                                   *generators)
        self.assertIsNotNone(loader.dataset.augmentation)

    @unittest.skipUnless(HAS_ORL, 'instructor data not present')
    def test_training_phase_load_excludes_test_data(self):
        loaded = Dataset_Loader('orl', folder=DATA).load(splits=('train',))
        self.assertEqual(set(loaded), {'train'})
        self.assertEqual(loaded['train']['X'].shape, (360, 1, 112, 92))


class ConfigTests(unittest.TestCase):
    def test_every_dataset_has_baseline_and_two_variants(self):
        for configs in REGISTRY.values():
            self.assertIn('baseline', configs)
            self.assertGreaterEqual(len(configs), 3)

    def test_variants_change_one_factor(self):
        allowed = {'depth': {'channels', 'convs_per_stage'}, 'kernel_size': {'kernel_size', 'padding'},
                   'pooling': {'downsample'}, 'stride': {'downsample'},
                   'hidden_dim': {'hidden_dims'}, 'loss': {'loss'}, 'dropout': {'dropout'}}
        labels = {'name', 'changed_factor', 'description'}
        for configs in REGISTRY.values():
            baseline = configs['baseline'].to_dict()
            for name, config in configs.items():
                if name == 'baseline':
                    continue
                changed = {key for key, value in config.to_dict().items()
                           if value != baseline[key]} - labels
                self.assertTrue(changed, name)
                self.assertLessEqual(changed, allowed[config.changed_factor], name)

    def test_formal_seeds_are_distinct(self):
        self.assertEqual(len(set(FORMAL_SEEDS)), len(FORMAL_SEEDS))

    def test_overrides_are_typed(self):
        config = apply_overrides(get_config('mnist', 'baseline'),
                                 ['learning_rate=0.01', 'channels=[8,16]', 'aug_hflip=true'])
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.channels, (8, 16))
        self.assertIs(config.aug_hflip, True)
        with self.assertRaises(ValueError):
            apply_overrides(config, ['not_a_field=1'])

    def test_config_survives_a_dict_round_trip(self):
        config = get_config('cifar', 'baseline')
        restored = type(config).from_dict(json.loads(json.dumps(config.to_dict())))
        self.assertEqual(restored.fingerprint(), config.fingerprint())


class ModelTests(unittest.TestCase):
    def test_all_configs_produce_class_logits(self):
        for dataset, configs in REGISTRY.items():
            spec = DATASET_SPECS[dataset]
            for config in configs.values():
                method = runner.METHOD_CLASSES[dataset](config)
                output = method(torch.zeros(2, *spec.input_shape))
                self.assertEqual(tuple(output.shape), (2, spec.num_classes), config.name)

    def test_method_keeps_nn_module_train_and_eval(self):
        method = tiny_method()
        method.eval()
        self.assertFalse(method.training)
        method.train()
        self.assertTrue(method.training)

    def test_predict_runs_in_eval_mode(self):
        method = tiny_method(tiny_config(dropout=0.5))
        loader = method.data['val_loader']
        first = method.predict(loader)['logits']
        second = method.predict(loader)['logits']
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(method.training)

    def test_wrong_dataset_config_is_rejected(self):
        with self.assertRaises(ValueError):
            Method_CNN_MNIST(get_config('orl', 'baseline'))

    def test_mse_loss_is_zero_for_a_perfect_prediction(self):
        method = Method_CNN_MNIST(tiny_config(loss='mse_softmax'))
        logits = torch.full((2, 10), -50.0)
        logits[0, 3] = logits[1, 7] = 50.0
        self.assertAlmostEqual(method.objective(logits, torch.tensor([3, 7])).item(), 0.0, places=6)
        self.assertAlmostEqual(method.objective(logits, torch.tensor([7, 3])).item(), 2.0, places=5)

    def test_training_reduces_loss(self):
        labels = np.arange(64, dtype=np.int64) % 10
        images = np.zeros((64, 1, 28, 28), dtype=np.uint8)
        for index, label in enumerate(labels):  # a class-specific bright band
            images[index, 0, 2 * label:2 * label + 2] = 255
        learnable = {'X': images, 'y': labels}
        seed_everything(1)
        method = Method_CNN_MNIST(tiny_config(max_epoch=25, aug_shift=0, dropout=0.0))
        method.log = lambda _message: None
        attach_training_data(method, learnable, learnable, NORMALIZATION, 1,
                             lambda _improved: None)
        history = method.fit()
        self.assertLess(history[-1]['train_loss'], 0.5 * history[0]['train_loss'])

    def test_diverged_training_stops(self):
        method = tiny_method(tiny_config(learning_rate=float('inf'), max_epoch=3))
        with self.assertRaises(NonFiniteLossError):
            method.fit()

    def test_resume_exactly_continues_an_uninterrupted_run(self):
        path = Path(tempfile.mkdtemp()) / 'state.pt'
        config = tiny_config(dropout=0.25)
        straight = tiny_method(config)
        straight.fit()

        interrupted = tiny_method(config)

        def stop_point(_improved):
            if interrupted.history[-1]['epoch'] == 2:
                interrupted.save_checkpoint(path)
        interrupted.data['on_epoch_end'] = stop_point
        interrupted.fit()

        resumed = tiny_method(config)
        resumed.restore_training_state(load_checkpoint(path))
        self.assertEqual(resumed.start_epoch, 3)
        resumed.fit()
        for name, value in straight.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.state_dict()[name]), name)
        self.assertEqual([row['train_loss'] for row in straight.history],
                         [row['train_loss'] for row in resumed.history])
        self.assertEqual(straight.best['epoch'], resumed.best['epoch'])

    def test_checkpoint_rejects_a_different_config(self):
        path = Path(tempfile.mkdtemp()) / 'state.pt'
        method = tiny_method()
        method.configure_optimizer()
        method.save_checkpoint(path)
        other = tiny_method(tiny_config(max_epoch=9))
        with self.assertRaises(ValueError):
            other.restore_training_state(load_checkpoint(path))


class SelectionTests(unittest.TestCase):
    def test_checkpoint_rule_order(self):
        incumbent = {'val_accuracy': 0.9, 'val_loss': 0.3}
        self.assertTrue(is_better_epoch({'val_accuracy': 0.91, 'val_loss': 0.9}, incumbent))
        self.assertTrue(is_better_epoch({'val_accuracy': 0.9, 'val_loss': 0.2}, incumbent))
        self.assertFalse(is_better_epoch({'val_accuracy': 0.9, 'val_loss': 0.3}, incumbent))
        self.assertFalse(is_better_epoch({'val_accuracy': 0.89, 'val_loss': 0.1}, incumbent))
        self.assertTrue(is_better_epoch(incumbent, None))

    @staticmethod
    def records(test_scores):
        rows = []
        for name, val, score in (('a', 0.90, test_scores[0]), ('b', 0.95, test_scores[1])):
            for seed in FORMAL_SEEDS:
                rows.append({'dataset': 'mnist', 'config': name, 'changed_factor': 'x',
                             'parameters': 10, 'seed': seed, 'val_accuracy': val,
                             'val_cross_entropy': 0.1, 'train_seconds': 1.0,
                             'test_accuracy': score, 'finished': True})
        return rows

    def test_best_config_ignores_test_scores(self):
        favoring_a = select_best_config(aggregate_configs(self.records((0.99, 0.50))))
        favoring_b = select_best_config(aggregate_configs(self.records((0.50, 0.99))))
        self.assertEqual(favoring_a['config'], 'b')
        self.assertEqual(favoring_b['config'], 'b')

    def test_config_tie_breaks_by_loss_then_size(self):
        full = {'seeds_finished': 3, 'seeds_planned': 3}
        rows = [{'config': 'big', 'val_accuracy_mean': 0.9, 'val_cross_entropy_mean': 0.1,
                 'parameters': 9, **full},
                {'config': 'small', 'val_accuracy_mean': 0.9, 'val_cross_entropy_mean': 0.1,
                 'parameters': 1, **full},
                {'config': 'lossy', 'val_accuracy_mean': 0.9, 'val_cross_entropy_mean': 0.2,
                 'parameters': 1, **full}]
        self.assertEqual(select_best_config(rows)['config'], 'small')

    def test_float_rounding_does_not_decide_an_accuracy_tie(self):
        full = {'seeds_finished': 3, 'seeds_planned': 3, 'parameters': 1}
        rows = [{'config': 'steady', 'val_accuracy_mean': 0.9499999999999998,
                 'val_cross_entropy_mean': 0.1, **full},
                {'config': 'noisy', 'val_accuracy_mean': 0.9500000000000001,
                 'val_cross_entropy_mean': 0.5, **full}]
        self.assertEqual(select_best_config(rows)['config'], 'steady')

    def test_unfinished_seeds_do_not_enter_means_or_selection(self):
        records = self.records((0.9, 0.9))
        for record in records:
            if record['config'] == 'b' and record['seed'] != FORMAL_SEEDS[0]:
                record.update(finished=False, val_accuracy=None)
        rows = aggregate_configs(records)
        by_name = {row['config']: row for row in rows}
        self.assertEqual(by_name['b']['seeds_finished'], 1)
        self.assertIsNone(by_name['b']['val_accuracy_std'])
        self.assertEqual(select_best_config(rows)['config'], 'a')

    def test_final_seed_follows_validation(self):
        rows = [{'seed': 1, 'val_accuracy': 0.9, 'val_cross_entropy': 0.1},
                {'seed': 2, 'val_accuracy': 0.9, 'val_cross_entropy': 0.05}]
        self.assertEqual(select_final_seed(rows)['seed'], 2)


class MetricTests(unittest.TestCase):
    def evaluate(self, true_y, pred_y, classes):
        evaluator = Evaluate_Metrics(num_classes=classes)
        evaluator.data = {'true_y': true_y, 'pred_y': pred_y}
        return evaluator.evaluate()

    def test_hand_computed_values(self):
        metrics = self.evaluate([0, 0, 0, 1], [0, 0, 1, 1], 2)
        self.assertAlmostEqual(metrics['accuracy'], 0.75)
        self.assertAlmostEqual(metrics['precision_macro'], (1.0 + 0.5) / 2)
        self.assertAlmostEqual(metrics['recall_macro'], (2 / 3 + 1.0) / 2)
        self.assertAlmostEqual(metrics['recall_weighted'], 0.75)
        self.assertEqual(metrics['confusion_matrix'], [[2, 1], [0, 1]])

    def test_never_predicted_class_still_counts(self):
        metrics = self.evaluate([0, 1, 2], [0, 0, 0], 3)
        self.assertAlmostEqual(metrics['recall_macro'], 1 / 3)
        self.assertEqual(len(metrics['confusion_matrix']), 3)

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate([0, 1], [0], 2)


class InfrastructureTests(unittest.TestCase):
    def test_atomic_write_leaves_nothing_on_failure(self):
        folder = Path(tempfile.mkdtemp())
        target = folder / 'file.txt'
        target.write_text('original')

        def fail(path):
            Path(path).write_text('partial')
            raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):
            atomic_write(target, fail)
        self.assertEqual(target.read_text(), 'original')
        self.assertEqual([path.name for path in folder.iterdir()], ['file.txt'])

    def test_result_saver_refuses_to_overwrite(self):
        saver = Result_Saver('saver', '')
        saver.result_destination_folder_path = tempfile.mkdtemp()
        saver.data = {'metrics': {'accuracy': 1.0}, 'predictions': {'pred_y': np.array([1])}}
        saver.save()
        with self.assertRaises(FileExistsError):
            saver.save()

    def test_interrupted_save_can_be_repeated(self):
        saver = Result_Saver('saver', '')
        saver.result_destination_folder_path = tempfile.mkdtemp()
        saver.data = {'metrics': {'accuracy': 1.0}, 'predictions': {'pred_y': np.array([1])}}
        saver.save()
        saver.metrics_path.unlink()  # the stop came before the metrics write
        saver.save()
        self.assertTrue(saver.metrics_path.exists())

    def test_unavailable_device_is_an_error(self):
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                resolve_device('cuda')
        self.assertIn(resolve_device('auto').type, ('cuda', 'mps', 'cpu'))

    def test_plan_starts_with_cifar_and_runs_all_seeds(self):
        plan = runner.planned_runs()
        self.assertEqual(plan[0][0], 'cifar')
        self.assertEqual(plan[0][2], FORMAL_SEEDS[0])
        self.assertEqual(len(plan), len(set(plan)))
        self.assertEqual(len(plan), sum(map(len, REGISTRY.values())) * len(FORMAL_SEEDS))

    def test_comparison_flags_a_changed_result(self):
        predictions = {'pred_y': np.array([0, 1]), 'true_y': np.array([0, 1])}
        evaluator = Evaluate_Metrics(num_classes=2)
        evaluator.data = predictions
        metrics = {**evaluator.evaluate(), 'test_cross_entropy': 0.3, 'device': 'cpu'}
        saved = {'metrics': metrics, 'predictions': predictions}
        self.assertTrue(runner.compare_evaluations(saved, metrics, predictions, 2)['passed'])
        worse = {**metrics, 'accuracy': 0.8}
        self.assertFalse(runner.compare_evaluations(saved, worse, predictions, 2)['passed'])

    def test_comparison_flags_predictions_that_contradict_metrics(self):
        predictions = {'pred_y': np.array([0, 1]), 'true_y': np.array([0, 1])}
        evaluator = Evaluate_Metrics(num_classes=2)
        evaluator.data = predictions
        metrics = {**evaluator.evaluate(), 'test_cross_entropy': 0.3, 'device': 'cpu'}
        edited = {'pred_y': np.array([1, 1]), 'true_y': np.array([0, 1])}
        report = runner.compare_evaluations({'metrics': metrics, 'predictions': edited}, metrics,
                                            predictions, 2)
        self.assertFalse(report['passed'])


@unittest.skipUnless(HAS_ORL, 'instructor data not present')
class EndToEndTests(unittest.TestCase):
    """A two-epoch ORL run through the same functions the CLI calls."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        short = replace(REGISTRY['orl']['baseline'], max_epoch=2)
        patcher = mock.patch.dict(REGISTRY['orl'], {'baseline': short})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.args = dict(device='cpu', root=self.root, data_dir=DATA)

    def test_train_never_requests_the_test_split(self):
        requested = []
        original = Dataset_Loader.load

        def spy(loader, splits=('train', 'test')):
            requested.append(tuple(splits))
            return original(loader, splits)
        with mock.patch.object(Dataset_Loader, 'load', spy):
            runner.train_run('orl', 'baseline', SEED, **self.args)
        self.assertEqual(requested, [('train',)])

    def test_full_cycle(self):
        with self.assertRaises(FileNotFoundError):
            runner.resume_run('orl', 'baseline', SEED, **self.args)
        runner.train_run('orl', 'baseline', SEED, **self.args)
        with self.assertRaises(FileExistsError):
            runner.train_run('orl', 'baseline', SEED, **self.args)
        with self.assertRaises(PermissionError):  # protocol not locked yet
            runner.evaluate_run('orl', 'baseline', SEED, **self.args)
        runner.lock_protocol(self.root)
        first = runner.evaluate_run('orl', 'baseline', SEED, **self.args)
        self.assertEqual(first['action'], 'evaluated')
        again = runner.evaluate_run('orl', 'baseline', SEED, **self.args)
        self.assertEqual(again['action'], 'verified')
        self.assertTrue(again['passed'])
        self.assertEqual(again['recomputed_from_predictions_difference'], 0.0)
        run_dir = runner.run_directory(self.root, 'orl', 'baseline', SEED)
        with np.load(run_dir / 'test_predictions.npz') as saved:
            self.assertEqual(saved['true_y'].size, 40)
            self.assertEqual(saved['true_label_original'].min(), 1)
            self.assertEqual(saved['true_label_original'].max(), 40)
        for name in ('config.json', 'split.json', 'environment.json', 'history.csv',
                     'train.log', 'model_summary.txt', 'checkpoints/best.pt',
                     'checkpoints/last.pt', 'test_metrics.json'):
            self.assertTrue((run_dir / name).exists(), name)
        split = json.loads((run_dir / 'split.json').read_text())
        self.assertEqual((split['optimization_samples'], split['validation_samples']), (320, 40))
        self.assertEqual(split['optimization_class_counts'], [8] * 40)
        output = summarize(self.root)
        self.assertTrue((output / 'SUMMARY.md').exists())
        self.assertTrue((output / 'figures' / 'learning_curves_orl.png').exists())

    def test_explicit_checkpoint_does_not_replace_the_formal_result(self):
        runner.train_run('orl', 'baseline', SEED, **self.args)
        runner.lock_protocol(self.root)
        run_dir = runner.run_directory(self.root, 'orl', 'baseline', SEED)
        report = runner.evaluate_run('orl', 'baseline', SEED, checkpoint=run_dir / 'checkpoints' / 'last.pt',
                                     **self.args)
        self.assertEqual(report['action'], 'evaluated')
        self.assertTrue((run_dir / 'evaluations' / 'last_metrics.json').exists())
        self.assertFalse((run_dir / 'test_metrics.json').exists())

    def test_pilot_runs_cannot_read_the_test_set(self):
        runner.train_run('orl', 'baseline', SEED, overrides=['max_epoch=1'], **self.args)
        pilot = runner.run_directory(self.root, 'orl', 'baseline', SEED, 'x')
        self.assertIn('pilot_runs', str(pilot))
        runner.lock_protocol(self.root)
        meta = {'formal': False}
        with self.assertRaises(PermissionError):
            runner._require_protocol(self.root, REGISTRY['orl']['baseline'], meta)

    def test_unfinished_run_cannot_read_the_test_set(self):
        runner.train_run('orl', 'baseline', SEED, **self.args)
        runner.lock_protocol(self.root)
        run_dir = runner.run_directory(self.root, 'orl', 'baseline', SEED)
        (run_dir / 'status.json').write_text(json.dumps({'state': 'training', 'epoch': 1}))
        with self.assertRaises(PermissionError):
            runner.evaluate_run('orl', 'baseline', SEED, **self.args)
        self.assertFalse((run_dir / 'test_metrics.json').exists())

    def test_pilot_checkpoint_cannot_stand_in_for_a_formal_run(self):
        runner.train_run('orl', 'baseline', SEED, **self.args)
        runner.train_run('orl', 'baseline', SEED, tag='pilot', **self.args)
        runner.lock_protocol(self.root)
        pilot = runner.run_directory(self.root, 'orl', 'baseline', SEED, 'pilot')
        with self.assertRaises(PermissionError):
            runner.evaluate_run('orl', 'baseline', SEED,
                                checkpoint=pilot / 'checkpoints' / 'best.pt', **self.args)

    def test_seed_outside_the_plan_needs_a_tag(self):
        with self.assertRaises(ValueError):
            runner.train_run('orl', 'baseline', 1, **self.args)

    def test_resume_rejects_a_checkpoint_from_another_config(self):
        other = replace(REGISTRY['orl']['avgpool'], max_epoch=2)
        with mock.patch.dict(REGISTRY['orl'], {'avgpool': other}):
            runner.train_run('orl', 'baseline', SEED, **self.args)
            runner.train_run('orl', 'avgpool', SEED, **self.args)
            foreign = runner.run_directory(self.root, 'orl', 'avgpool', SEED)
            with self.assertRaises(PermissionError):
                runner.resume_run('orl', 'baseline', SEED,
                                  checkpoint=foreign / 'checkpoints' / 'last.pt', **self.args)

    def test_evaluated_run_cannot_train_further(self):
        runner.train_run('orl', 'baseline', SEED, **self.args)
        runner.lock_protocol(self.root)
        runner.evaluate_run('orl', 'baseline', SEED, **self.args)
        run_dir = runner.run_directory(self.root, 'orl', 'baseline', SEED)
        earlier = load_checkpoint(run_dir / 'checkpoints' / 'last.pt')
        earlier['epoch'] = 1
        torch.save(earlier, run_dir / 'checkpoints' / 'earlier.pt')
        with self.assertRaises(PermissionError):
            runner.resume_run('orl', 'baseline', SEED,
                              checkpoint=run_dir / 'checkpoints' / 'earlier.pt', **self.args)

    def test_a_run_directory_admits_one_process(self):
        run_dir = runner.run_directory(self.root, 'orl', 'baseline', SEED)
        with exclusive_run_lock(run_dir):
            with self.assertRaises(RuntimeError):
                runner.train_run('orl', 'baseline', SEED, **self.args)

    def test_resume_of_a_finished_run_keeps_its_state(self):
        runner.train_run('orl', 'baseline', SEED, **self.args)
        runner.lock_protocol(self.root)
        runner.evaluate_run('orl', 'baseline', SEED, **self.args)
        runner.resume_run('orl', 'baseline', SEED, **self.args)
        run_dir = runner.run_directory(self.root, 'orl', 'baseline', SEED)
        self.assertEqual(json.loads((run_dir / 'status.json').read_text())['state'], 'evaluated')

    def test_stop_after_saving_recovers_on_the_next_evaluation(self):
        runner.train_run('orl', 'baseline', SEED, **self.args)
        runner.lock_protocol(self.root)
        runner.evaluate_run('orl', 'baseline', SEED, **self.args)
        run_dir = runner.run_directory(self.root, 'orl', 'baseline', SEED)
        status_path = run_dir / 'status.json'
        status = json.loads(status_path.read_text())
        status_path.write_text(json.dumps({**status, 'state': 'trained'}))
        report = runner.evaluate_run('orl', 'baseline', SEED, **self.args)
        self.assertEqual(report['action'], 'verified')
        self.assertIn('accuracy', report)
        self.assertEqual(json.loads(status_path.read_text())['state'], 'evaluated')

    def test_summary_survives_a_dataset_with_no_finished_run(self):
        runner.train_run('orl', 'baseline', SEED, **self.args)
        run_dir = runner.run_directory(self.root, 'orl', 'baseline', SEED)
        (run_dir / 'status.json').write_text(json.dumps({'state': 'failed', 'epoch': 1}))
        (run_dir / 'environment.json').unlink()
        output = summarize(self.root)
        self.assertIn('Unfinished runs: orl/baseline', (output / 'SUMMARY.md').read_text())

    def test_finished_checkpoint_repairs_an_unmarked_status(self):
        runner.train_run('orl', 'baseline', SEED, **self.args)
        run_dir = runner.run_directory(self.root, 'orl', 'baseline', SEED)
        status_path = run_dir / 'status.json'
        status_path.write_text(json.dumps({'state': 'training', 'epoch': 2}))
        (run_dir / 'history.json').write_text('[]')
        runner.resume_run('orl', 'baseline', SEED, **self.args)
        self.assertEqual(json.loads(status_path.read_text())['state'], 'trained')
        self.assertEqual(len(json.loads((run_dir / 'history.json').read_text())), 2)


if __name__ == '__main__':
    unittest.main()
