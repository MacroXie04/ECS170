"""Logic and stage-order tests for the ORL follow-up; no training or test data.

    python -m unittest discover -s tests/stage3 -p 'test_orl_followup.py'
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'run_orl_followup', REPO / 'script' / 'stage_3_followup' / 'run_orl_followup.py')
followup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(followup)


class FoldTests(unittest.TestCase):
    def test_folds_partition_the_data_one_image_per_person(self):
        labels = np.repeat(np.arange(40), 9)
        folds = followup.make_folds(labels)
        self.assertEqual(len(folds), 9)
        self.assertEqual(sorted(np.concatenate(folds).tolist()), list(range(360)))
        for held in folds:
            np.testing.assert_array_equal(np.sort(labels[held]), np.arange(40))

    def test_folds_are_stable(self):
        labels = np.repeat(np.arange(40), 9)
        for first, second in zip(followup.make_folds(labels), followup.make_folds(labels)):
            np.testing.assert_array_equal(first, second)


class SelectionTests(unittest.TestCase):
    def test_rule_order(self):
        rows = [{'config': 'loss', 'correct': 350, 'mean_cross_entropy': 0.3, 'parameters': 1},
                {'config': 'big', 'correct': 350, 'mean_cross_entropy': 0.2, 'parameters': 9},
                {'config': 'small', 'correct': 350, 'mean_cross_entropy': 0.2, 'parameters': 1},
                {'config': 'worse', 'correct': 349, 'mean_cross_entropy': 0.1, 'parameters': 1}]
        self.assertEqual(followup.select_config(rows)['config'], 'small')

    def test_ensemble_averages_probabilities(self):
        first = np.array([[0.9, 0.1], [0.6, 0.4]])
        second = np.array([[0.2, 0.8], [0.1, 0.9]])
        third = np.array([[0.2, 0.8], [0.7, 0.3]])
        predicted = followup.ensemble_predictions(np.stack([first, second, third]))
        np.testing.assert_array_equal(predicted, [1, 1])

    def test_history_columns_say_what_they_measured(self):
        renamed = followup.renamed_history([{'epoch': 1, 'val_loss': 0.5, 'selected_so_far': 1}],
                                           'train_eval_')
        self.assertEqual(renamed, [{'epoch': 1, 'train_eval_loss': 0.5}])


class StageOrderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        patcher = mock.patch.object(followup, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_stages_refuse_to_start_unregistered(self):
        for stage in (lambda: followup.stage_cv('cpu', 1, 1),
                      lambda: followup.stage_refit('cpu', 1),
                      lambda: followup.stage_test('cpu', 1)):
            with self.assertRaises(PermissionError):
                stage()

    def test_register_runs_once(self):
        followup.stage_register('cpu')
        with self.assertRaises(FileExistsError):
            followup.stage_register('cpu')

    def test_refit_needs_a_selection(self):
        followup.stage_register('cpu')
        with self.assertRaises(PermissionError):
            followup.stage_refit('cpu', 1)

    def test_test_stage_needs_finished_refits(self):
        followup.stage_register('cpu')
        (self.root / 'selection.json').write_text('{"config": "baseline"}')
        with self.assertRaises(PermissionError):
            followup.stage_test('cpu', 1)
        self.assertFalse((self.root / 'test_access_log.jsonl').exists())

    def test_test_stage_runs_once(self):
        followup.stage_register('cpu')
        (self.root / 'test_results.json').write_text('{}')
        with self.assertRaises(FileExistsError):
            followup.stage_test('cpu', 1)

    def test_a_changed_script_stops_the_stages(self):
        followup.stage_register('cpu')
        with mock.patch.object(followup, 'sha256_file', return_value='different'):
            with self.assertRaises(PermissionError):
                followup.require_registered()


if __name__ == '__main__':
    unittest.main()
