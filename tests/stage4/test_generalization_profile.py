"""Fail-closed profile bindings and prediction contracts; no quality assertions."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from local_code.stage_4_code.Dataset_Loader import Vocabulary
from local_code.stage_4_code.Novel_Generation import TrainingNoveltyIndex
from script.stage_4_script import predict_novel_generation as cli


class GeneralizationProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(cli, 'ROOT', self.root).start()
        self.data = {
            'train': [dict(id='train/1', text='What do you call a silent detective? A quiet investigator.')],
            'vocab': Vocabulary(['what', 'do', 'you', 'call', 'a', 'silent', 'detective', '?',
                                 'quiet', 'investigator', '.']),
            'metadata': {
                'source_sha256': 'a' * 64,
                'vocabulary_sha256': 'b' * 64,
                'split_id_sha256': {'train': 'c' * 64, 'validation': 'd' * 64, 'test': 'e' * 64},
            },
        }
        source_names = set(cli.REQUIRED_SOURCES) | {
            'local_code/stage_4_code/Generation_Ensemble.py',
            'local_code/stage_4_code/Regularized_Generation.py',
        }
        sources = {}
        for name in source_names:
            source = self.root / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text('# Frozen unit fixture: ' + name + '\n')
            sources[name] = cli.digest(source)
        self.checkpoint_name = 'result/unit/checkpoints/best.pt'
        self.checkpoint = self.root / self.checkpoint_name
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b'checkpoint fixture - model load mocked')
        self.config_path = self.checkpoint.parent.parent / 'config.json'
        self.config = {'data_metadata': copy.deepcopy(self.data['metadata'])}
        self.config_path.write_text(json.dumps(self.config))
        self.profile = {
            'schema_version': 1, 'mode': 'screened', 'candidate': 'fixture_screened',
            'data_binding': dict(copy.deepcopy(self.data['metadata']),
                                 training_rows_sha256=cli.training_digest(self.data['train'])),
            'source_sha256': sources,
            'model_spec': {'name': 'fixture', 'components': [
                {'loader': 'regularized', 'checkpoint': self.checkpoint_name,
                 'checkpoint_sha256': cli.digest(self.checkpoint)}]},
            'sampling': {'num_samples': 0, 'seed': 1704, 'temperature': .8,
                         'top_p': .9, 'top_k': 40, 'max_new_tokens': 48},
            'screening': {'max_similarity': .8, 'min_content_words': 5},
        }
        self.profile_path = self.root / 'profile.json'
        self.save_profile()

    def save_profile(self):
        self.profile_path.write_text(json.dumps(self.profile))

    def copied_candidate(self):
        text = self.data['train'][0]['text']
        continuation = 'call a silent detective? A quiet investigator.'
        return dict(candidate_id=0, text=text, continuation=continuation,
                    ended_with_eos=True, new_word_count=8, repeated_trigrams=0,
                    average_log_probability=-.1,
                    novelty=TrainingNoveltyIndex(self.data['train']).assess(text, continuation))

    def mocked_predict(self, pool, **kwargs):
        with patch.object(cli, 'Dataset_Loader') as loader, \
             patch.object(cli, 'load_model', return_value=(object(), self.data['vocab'], [])), \
             patch.object(cli, 'generate_candidates', return_value=pool) as generation:
            loader.return_value.load_generation.return_value = self.data
            result = cli.predict('What do you', profile=self.profile_path, device='cpu', **kwargs)
        return result, generation

    def test_valid_profile_is_nonmutating(self):
        before_profile, before_data = copy.deepcopy(self.profile), copy.deepcopy(self.data)
        cli.verify_profile(self.profile, self.data)
        self.assertEqual(self.profile, before_profile)
        self.assertEqual(self.data['train'], before_data['train'])
        self.assertEqual(self.data['metadata'], before_data['metadata'])

    def test_training_reference_and_each_dataset_binding_are_checked(self):
        changed = copy.deepcopy(self.data)
        changed['train'][0]['text'] += ' Replaced content.'
        with self.assertRaisesRegex(ValueError, 'Training novelty reference'):
            cli.verify_profile(self.profile, changed)
        for key in ('source_sha256', 'vocabulary_sha256', 'split_id_sha256'):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.data)
                changed['metadata'][key] = 'changed'
                with self.assertRaisesRegex(ValueError, 'Dataset binding changed'):
                    cli.verify_profile(self.profile, changed)

    def test_required_source_cannot_be_omitted_and_changed_source_is_rejected(self):
        for name in (next(iter(cli.REQUIRED_SOURCES)),
                     'local_code/stage_4_code/Regularized_Generation.py'):
            with self.subTest(name=name):
                profile = copy.deepcopy(self.profile)
                profile['source_sha256'].pop(name)
                with self.assertRaisesRegex(ValueError, 'missing required source'):
                    cli.verify_profile(profile, self.data)
        source = self.root / 'local_code/stage_4_code/Novel_Generation.py'
        source.write_text('# Changed decoder\n')
        with self.assertRaisesRegex(ValueError, 'Generation source changed'):
            cli.verify_profile(self.profile, self.data)

    def test_mixture_requires_ensemble_source_and_valid_components(self):
        profile = copy.deepcopy(self.profile)
        profile['model_spec']['components'] *= 2
        profile['model_spec']['late_weight'] = .5
        profile['source_sha256'].pop('local_code/stage_4_code/Generation_Ensemble.py')
        with self.assertRaisesRegex(ValueError, 'missing required source'):
            cli.verify_profile(profile, self.data)
        for components in ([], self.profile['model_spec']['components'] * 3,
                           [dict(self.profile['model_spec']['components'][0], loader='unknown')]):
            profile = copy.deepcopy(self.profile)
            profile['model_spec']['components'] = components
            with self.assertRaisesRegex(ValueError, 'supported loaders'):
                cli.verify_profile(profile, self.data)

    def test_source_and_checkpoint_cannot_escape_repository(self):
        profile = copy.deepcopy(self.profile)
        profile['source_sha256']['../outside.py'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'inside this repository'):
            cli.verify_profile(profile, self.data)
        profile = copy.deepcopy(self.profile)
        profile['model_spec']['components'][0]['checkpoint'] = '../outside.pt'
        with self.assertRaisesRegex(ValueError, 'inside this repository'):
            cli.verify_profile(profile, self.data)
        # resolve() must check symlink destinations, not just a lexical prefix.
        outside = self.root.parent / (self.root.name + '-outside')
        outside.write_bytes(b'outside')
        self.addCleanup(outside.unlink)
        (self.root / 'linked.pt').symlink_to(outside)
        profile['model_spec']['components'][0]['checkpoint'] = 'linked.pt'
        with self.assertRaisesRegex(ValueError, 'inside this repository'):
            cli.verify_profile(profile, self.data)

    def test_checkpoint_bytes_and_each_checkpoint_dataset_binding_are_checked(self):
        self.checkpoint.write_bytes(b'replaced checkpoint')
        with self.assertRaisesRegex(ValueError, 'Selected checkpoint changed'):
            cli.verify_profile(self.profile, self.data)
        self.checkpoint.write_bytes(b'checkpoint fixture - model load mocked')
        for key in ('source_sha256', 'vocabulary_sha256', 'split_id_sha256'):
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config['data_metadata'][key] = 'changed'
                self.config_path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, 'Checkpoint data binding changed'):
                    cli.verify_profile(self.profile, self.data)

    def test_no_eligible_candidate_returns_null_and_does_not_fallback(self):
        pool = [self.copied_candidate()]
        before = copy.deepcopy(pool)
        result, _ = self.mocked_predict(pool)
        self.assertEqual(result['status'], 'no_eligible_candidate')
        self.assertIsNone(result['text'])
        self.assertIsNone(result['result'])
        self.assertEqual(result['selection']['eligible_count'], 0)
        self.assertIn('exact_training_copy', result['selection']['rejections'][0]['reasons'])
        self.assertEqual(pool, before)

    def test_strict_override_preserves_profile_and_seed_override_is_explicit(self):
        self.profile['mode'] = 'beam'; self.save_profile()
        saved = self.profile_path.read_bytes()
        ordinary, _ = self.mocked_predict([self.copied_candidate()])
        self.assertEqual(ordinary['status'], 'selected')
        self.assertTrue(ordinary['result']['novelty']['full_text']['exact_copy'])
        strict, generation = self.mocked_predict([self.copied_candidate()], strict_novelty=True, seed=72)
        self.assertEqual(strict['mode'], 'screened')
        self.assertEqual(strict['status'], 'no_eligible_candidate')
        self.assertEqual(strict['sampling']['seed'], 72)
        self.assertEqual(generation.call_args.kwargs['seed'], 72)
        self.assertEqual(self.profile_path.read_bytes(), saved)
        self.assertEqual(strict['profile_sha256'], hashlib.sha256(saved).hexdigest())

    def test_empty_candidate_pool_and_mismatched_vocabulary_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'no auditable candidates'):
            self.mocked_predict([])
        with patch.object(cli, 'Dataset_Loader') as loader, \
             patch.object(cli, 'load_model', return_value=(object(), Vocabulary(['different']), [])), \
             patch.object(cli, 'generate_candidates') as generation:
            loader.return_value.load_generation.return_value = self.data
            with self.assertRaisesRegex(ValueError, 'Model vocabulary differs'):
                cli.predict('What do you', profile=self.profile_path, device='cpu')
            generation.assert_not_called()


if __name__ == '__main__': unittest.main()
