import copy
import math
import random
import unittest

import torch
from torch import nn

from local_code.stage_4_code.Dataset_Loader import Vocabulary
from local_code.stage_4_code.Novel_Generation import (
    TrainingNoveltyIndex, generate_candidates, lexical_tokens, select_candidate,
)


class ToyModel(nn.Module):
    task = 'generation'

    def __init__(self, vocab_size, deterministic=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size
        self.deterministic = deterministic

    def step(self, tokens, hidden=None):
        history = [] if hidden is None else hidden[0].tolist()
        rows = []
        for token in tokens[0].tolist():
            history.append(token)
            emitted = max(0, len(history) - 4)
            scores = torch.full((self.vocab_size,), -torch.inf)
            if self.deterministic:
                # BOS remains in the original model distribution although decoding
                # blocks it: returned likelihood must not renormalize it away.
                scores[2] = math.log(.3)
                scores[7 + emitted if emitted < 4 else 3] = math.log(.7)
            else:
                for word in range(7, 13):
                    scores[word] = math.log((.2 if emitted >= 4 else 1.) / 6)
                if emitted >= 4:
                    scores[3] = math.log(.8)
            rows.append(scores)
        return torch.stack(rows)[None], (torch.tensor(history),)


class NovelGenerationTests(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary(['my', 'little', 'robot', 'eats', 'purple', 'paper',
                                 'under', 'clouds', 'today', '!', '.'])
        self.training = [{'id': 'train/1', 'text': 'Old jokes never enter this example corpus.'}]

    def candidate(self, text='my little robot eats purple paper under clouds', score=-2.):
        continuation = ' '.join(text.split()[3:])
        return dict(candidate_id=0, text=text, continuation=continuation,
                    ended_with_eos=True, new_word_count=len(lexical_tokens(continuation)),
                    repeated_trigrams=0, average_log_probability=score,
                    novelty=TrainingNoveltyIndex(self.training).assess(text, continuation))

    def test_punctuation_and_case_cannot_disguise_exact_copy(self):
        index = TrainingNoveltyIndex([{'id': 'copy', 'text': "I can't wait: robots borrow umbrellas at noon!"}])
        result = index.assess('i cant wait — ROBOTS borrow umbrellas at NOON.',
                              'ROBOTS, borrow umbrellas at NOON?')
        self.assertTrue(result['full_text']['exact_copy'])
        self.assertEqual(result['full_text']['exact_match_ids'], ['copy'])
        self.assertTrue(result['continuation']['exact_copy'])
        self.assertEqual(result['continuation']['nearest_similarity'], 1.)

    def test_changed_prompt_cannot_hide_copied_suffix(self):
        index = TrainingNoveltyIndex([{'id': 'suffix', 'text':
            'What did Alice say about robots dancing quietly in the rain?'}])
        result = index.assess('My silly brother dancing quietly in the rain!',
                              'dancing quietly in the rain!')
        self.assertFalse(result['full_text']['exact_copy'])
        self.assertFalse(result['continuation']['exact_copy'])
        self.assertTrue(result['continuation']['exact_training_suffix'])
        self.assertTrue(result['continuation']['exact_training_span'])
        self.assertEqual(result['continuation']['exact_suffix_match_ids'], ['suffix'])
        self.assertEqual(result['continuation']['shared_fraction'], 1.)
        candidate = self.candidate()
        candidate['novelty'] = result
        self.assertIsNone(select_candidate([candidate])['selected'])

    def test_common_prompt_is_excluded_from_continuation_similarity(self):
        result = TrainingNoveltyIndex([{'id': 'prompt', 'text':
            'The little robot eats pudding behind glass doors.'}]).assess(
            'The little robot solves puzzles above dusty rooftops.',
            'solves puzzles above dusty rooftops.')
        self.assertGreater(result['full_text']['nearest_similarity'], 0.)
        self.assertEqual(result['continuation']['nearest_similarity'], 0.)
        self.assertEqual(result['continuation']['shared_fraction'], 0.)

    def test_known_model_probability_keeps_original_distribution(self):
        model = ToyModel(len(self.vocab), deterministic=True)
        result = generate_candidates(model, self.vocab, 'my little robot', self.training,
                                     num_samples=2, max_new_tokens=8)
        self.assertEqual(len(result), 1)
        candidate = result[0]
        self.assertEqual(candidate['ids'], [2, 4, 5, 6, 7, 8, 9, 10, 3])
        self.assertAlmostEqual(candidate['log_probability'], 5 * math.log(.7), places=6)
        self.assertAlmostEqual(candidate['average_log_probability'], math.log(.7), places=6)
        self.assertEqual(candidate['new_word_count'], 4)
        self.assertEqual(candidate['generated_token_count'], 5)
        self.assertTrue(candidate['ended_with_eos'])
        self.assertEqual([source['seed'] for source in candidate['sources']], [1704, 1705, 1706])

    def test_deterministic_sampling_preserves_rng_mode_and_training_data(self):
        model = ToyModel(len(self.vocab)).train()
        before = copy.deepcopy(self.training)
        torch_state, python_state = torch.get_rng_state().clone(), random.getstate()
        first = generate_candidates(model, self.vocab, 'my little robot', self.training,
                                    num_samples=8, max_new_tokens=8, seed=42)
        second = generate_candidates(model, self.vocab, 'my little robot', self.training,
                                     num_samples=8, max_new_tokens=8, seed=42)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 1)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(model.training)
        self.assertEqual(before, self.training)
        self.assertEqual(sum(len(row['sources']) for row in first), 9)

    def test_no_eligible_candidate_is_explicit_and_no_input_mutation(self):
        candidate = self.candidate()
        candidate['ended_with_eos'] = False
        candidate['repeated_trigrams'] = 1
        before = copy.deepcopy(candidate)
        result = select_candidate([candidate])
        self.assertEqual(result['status'], 'no_eligible_candidate')
        self.assertIsNone(result['selected'])
        self.assertEqual(result['eligible_count'], 0)
        self.assertEqual(result['rejections'][0]['reasons'], ['incomplete', 'repeated_trigrams'])
        self.assertEqual(candidate, before)
        self.assertEqual(select_candidate([])['status'], 'no_eligible_candidate')

    def test_eligible_ranking_uses_mean_model_likelihood(self):
        low, high = self.candidate(score=-3.), self.candidate(score=-1.)
        high['candidate_id'] = 1
        result = select_candidate([low, high])
        self.assertIs(result['selected'], high)
        self.assertEqual(result['eligible_candidate_ids'], [1, 0])
        # Better likelihood cannot overcome a known copied continuation.
        high['novelty']['continuation']['exact_training_span'] = True
        self.assertIs(select_candidate([low, high])['selected'], low)

    def test_four_word_boilerplate_does_not_exceed_default_span_limit(self):
        index = TrainingNoveltyIndex([{'id': 'boilerplate', 'text':
            'Some old joke contains the phrase under the cloudy sky with several unrelated details.'}])
        candidate = self.candidate('my little robot under the cloudy sky laughs')
        candidate['novelty'] = index.assess(candidate['text'], candidate['continuation'])
        self.assertEqual(candidate['novelty']['continuation']['max_shared_token_count'], 4)
        self.assertEqual(candidate['novelty']['continuation']['shared_fraction'], .8)
        self.assertEqual(select_candidate([candidate])['status'], 'selected')

    def test_incomplete_results_are_kept_for_audit(self):
        result = generate_candidates(ToyModel(len(self.vocab)), self.vocab,
                                     'my little robot', self.training,
                                     num_samples=0, max_new_tokens=2)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]['ended_with_eos'])
        self.assertIsNone(select_candidate(result)['selected'])

    def test_requires_three_words_and_training_only_index(self):
        model = ToyModel(len(self.vocab))
        for prompt in ('my robot', 'my little robot eats', 'my unknown robot'):
            with self.assertRaises(ValueError):
                generate_candidates(model, self.vocab, prompt, self.training, num_samples=0)
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            TrainingNoveltyIndex([dict(self.training[0], split='validation')])
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            TrainingNoveltyIndex([])


if __name__ == '__main__':
    unittest.main()
