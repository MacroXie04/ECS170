import math
import unittest

import torch
from torch import nn

from local_code.stage_4_code.Generation_Decoder import decode
from local_code.stage_4_code.Method_RNN import SequenceModel


class TableModel(nn.Module):
    """A state-sensitive toy LM: each full token history has its own transition."""
    task = 'generation'
    vocab_size = 10

    def __init__(self, table=None, default=None):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.table = table or {}
        self.default = default or {3: 1.0}

    def step(self, tokens, hidden=None):
        history = [] if hidden is None else hidden[0].tolist()
        logits = []
        for token in tokens[0].tolist():
            history.append(token)
            probabilities = self.table.get(tuple(history), self.default)
            scores = torch.full((self.vocab_size,), -torch.inf)
            for value, probability in probabilities.items():
                scores[value] = math.log(probability)
            logits.append(scores)
        # A tuple hidden state deliberately checks LSTM-style branch handling.
        return torch.stack(logits)[None, :, :], (torch.tensor(history), torch.tensor([len(history)]))


class DecoderTests(unittest.TestCase):
    prefix = [2, 4, 5, 6]

    def test_beam_tracks_branch_state_and_beats_greedy(self):
        p = tuple(self.prefix)
        model = TableModel({p: {7: .6, 8: .4}, p + (7,): {9: .6, 3: .4},
                            p + (8,): {3: .99, 9: .01}})
        common = dict(max_new_tokens=3, min_new_tokens=0, length_penalty=0,
                      repetition_penalty=1, no_repeat_ngram_size=0)
        greedy = decode(model, self.prefix, strategy='greedy', **common)
        beam = decode(model, self.prefix, strategy='beam', beam_width=2, **common)
        self.assertEqual(greedy['ids'], self.prefix + [7, 9, 3])
        self.assertEqual(beam['ids'], self.prefix + [8, 3])
        self.assertGreater(beam['log_probability'], greedy['log_probability'])
        self.assertAlmostEqual(beam['log_probability'], math.log(.4 * .99), places=6)
        self.assertTrue(beam['ended_with_eos'])

    def test_length_penalty_changes_completed_hypothesis_ranking(self):
        p = tuple(self.prefix)
        # Short path probability .55; longer path .45. A large length penalty
        # may prefer the longer completed hypothesis, but never loses its state.
        model = TableModel({p: {3: .55, 7: .45}, p + (7,): {8: 1.0},
                            p + (7, 8): {9: 1.0}, p + (7, 8, 9): {3: 1.0}})
        common = dict(strategy='beam', beam_width=2, max_new_tokens=4,
                      min_new_tokens=0, repetition_penalty=1, no_repeat_ngram_size=0)
        short = decode(model, self.prefix, length_penalty=0, **common)
        long = decode(model, self.prefix, length_penalty=2, **common)
        self.assertEqual(short['ids'], self.prefix + [3])
        self.assertEqual(long['ids'], self.prefix + [7, 8, 9, 3])
        self.assertAlmostEqual(long['normalized_score'], math.log(.45) / (9 / 6) ** 2, places=6)

    def test_special_tokens_blocked_and_minimum_tokens_before_eos(self):
        model = TableModel(default={0: .4, 1: .3, 2: .2, 3: .09, 7: .01})
        for strategy in ('greedy', 'beam', 'sample'):
            result = decode(model, self.prefix, strategy=strategy, max_new_tokens=5,
                            min_new_tokens=2, top_k=1, repetition_penalty=1,
                            no_repeat_ngram_size=0, length_penalty=0)
            self.assertEqual(result['ids'], self.prefix + [7, 7, 3])
            self.assertEqual(result['finish_reason'], 'eos')

    def test_repeated_ngram_constraints_include_prefix(self):
        model = TableModel(default={6: 1.0})
        for strategy in ('greedy', 'beam', 'sample'):
            result = decode(model, self.prefix, strategy=strategy, max_new_tokens=8,
                            min_new_tokens=0, no_repeat_ngram_size=2, repetition_penalty=1)
            pairs = list(zip(result['ids'], result['ids'][1:]))
            self.assertEqual(len(pairs), len(set(pairs)))
            self.assertEqual(result['finish_reason'], 'constraints_exhausted')
            self.assertFalse(result['ended_with_eos'])

    def test_maximum_does_not_fabricate_eos(self):
        model = TableModel(default={7: 1.0})
        for strategy in ('greedy', 'beam', 'sample'):
            result = decode(model, self.prefix, strategy=strategy, max_new_tokens=2,
                            min_new_tokens=0, no_repeat_ngram_size=0)
            self.assertEqual(result['ids'], self.prefix + [7, 7])
            self.assertEqual(result['generated_token_count'], 2)
            self.assertEqual(result['finish_reason'], 'max_new_tokens')
            self.assertFalse(result['ended_with_eos'])

    def test_nucleus_sampling_private_rng_and_training_mode(self):
        model = TableModel(default={7: .6, 8: .3, 9: .1}).train()
        state = torch.get_rng_state().clone()
        kwargs = dict(strategy='sample', max_new_tokens=6, min_new_tokens=0,
                      top_k=0, top_p=1, temperature=1, repetition_penalty=1,
                      no_repeat_ngram_size=0, seed=317)
        one = decode(model, self.prefix, **kwargs)
        two = decode(model, self.prefix, **kwargs)
        self.assertEqual(one, two)
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
        self.assertTrue(model.training)
        nucleus = decode(model, self.prefix, **{**kwargs, 'top_p': .5})
        self.assertEqual(nucleus['ids'], self.prefix + [7] * 6)

    def test_real_recurrent_cells_and_tuple_hidden(self):
        torch.manual_seed(19)
        for cell in ('rnn', 'lstm', 'gru'):
            model = SequenceModel(10, task='generation', cell=cell,
                                  embedding_dim=5, hidden_dim=7, dropout=.2).train()
            result = decode(model, self.prefix, strategy='beam', beam_width=3,
                            min_new_tokens=2, max_new_tokens=8)
            self.assertEqual(result['ids'][:len(self.prefix)], self.prefix)
            self.assertTrue(math.isfinite(result['log_probability']))
            self.assertTrue(math.isfinite(result['normalized_score']))
            self.assertTrue(model.training)
            self.assertFalse(set(result['ids'][len(self.prefix):]) & {0, 1, 2})

    def test_existing_eos_and_zero_budget(self):
        model = TableModel()
        result = decode(model, self.prefix + [3])
        self.assertEqual(result['ids'], self.prefix + [3])
        self.assertEqual(result['finish_reason'], 'prefix_eos')
        empty = decode(model, self.prefix, min_new_tokens=0, max_new_tokens=0)
        self.assertEqual(empty['ids'], self.prefix)
        self.assertFalse(empty['ended_with_eos'])

    def test_invalid_arguments_and_model_failure_restore_mode(self):
        model = TableModel().train()
        for kwargs in ({'top_p': 0}, {'temperature': 0}, {'repetition_penalty': .5},
                       {'beam_width': 0}, {'min_new_tokens': 8, 'max_new_tokens': 7},
                       {'strategy': 'retrieve'}, {'max_new_tokens': 1.2}):
            with self.assertRaises(ValueError):
                decode(model, self.prefix, **kwargs)
        for prefix in ([], [2, 4, 1], [2, 4, 3, 5], [0, 4, 5], [2, 10, 4]):
            with self.assertRaises(ValueError):
                decode(model, prefix)
        class Broken(TableModel):
            def step(self, tokens, hidden=None):
                raise RuntimeError('deliberate model failure')
        broken = Broken().train()
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            decode(broken, self.prefix)
        self.assertTrue(broken.training)


if __name__ == '__main__':
    unittest.main()
