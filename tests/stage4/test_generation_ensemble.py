import unittest

import torch
from torch import nn

from local_code.stage_4_code.Generation_Ensemble import MixtureModel
from local_code.stage_4_code.Generation_Decoder import decode
from local_code.stage_4_code.Method_RNN import SequenceModel


class ConstantModel(nn.Module):
    task = 'generation'

    def __init__(self, probabilities):
        super().__init__()
        self.vocab_size = len(probabilities)
        self.logits = nn.Parameter(torch.tensor(probabilities, dtype=torch.float64).log())

    def forward(self, tokens, lengths=None):
        return self.logits.expand(*tokens.shape, self.vocab_size)

    def step(self, tokens, hidden=None):
        count = 0 if hidden is None else hidden.item()
        return self.forward(tokens), torch.tensor([count + tokens.shape[1]])


class EnsembleTests(unittest.TestCase):
    early = [.1, .1, .1, .2, .3, .1, .1]
    late = [.05, .05, .05, .4, .05, .2, .2]

    def test_exact_probability_mixture_and_endpoints(self):
        tokens = torch.tensor([[2, 4, 5], [2, 6, 4]])
        for weight in (0., .25, .5, .75, 1.):
            mixture = MixtureModel([ConstantModel(self.early), ConstantModel(self.late)], weight)
            actual = mixture(tokens).exp()
            expected = ((1 - weight) * torch.tensor(self.early, dtype=torch.float64)
                        + weight * torch.tensor(self.late, dtype=torch.float64))
            torch.testing.assert_close(actual, expected.expand_as(actual), rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual.sum(-1)), rtol=1e-12, atol=1e-12)

    def test_step_forward_agreement_with_independent_nested_states(self):
        torch.manual_seed(619)
        models = [SequenceModel(11, task='generation', cell=cell, embedding_dim=5,
                                hidden_dim=7, dropout=0) for cell in ('gru', 'lstm')]
        mixture = MixtureModel(models, .75).eval()
        tokens = torch.tensor([[2, 4, 5, 6, 7], [2, 7, 6, 4, 8]])
        full = mixture(tokens)
        state = None
        steps = []
        for i in range(tokens.shape[1]):
            output, state = mixture.step(tokens[:, i:i+1], state)
            steps.append(output)
        torch.testing.assert_close(torch.cat(steps, dim=1), full, rtol=1e-5, atol=1e-6)
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 2)
        self.assertIsInstance(state[0], torch.Tensor)
        self.assertIsInstance(state[1], tuple)
        # Continuing the mixture must use the state of the matching component.
        next_token = torch.tensor([[9], [10]])
        result, _ = mixture.step(next_token, state)
        component = [model.step(next_token, own_state)[0] for model, own_state in zip(models, state)]
        expected = (.25 * component[0].softmax(-1) + .75 * component[1].softmax(-1)).log()
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_state_counters_remain_independent(self):
        mixture = MixtureModel([ConstantModel(self.early), ConstantModel(self.late)], .5)
        tokens = torch.tensor([[4, 5]])
        _, state = mixture.step(tokens, (torch.tensor([3]), torch.tensor([19])))
        self.assertEqual([s.item() for s in state], [5, 21])

    def test_shared_decoder_special_mask_and_tuple_compatibility(self):
        # Specials carry most of the probability; the shared decoder must mask
        # them, preserve the prompt, and respect independent mixture states.
        models = [ConstantModel([.4,.2,.2,.1,.05,.03,.02]),
                  ConstantModel([.3,.3,.2,.1,.04,.04,.02])]
        mixture = MixtureModel(models, .5).train()
        prefix = [2, 4, 5, 6]
        for strategy in ('greedy', 'beam', 'sample'):
            result = decode(mixture, prefix, strategy=strategy, min_new_tokens=2,
                            max_new_tokens=8, top_k=1, no_repeat_ngram_size=0)
            self.assertEqual(result['ids'][:len(prefix)], prefix)
            self.assertFalse({0,1,2} & set(result['ids'][len(prefix):]))
            self.assertTrue(result['ended_with_eos'])
            self.assertTrue(mixture.training)

    def test_invalid_model_configuration(self):
        early, late = ConstantModel(self.early), ConstantModel(self.late)
        for weight in (-.1, 1.1, float('nan'), True):
            with self.assertRaises(ValueError):
                MixtureModel([early, late], weight)
        with self.assertRaises(ValueError):
            MixtureModel([early])
        with self.assertRaises(ValueError):
            MixtureModel([early, ConstantModel([.2] * 5)])
        mixture = MixtureModel([early, late])
        with self.assertRaises(ValueError):
            mixture.step(torch.tensor([[2, 4]]), (None,))


if __name__ == '__main__':
    unittest.main()
