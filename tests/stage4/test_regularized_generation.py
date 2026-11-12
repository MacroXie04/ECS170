import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from local_code.stage_4_code.Dataset_Loader import Vocabulary
from local_code.stage_4_code.Regularized_Generation import (
    TiedGenerationGRU, config_digest, vocabulary_digest, load_regularized_checkpoint,
)


class RegularizedGenerationTests(unittest.TestCase):
    def make_checkpoint(self, directory):
        run = Path(directory) / 'run'
        (run / 'checkpoints').mkdir(parents=True)
        vocab = Vocabulary(['what', 'do', 'you', 'call', 'a', 'cat', '?'])
        kwargs = dict(vocab_size=len(vocab), dimension=8, dropout=.2)
        config = dict(architecture='tied_gru_v1', model=kwargs,
            data_metadata={'vocabulary_sha256': vocabulary_digest(vocab)})
        (run / 'config.json').write_text(json.dumps(config))
        checkpoint = run / 'checkpoints/best.pt'
        model = TiedGenerationGRU(**kwargs)
        torch.save(dict(model=model.state_dict(), vocab=vocab.to_dict(), epoch=3,
                        config_hash=config_digest(config)), checkpoint)
        return checkpoint, config

    def test_tied_weights_pad_gradient_and_adamw_update(self):
        model = TiedGenerationGRU(10, dimension=8, dropout=0.)
        self.assertIs(model.embedding.weight, model.output.weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
        before = model.embedding.weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.tensor([[2, 4, 5, 0], [2, 7, 0, 0]]))
            targets = torch.tensor([[4, 5, 3, 0], [7, 3, 0, 0]])
            torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                              ignore_index=0).backward()
            self.assertEqual(torch.count_nonzero(model.embedding.weight.grad[0]).item(), 0)
            self.assertGreater(torch.count_nonzero(model.embedding.weight.grad[6]).item(), 0)
            optimizer.step()
            self.assertEqual(torch.count_nonzero(model.embedding.weight[0]).item(), 0)
        self.assertFalse(torch.equal(before[6], model.embedding.weight[6]))

    def test_full_sequence_equals_incremental_step(self):
        model = TiedGenerationGRU(12, dimension=8, dropout=.4).eval()
        tokens = torch.tensor([[2, 4, 8, 5, 7], [2, 6, 9, 5, 4]])
        whole = model(tokens)
        hidden, parts = None, []
        for index in range(tokens.shape[1]):
            logits, hidden = model.step(tokens[:, index:index + 1], hidden)
            parts.append(logits)
        torch.testing.assert_close(whole, torch.cat(parts, dim=1), rtol=1e-5, atol=1e-7)

    def test_load_is_bound_preserves_rng_and_rejects_config_change(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, config = self.make_checkpoint(directory)
            before = torch.get_rng_state()
            model, vocab, metadata = load_regularized_checkpoint(checkpoint)
            torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
            self.assertEqual(metadata['epoch'], 3)
            self.assertIs(model.embedding.weight, model.output.weight)
            config['model']['dropout'] = .5
            (checkpoint.parent.parent / 'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'configuration'):
                load_regularized_checkpoint(checkpoint)

    def test_reject_vocabulary_source_and_untied_state(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, config = self.make_checkpoint(directory)
            saved = torch.load(checkpoint, weights_only=True)
            saved['vocab']['tokens'][-1] = 'replacement'
            torch.save(saved, checkpoint)
            with self.assertRaisesRegex(ValueError, 'vocabulary'):
                load_regularized_checkpoint(checkpoint)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, config = self.make_checkpoint(directory)
            saved = torch.load(checkpoint, weights_only=True)
            saved['model']['output.weight'] = saved['model']['output.weight'].clone() + 1.
            torch.save(saved, checkpoint)
            with self.assertRaisesRegex(ValueError, 'not tied'):
                load_regularized_checkpoint(checkpoint)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, config = self.make_checkpoint(directory)
            config['source_sha256'] = {'local_code/stage_4_code/Regularized_Generation.py': '0' * 64}
            (checkpoint.parent.parent / 'config.json').write_text(json.dumps(config))
            saved = torch.load(checkpoint, weights_only=True)
            saved['config_hash'] = config_digest(config)
            torch.save(saved, checkpoint)
            with self.assertRaisesRegex(ValueError, 'source changed'):
                load_regularized_checkpoint(checkpoint)


if __name__ == '__main__': unittest.main()
