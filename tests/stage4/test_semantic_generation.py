import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from local_code.stage_4_code.Dataset_Loader import Vocabulary
import local_code.stage_4_code.Semantic_Generation as semantic


class SemanticGenerationTests(unittest.TestCase):
    def make_fixture(self,directory):
        root=Path(directory);external=root/'external';external.mkdir()
        vectors=external/'vectors.txt';vectors.write_text('what 1 0 0 0\ndo 0 2 0 0\nyou 0 0 3 0\n')
        manifest=external/'manifest.json'
        manifest.write_text(json.dumps(dict(member=vectors.name,member_sha256=semantic.file_digest(vectors),
            source='unit fixture, not external GloVe')))
        vocab=Vocabulary(['what','do','you','call','a','cat','?'])
        prior,matched,metadata=semantic.load_word_prior(vocab,vectors,manifest,dimension=4)
        metadata.update(vector_file=str(vectors.relative_to(root)),manifest_file=str(manifest.relative_to(root)))
        kwargs=dict(vocab_size=len(vocab),embedding_dim=4,hidden_dim=6,dropout=.2)
        model=semantic.SemanticGenerationGRU(**kwargs);model.initialize_prior(prior,matched)
        sources=['local_code/stage_4_code/Semantic_Generation.py','local_code/stage_4_code/Dataset_Loader.py',
            'script/stage_4_script/train_semantic_generation.py','local_code/stage_4_code/Experiment_Runner.py']
        for name in sources:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture '+name)
        config=dict(architecture='semantic_prior_gru_v1',model=kwargs,pretrained=metadata,
            data_metadata=dict(vocabulary_sha256=semantic.vocabulary_digest(vocab)),
            source_sha256={name:semantic.file_digest(root/name) for name in sources})
        run=root/'run';(run/'checkpoints').mkdir(parents=True)
        (run/'config.json').write_text(json.dumps(config))
        checkpoint=run/'checkpoints/best.pt'
        torch.save(dict(model=model.state_dict(),vocab=vocab.to_dict(),epoch=3,config_hash=semantic.canonical_digest(config)),checkpoint)
        return root,checkpoint,config,model

    def test_prior_normalization_coverage_and_special_invariants(self):
        with tempfile.TemporaryDirectory() as directory:
            root,checkpoint,config,model=self.make_fixture(directory)
            self.assertEqual(int(model.matched.sum()),3)
            torch.testing.assert_close(model.prior[model.matched].norm(dim=1),torch.ones(3))
            self.assertEqual(int(torch.count_nonzero(model.prior[~model.matched])),0)
            self.assertEqual(float(model.anchor_penalty().detach()),0.)
            with torch.no_grad():model.embedding.weight[4,0]+=.5
            self.assertAlmostEqual(float(model.anchor_penalty().detach()),.25/3,places=6)

    def test_shared_output_gradients_keep_pad_zero_with_adamw(self):
        model=semantic.SemanticGenerationGRU(12,embedding_dim=4,hidden_dim=6,dropout=0.)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
        before=model.embedding.weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits=model(torch.tensor([[2,4,5,0],[2,6,0,0]]))
            targets=torch.tensor([[4,5,3,0],[6,3,0,0]])
            torch.nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten(),ignore_index=0).backward()
            self.assertEqual(int(torch.count_nonzero(model.embedding.weight.grad[0])),0)
            self.assertGreater(int(torch.count_nonzero(model.embedding.weight.grad[9])),0)
            optimizer.step();self.assertEqual(int(torch.count_nonzero(model.embedding.weight[0])),0)
        self.assertFalse(torch.equal(before[9],model.embedding.weight[9]))

    def test_full_forward_and_incremental_step_agree(self):
        model=semantic.SemanticGenerationGRU(12,embedding_dim=4,hidden_dim=6,dropout=.4).eval()
        tokens=torch.tensor([[2,4,5,6,7],[2,8,7,4,6]])
        whole=model(tokens);hidden=None;parts=[]
        for j in range(tokens.shape[1]):
            logits,hidden=model.step(tokens[:,j:j+1],hidden);parts.append(logits)
        torch.testing.assert_close(whole,torch.cat(parts,dim=1),rtol=1e-5,atol=1e-7)

    def test_strict_load_preserves_rng_and_rejects_source_and_config_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root,checkpoint,config,model=self.make_fixture(directory)
            with patch.object(semantic,'__file__',str(root/'local_code/stage_4_code/Semantic_Generation.py')):
                before=torch.get_rng_state();loaded,vocab,meta=semantic.load_semantic_checkpoint(checkpoint)
                torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
                self.assertFalse(loaded.training);self.assertEqual(meta['epoch'],3)
                (root/'local_code/stage_4_code/Dataset_Loader.py').write_text('changed')
                with self.assertRaisesRegex(ValueError,'source changed'):semantic.load_semantic_checkpoint(checkpoint)
        with tempfile.TemporaryDirectory() as directory:
            root,checkpoint,config,model=self.make_fixture(directory)
            config['model']['dropout']=.6;(checkpoint.parent.parent/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'configuration'):semantic.load_semantic_checkpoint(checkpoint)

    def test_pretrained_manifest_and_vector_source_changes_are_rejected(self):
        for member in ('manifest.json','vectors.txt'):
            with self.subTest(member=member),tempfile.TemporaryDirectory() as directory:
                root,checkpoint,config,model=self.make_fixture(directory)
                with (root/'external'/member).open('a') as handle:handle.write('changed')
                with patch.object(semantic,'__file__',str(root/'local_code/stage_4_code/Semantic_Generation.py')):
                    with self.assertRaisesRegex(ValueError,'Pretrained'):semantic.load_semantic_checkpoint(checkpoint)

    def test_changed_pretrained_checkpoint_reference_and_vocab_are_rejected(self):
        for field in ('prior','vocab'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as directory:
                root,checkpoint,config,model=self.make_fixture(directory)
                saved=torch.load(checkpoint,weights_only=True)
                if field=='prior':saved['model']['prior'][4,0]+=.5
                else:saved['vocab']['tokens'][-1]='replacement'
                torch.save(saved,checkpoint)
                with patch.object(semantic,'__file__',str(root/'local_code/stage_4_code/Semantic_Generation.py')):
                    with self.assertRaises(ValueError):semantic.load_semantic_checkpoint(checkpoint)

    def test_word_prior_fails_on_mismatched_hash_and_malformed_known_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root,checkpoint,config,model=self.make_fixture(directory)
            vectors=root/'external/vectors.txt';manifest=root/'external/manifest.json'
            vectors.write_text('what 1 2\n')
            with self.assertRaisesRegex(ValueError,'manifest'):
                semantic.load_word_prior(Vocabulary(['what']),vectors,manifest,4)
            state=json.loads(manifest.read_text());state['member_sha256']=semantic.file_digest(vectors)
            manifest.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'Malformed'):
                semantic.load_word_prior(Vocabulary(['what']),vectors,manifest,4)


if __name__=='__main__':unittest.main()
