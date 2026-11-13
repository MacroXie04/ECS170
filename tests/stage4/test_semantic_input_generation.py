import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import test_semantic_generation as fixtures
import local_code.stage_4_code.Semantic_Input_Generation as semantic


class SemanticInputGenerationTests(unittest.TestCase):
    def make_fixture(self,directory):
        root,checkpoint,config,original=fixtures.SemanticGenerationTests().make_fixture(directory)
        config['architecture']='semantic_input_gru_v1'
        for name in ('local_code/stage_4_code/Semantic_Input_Generation.py','script/stage_4_script/train_semantic_input_generation.py'):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture '+name)
            config['source_sha256'][name]=semantic.file_digest(path)
        model=semantic.SemanticInputGenerationGRU(**config['model'])
        model.initialize_prior(original.prior,original.matched)
        (checkpoint.parent.parent/'config.json').write_text(json.dumps(config))
        torch.save(dict(model=model.state_dict(),vocab=torch.load(checkpoint,weights_only=True)['vocab'],epoch=3,
                        config_hash=semantic.canonical_digest(config)),checkpoint)
        return root,checkpoint,config,model

    def test_untied_output_does_not_update_unconsumed_word_embedding(self):
        model=semantic.SemanticInputGenerationGRU(12,embedding_dim=4,hidden_dim=6,dropout=0.)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
        logits=model(torch.tensor([[2,4,5],[2,6,7]]))
        targets=torch.tensor([[4,5,3],[6,7,3]])
        torch.nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten()).backward()
        self.assertEqual(int(torch.count_nonzero(model.embedding.weight.grad[9])),0)
        self.assertGreater(int(torch.count_nonzero(model.output.weight.grad[9])),0)
        self.assertEqual(int(torch.count_nonzero(model.embedding.weight.grad[0])),0)
        optimizer.step();self.assertEqual(int(torch.count_nonzero(model.embedding.weight[0])),0)
        self.assertNotEqual(model.embedding.weight.shape,model.output.weight.shape)

    def test_full_forward_matches_incremental_steps(self):
        model=semantic.SemanticInputGenerationGRU(12,embedding_dim=4,hidden_dim=6,dropout=.4).eval()
        tokens=torch.tensor([[2,4,5,6],[2,8,7,6]])
        hidden=None;parts=[]
        for j in range(tokens.shape[1]):
            logits,hidden=model.step(tokens[:,j:j+1],hidden);parts.append(logits)
        torch.testing.assert_close(model(tokens),torch.cat(parts,dim=1),rtol=1e-5,atol=1e-7)

    def test_loader_binds_new_sources_and_preserves_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root,checkpoint,config,model=self.make_fixture(directory)
            with patch.object(semantic,'__file__',str(root/'local_code/stage_4_code/Semantic_Input_Generation.py')):
                before=torch.get_rng_state();loaded,_,meta=semantic.load_semantic_input_checkpoint(checkpoint)
                torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
                self.assertEqual(loaded.cell,'semantic_input_gru');self.assertEqual(meta['epoch'],3)
                (root/'local_code/stage_4_code/Semantic_Input_Generation.py').write_text('changed')
                with self.assertRaisesRegex(ValueError,'source changed'):semantic.load_semantic_input_checkpoint(checkpoint)

    def test_loader_rejects_external_vector_drift_and_prior_corruption(self):
        for change in ('vector','prior'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                root,checkpoint,config,model=self.make_fixture(directory)
                if change=='vector':(root/'external/vectors.txt').write_text('changed')
                else:
                    saved=torch.load(checkpoint,weights_only=True);saved['model']['prior'][4,0]+=.2;torch.save(saved,checkpoint)
                with patch.object(semantic,'__file__',str(root/'local_code/stage_4_code/Semantic_Input_Generation.py')):
                    with self.assertRaisesRegex(ValueError,'[Pp]retrained'):semantic.load_semantic_input_checkpoint(checkpoint)


if __name__=='__main__':unittest.main()
