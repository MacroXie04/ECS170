import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from local_code.stage_4_code.Method_RNN import SequenceModel
from local_code.stage_4_code.Dataset_Loader import Vocabulary
from local_code.stage_4_code.Generation_Optimization import load_generation_checkpoint,validate_prefix
from script.stage_4_script.predict_generation import predict


class OptimizationTests(unittest.TestCase):
    def make_run(self,directory):
        run=Path(directory)/'run'; (run/'checkpoints').mkdir(parents=True)
        vocab=Vocabulary(['what','do','you','call','a','cat','?','meow','.'])
        kwargs=dict(vocab_size=len(vocab),task='generation',cell='gru',embedding_dim=4,hidden_dim=6,dropout=0.)
        model=SequenceModel(**kwargs)
        with torch.no_grad():
            model.output.weight.zero_();model.output.bias.zero_();model.output.bias[3]=10.
        config={'model':kwargs,'data_metadata':{'vocabulary_sha256':hashlib.sha256(b''.join(token.encode()+b'\0' for token in vocab.tokens)).hexdigest()}}
        (run/'config.json').write_text(json.dumps(config))
        digest=hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        (run.parent/'generation_vocabulary.json').write_text(json.dumps(vocab.to_dict()))
        torch.save({'model':model.state_dict(),'model_kwargs':kwargs,'vocab':vocab.to_dict(),
                    'config_hash':digest,'epoch':3},run/'checkpoints/best.pt')
        torch.save({'model':model.state_dict(),'config_hash':digest,'history':[{},{}]},run/'checkpoints/last.pt')
        return run,vocab

    def test_best_and_last_load_with_bound_vocabulary(self):
        with tempfile.TemporaryDirectory() as temp:
            run,vocab=self.make_run(temp)
            before=torch.get_rng_state()
            model,actual,meta=load_generation_checkpoint(run/'checkpoints/best.pt')
            torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
            _,late,late_meta=load_generation_checkpoint(run/'checkpoints/last.pt')
            self.assertEqual(actual.to_dict(),vocab.to_dict())
            self.assertEqual(actual.to_dict(),late.to_dict())
            self.assertEqual(meta['epoch'],3);self.assertEqual(late_meta['epoch'],2)

    def test_reject_modified_config_and_bad_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            run,vocab=self.make_run(temp)
            self.assertEqual(len(validate_prefix('What do you?',vocab)),5)
            for prompt in ('what do','what do you call','what do elephants'):
                with self.assertRaises(ValueError):validate_prefix(prompt,vocab)
            config=json.loads((run/'config.json').read_text());config['new_setting']=True
            (run/'config.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):load_generation_checkpoint(run/'checkpoints/best.pt')

    def test_profile_binding_and_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            run,_=self.make_run(temp);checkpoint=run/'checkpoints/best.pt'
            profile={'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     'candidate':'unit_fixture','decoding':{'strategy':'beam','min_new_tokens':0,'max_new_tokens':4}}
            profile_path=Path(temp)/'profile.json';profile_path.write_text(json.dumps(profile))
            result=predict('What do you',profile=profile_path,device='cpu')
            self.assertEqual(result['text'],'What do you');self.assertTrue(result['ended_with_eos'])
            self.assertEqual(result['profile'],'unit_fixture')
            profile['checkpoint_sha256']='0'*64;profile_path.write_text(json.dumps(profile))
            with self.assertRaises(ValueError):predict('What do you',profile=profile_path,device='cpu')

    def test_late_checkpoint_rejects_changed_external_vocabulary(self):
        with tempfile.TemporaryDirectory() as temp:
            run,vocab=self.make_run(temp)
            saved=vocab.to_dict();saved['tokens'][-1]='replacement'
            (run.parent/'generation_vocabulary.json').write_text(json.dumps(saved))
            with self.assertRaises(ValueError):load_generation_checkpoint(run/'checkpoints/last.pt')

    def test_bound_training_source_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run,_=self.make_run(temp)
            config=json.loads((run/'config.json').read_text())
            config['source_sha256']={'local_code/stage_4_code/Method_RNN.py':'0'*64}
            (run/'config.json').write_text(json.dumps(config))
            checkpoint=run/'checkpoints/best.pt'
            saved=torch.load(checkpoint,weights_only=True)
            saved['config_hash']=hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
            torch.save(saved,checkpoint)
            with self.assertRaisesRegex(ValueError,'Training source changed'):
                load_generation_checkpoint(checkpoint)


if __name__=='__main__':unittest.main()
