import copy
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from local_code.stage_4_code import Alternative_Generation as alternative
from local_code.stage_4_code.Dataset_Loader import Vocabulary
from local_code.stage_4_code.Generation_Optimization import validate_prefix
from local_code.stage_4_code.Method_RNN import SequenceModel
from local_code.stage_4_code.Novel_Generation import TrainingNoveltyIndex, lexical_tokens


def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


class AlternativeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.training=[dict(id='train/1',text='what did one fox bring to school? a bag of books.')]
        self.index=TrainingNoveltyIndex(self.training)

    def candidate(self,text,continuation,identifier=0,score=-1.):
        words=lexical_tokens(continuation)
        return dict(candidate_id=identifier,text=text,continuation=continuation,ids=[2,4,5,6,7+identifier,3],
            ended_with_eos=True,new_word_count=len(words),repeated_trigrams=0,
            average_log_probability=score,normalized_search_score=score,novelty=self.index.assess(text,continuation))

    def test_exact_full_and_changed_setup_copied_ending_are_rejected(self):
        exact=self.candidate('WHAT did one fox bring to school? A bag of books!',
                             'fox bring to school? A bag of books!')
        suffix=self.candidate('why did one school a bag of books', 'school a bag of books',1)
        # Five generated words pass the length rule; the copied suffix must
        # independently reject this changed three-word setup.
        self.assertIn('exact_full_copy',alternative.direct_copy_reasons(exact))
        self.assertTrue(suffix['novelty']['continuation']['exact_training_suffix'])
        self.assertIn('copied_continuation',alternative.direct_copy_reasons(suffix))
        self.assertNotIn('too_short',alternative.direct_copy_reasons(suffix))
        result=alternative.choose_direct([exact,suffix])
        self.assertIsNone(result['selected'])
        self.assertEqual(result['status'],'no_eligible_candidate')
        self.assertEqual(result['eligible_count'],0)
        self.assertEqual(len(result['rejections']),2)

    def test_question_requires_a_lexical_tail_but_guard_is_not_a_critic(self):
        question=self.candidate('what did one fox bring to the party?', 'fox bring to the party?')
        punctuation=copy.deepcopy(question);punctuation['text']+=' ...!!!'
        self.assertIn('question_without_answer',alternative.direct_copy_reasons(question))
        self.assertIn('question_without_answer',alternative.direct_copy_reasons(punctuation))
        answered=self.candidate('what did one fox bring to the party? fresh snacks.',
                                'fox bring to the party? fresh snacks.')
        self.assertNotIn('question_without_answer',alternative.direct_copy_reasons(answered))

    def test_eligibility_never_falls_back_to_an_invalid_high_likelihood_candidate(self):
        valid=self.candidate('what did one rabbit bring for dinner? fresh soup.',
                            'rabbit bring for dinner? fresh soup.',2,-3.)
        invalid=[]
        for index,mutation in enumerate(({'ended_with_eos':False},{'new_word_count':3},
            {'repeated_trigrams':1},{'average_log_probability':None},{'average_log_probability':float('nan')})):
            row=copy.deepcopy(valid);row['candidate_id']=index+10;row['average_log_probability']=-.01;row.update(mutation)
            invalid.append(row)
        self.assertEqual(alternative.choose_direct(invalid+[valid])['selected']['candidate_id'],2)
        self.assertIsNone(alternative.choose_direct(invalid)['selected'])
        self.assertIsNone(alternative.choose_direct([])['selected'])

    def test_soft_similarity_cost_and_stable_tie_break_do_not_mutate_candidates(self):
        copied=self.candidate('what did one rabbit bring for dinner? fresh soup.',
                             'rabbit bring for dinner? fresh soup.',4,-1.)
        novel=copy.deepcopy(copied);novel['candidate_id']=2;novel['average_log_probability']=-1.1
        copied['novelty']['continuation']['nearest_similarity']=.95
        novel['novelty']['continuation']['nearest_similarity']=.65
        before=copy.deepcopy([copied,novel])
        self.assertEqual(alternative.choose_direct([copied,novel])['selected']['candidate_id'],2)
        self.assertEqual([copied,novel],before)
        other=copy.deepcopy(novel);other['candidate_id']=1
        self.assertEqual(alternative.choose_direct([novel,other])['selected']['candidate_id'],1)

    def test_union_deduplicates_tokens_reassigns_ids_and_preserves_source_pools(self):
        a=self.candidate('what did one rabbit bring for dinner? fresh soup.',
                         'rabbit bring for dinner? fresh soup.',10,-1.)
        b=self.candidate('what did one bird bring for lunch? warm bread.',
                         'bird bring for lunch? warm bread.',20,-2.)
        c=self.candidate('what did one otter bring for breakfast? hot eggs.',
                         'otter bring for breakfast? hot eggs.',30,-3.)
        pools={0.0:[a,b],.75:[copy.deepcopy(b),c],1.5:[copy.deepcopy(a)]}
        frozen=copy.deepcopy(pools)
        settings=dict(copy_penalties=[0.0,.75,1.5],min_match_words=5,seed=1704,max_new_tokens=48)
        def search(*args,copy_penalty,**kwargs):return pools[copy_penalty]
        with patch.object(alternative,'search_candidates',side_effect=search),patch.object(alternative,'generate_candidates',return_value=[copy.deepcopy(a)]):
            result=alternative.generate_variants(object(),object(),'what did one',self.training,settings,True)
        self.assertEqual(pools,frozen)
        self.assertEqual([r['candidate_id'] for r in result['union']],[0,1,2])
        self.assertEqual([len(r['search_paths']) for r in result['union']],[2,2,1])
        self.assertEqual(result['variants']['beam12']['selected']['candidate_id'],10)
        self.assertEqual(result['variants']['soft075']['selected']['candidate_id'],20)
        self.assertEqual(result['variants']['direct_union']['selected']['candidate_id'],0)
        self.assertIn('old_beam5',result['variants'])
        result['union'][0]['novelty']['full_text']['exact_copy']=True
        self.assertEqual(pools,frozen)


class AlternativeBindingTests(unittest.TestCase):
    def fixture(self,root):
        run=root/'run';(run/'checkpoints').mkdir(parents=True)
        vocab=Vocabulary(['what','did','one','fox','bring','books','?','.'])
        kwargs=dict(vocab_size=len(vocab),task='generation',cell='gru',embedding_dim=4,hidden_dim=6,dropout=0.)
        with torch.random.fork_rng(devices=[]):model=SequenceModel(**kwargs)
        source=Path(__file__).resolve().parents[2]/'local_code/stage_4_code/Method_RNN.py'
        metadata=dict(source_sha256={'data.csv':'raw-bound'},vocabulary_sha256=hashlib.sha256(b''.join(t.encode()+b'\0' for t in vocab.tokens)).hexdigest(),
                      split_id_sha256={'train':'train-bound','validation':'val-bound','test':'test-bound'})
        config=dict(model=kwargs,data_metadata=metadata,source_sha256={'local_code/stage_4_code/Method_RNN.py':alternative.sha(source)})
        checkpoint=run/'checkpoints/best.pt'
        saved=dict(model=model.state_dict(),vocab=vocab.to_dict(),epoch=2,config_hash=json_hash(config))
        (run/'config.json').write_text(json.dumps(config))
        torch.save(saved,checkpoint)
        spec=dict(name='unit',components=[dict(checkpoint='run/checkpoints/best.pt',loader='original',checkpoint_sha256=alternative.sha(checkpoint))])
        return spec,metadata,config,saved,checkpoint

    def test_real_checkpoint_load_preserves_rng_and_binds_data(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(alternative,'ROOT',Path(temp)):
            spec,metadata,config,saved,checkpoint=self.fixture(Path(temp))
            state=torch.get_rng_state().clone()
            model,vocab,provenance=alternative.load_model(spec,'cpu',metadata)
            torch.testing.assert_close(state,torch.get_rng_state(),rtol=0,atol=0)
            self.assertFalse(model.training);self.assertEqual(provenance[0]['epoch'],2)
            self.assertEqual(len(validate_prefix('what did one',vocab)),4)
            for key in ('source_sha256','vocabulary_sha256','split_id_sha256'):
                changed=copy.deepcopy(metadata);changed[key]='changed'
                with self.assertRaisesRegex(ValueError,'Checkpoint data changed'):
                    alternative.load_model(spec,'cpu',changed)

    def test_checkpoint_config_and_training_source_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(alternative,'ROOT',Path(temp)):
            spec,metadata,config,saved,checkpoint=self.fixture(Path(temp))
            bad=copy.deepcopy(spec);bad['components'][0]['checkpoint_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'Checkpoint changed'):alternative.load_model(bad,metadata=metadata)
            config['changed_setting']=True
            (checkpoint.parent.parent/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'Mismatched configuration'):alternative.load_model(spec,metadata=metadata)
            config['source_sha256']['local_code/stage_4_code/Method_RNN.py']='0'*64
            saved['config_hash']=json_hash(config)
            (checkpoint.parent.parent/'config.json').write_text(json.dumps(config));torch.save(saved,checkpoint)
            spec['components'][0]['checkpoint_sha256']=alternative.sha(checkpoint)
            with self.assertRaisesRegex(ValueError,'Training source changed'):alternative.load_model(spec,metadata=metadata)

    def test_component_count_loader_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(alternative,'ROOT',Path(temp)):
            spec,metadata,*_=self.fixture(Path(temp))
            for components in ([],spec['components']*3):
                with self.assertRaisesRegex(ValueError,'one or two'):alternative.load_model(dict(components=components),metadata=metadata)
            bad=copy.deepcopy(spec);bad['components'][0]['loader']='unregistered'
            with self.assertRaisesRegex(ValueError,'Unknown generation architecture'):alternative.load_model(bad,metadata=metadata)
            bad=copy.deepcopy(spec);bad['components'][0]['checkpoint']='../outside.pt'
            with self.assertRaisesRegex(ValueError,'outside repository'):alternative.load_model(bad,metadata=metadata)
            (Path(temp)/'escape.pt').symlink_to(Path(temp).parent/'outside.pt')
            bad['components'][0]['checkpoint']='escape.pt'
            with self.assertRaisesRegex(ValueError,'outside repository'):alternative.load_model(bad,metadata=metadata)

    def test_comparator_rejects_source_and_row_drift_before_model_loading(self):
        compare=importlib.import_module('script.stage_4_script.compare_alternatives')
        for drift in ('none','train','validation','vocabulary','panel_source','runtime_source','panel_hash'):
            with self.subTest(drift=drift),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                data=dict(train=[dict(id='train/1',text='what did one fox bring to school')],
                          validation=[dict(id='val/1',text='what did one fox bring to dinner')],
                          metadata=dict(vocabulary_sha256='vocab-bound',source_sha256={'data.csv':'raw-bound'}))
                panel_source=root/'panel_source.py';panel_source.write_text('panel-source')
                runtime_source=root/'runtime_source.py';runtime_source.write_text('runtime-source')
                prompts=[dict(id='val/1',prompt='what did one',source_split='validation',prefix_seen_in_train=True)]
                manifest=dict(prompts=prompts,sha256=hashlib.sha256(json.dumps(prompts,sort_keys=True).encode()).hexdigest(),
                    train_rows_sha256=compare.training_digest(data['train']),validation_rows_sha256=compare.training_digest(data['validation']),
                    vocabulary_sha256='vocab-bound',source_sha256={'panel_source.py':alternative.sha(panel_source)})
                spec=dict(source_sha256={'runtime_source.py':alternative.sha(runtime_source)},search=dict(min_match_words=5),models=[dict(name='unreached')])
                if drift in ('train','validation'):data[drift][0]['text']+=' changed'
                elif drift=='vocabulary':data['metadata']['vocabulary_sha256']='different'
                elif drift=='panel_source':panel_source.write_text('changed')
                elif drift=='runtime_source':runtime_source.write_text('changed')
                elif drift=='panel_hash':manifest['sha256']='0'*64
                panel_path=root/'panel.json';panel_path.write_text(json.dumps(manifest))
                spec_path=root/'spec.json';spec_path.write_text(json.dumps(spec))
                argv=['compare_alternatives.py','--spec',str(spec_path),'--panel',str(panel_path),'--output',str(root/'output')]
                with patch.object(compare,'ROOT',root),patch.object(compare,'Dataset_Loader',return_value=SimpleNamespace(load_generation=lambda:data)),\
                     patch.object(compare,'load_model',side_effect=RuntimeError('Reached model loading')) as load,patch('sys.argv',argv):
                    if drift=='none':
                        with self.assertRaisesRegex(RuntimeError,'Reached model loading'):compare.main()
                        load.assert_called_once()
                    else:
                        with self.assertRaises(ValueError):compare.main()
                        load.assert_not_called()


if __name__=='__main__':unittest.main()
