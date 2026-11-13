import copy
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from local_code.stage_4_code.Dataset_Loader import Vocabulary

cli=importlib.import_module('script.stage_4_script.predict_alternative_generation')
finalizer=importlib.import_module('script.stage_4_script.finalize_alternatives')


class AlternativeProfileTests(unittest.TestCase):
    def fixture(self,root,mode='direct_union'):
        vocab=Vocabulary(['what','did','one','rabbit','bring','for','dinner','fresh','soup','?','.'])
        data=dict(train=[dict(id='train/1',text='what did one fox bring to school')],vocab=vocab,
                  metadata=dict(source_sha256={'data.csv':'raw'},vocabulary_sha256='vocabulary',split_id_sha256={'train':'split'}))
        sources={}
        for name in cli.REQUIRED_SOURCES:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture '+name)
            sources[name]=cli.sha(path)
        profile=dict(schema_version=1,candidate='unit__'+mode,mode=mode,source_sha256=sources,
            model_spec=dict(name='unit',components=[dict(loader='semantic_input',checkpoint='model.pt',checkpoint_sha256='saved')]),
            search=dict(seed=1704,max_new_tokens=48,copy_penalties=[0.0,.75,1.5],min_match_words=5),
            data_binding=dict(data['metadata'],train_rows_sha256=cli.training_digest(data['train'])))
        path=root/'profile.json';path.write_text(json.dumps(profile))
        return profile,data,path

    def test_requires_all_sources_and_rejects_modified_source_bytes(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(cli,'ROOT',Path(temp)):
            profile,data,_=self.fixture(Path(temp))
            cli.verify_profile(profile,data)
            for source in cli.REQUIRED_SOURCES:
                bad=copy.deepcopy(profile);del bad['source_sha256'][source]
                with self.assertRaisesRegex(ValueError,'Missing required source'):cli.verify_profile(bad,data)
            chosen=next(iter(cli.REQUIRED_SOURCES));(Path(temp)/chosen).write_text('changed')
            with self.assertRaisesRegex(ValueError,'Profile source changed'):cli.verify_profile(profile,data)

    def test_rejects_source_path_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'repo';root.mkdir()
            outside=Path(temp)/'outside.py';outside.write_text('outside')
            with patch.object(cli,'ROOT',root):
                profile,data,_=self.fixture(root)
                bad=copy.deepcopy(profile);bad['source_sha256']['../outside.py']=cli.sha(outside)
                with self.assertRaisesRegex(ValueError,'Profile source changed'):cli.verify_profile(bad,data)
                (root/'link.py').symlink_to(outside)
                bad=copy.deepcopy(profile);bad['source_sha256']['link.py']=cli.sha(outside)
                with self.assertRaisesRegex(ValueError,'Profile source changed'):cli.verify_profile(bad,data)

    def test_data_and_exact_training_reference_are_bound(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(cli,'ROOT',Path(temp)):
            profile,data,_=self.fixture(Path(temp))
            for key in ('source_sha256','vocabulary_sha256','split_id_sha256'):
                changed=copy.deepcopy(data);changed['metadata'][key]='different'
                with self.assertRaisesRegex(ValueError,'Profile data changed'):cli.verify_profile(profile,changed)
            changed=copy.deepcopy(data);changed['train'][0]['text']+=' different'
            with self.assertRaisesRegex(ValueError,'Training-copy reference changed'):cli.verify_profile(profile,changed)
            for mode in ('retrieval','fallback'):
                bad=copy.deepcopy(profile);bad['mode']=mode
                with self.assertRaisesRegex(ValueError,'Unsupported alternative profile'):cli.verify_profile(bad,data)

    def test_no_eligible_result_is_explicit_and_current_metadata_reaches_loader(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(cli,'ROOT',Path(temp)):
            profile,data,path=self.fixture(Path(temp))
            before=path.read_bytes()
            selection=dict(selected=None,status='no_eligible_candidate',eligible_count=0,
                rejections=[dict(candidate_id=0,reasons=['copied_continuation'])])
            values=dict(variants={'direct_union':selection})
            with patch.object(cli,'Dataset_Loader',return_value=SimpleNamespace(load_generation=lambda:data)),\
                 patch.object(cli,'load_model',return_value=(object(),data['vocab'],[dict(epoch=2)])) as load,\
                 patch.object(cli,'generate_variants',return_value=values),patch.object(cli,'generate_candidates') as old:
                result=cli.predict('what did one',path,device='cpu')
            load.assert_called_once_with(profile['model_spec'],'cpu',data['metadata'])
            old.assert_not_called()
            self.assertIsNone(result['text']);self.assertIsNone(result['result'])
            self.assertEqual(result['status'],'no_eligible_candidate')
            self.assertTrue(result['external_word_vectors'])
            self.assertIn('selected',selection)
            self.assertEqual(before,path.read_bytes())

    def test_vocabulary_and_checkpoint_loader_errors_fail_before_generation(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(cli,'ROOT',Path(temp)):
            profile,data,path=self.fixture(Path(temp))
            with patch.object(cli,'Dataset_Loader',return_value=SimpleNamespace(load_generation=lambda:data)),\
                 patch.object(cli,'load_model',side_effect=ValueError('Checkpoint changed')),patch.object(cli,'generate_variants') as generate:
                with self.assertRaisesRegex(ValueError,'Checkpoint changed'):cli.predict('what did one',path,'cpu')
                generate.assert_not_called()
            other=Vocabulary(['what','did','one','replacement'])
            with patch.object(cli,'Dataset_Loader',return_value=SimpleNamespace(load_generation=lambda:data)),\
                 patch.object(cli,'load_model',return_value=(object(),other,[])),patch.object(cli,'generate_variants') as generate:
                with self.assertRaisesRegex(ValueError,'Vocabulary differs'):cli.predict('what did one',path,'cpu')
                generate.assert_not_called()

    def test_old_beam_is_an_explicit_mode_not_a_guarded_fallback(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(cli,'ROOT',Path(temp)):
            profile,data,path=self.fixture(Path(temp),'old_beam5')
            profile['model_spec']['components'][0]['loader']='original';path.write_text(json.dumps(profile))
            candidate=dict(text='what did one rabbit bring for dinner? fresh soup.',ended_with_eos=True)
            with patch.object(cli,'Dataset_Loader',return_value=SimpleNamespace(load_generation=lambda:data)),\
                 patch.object(cli,'load_model',return_value=(object(),data['vocab'],[])),\
                 patch.object(cli,'generate_candidates',return_value=[candidate]) as old,patch.object(cli,'generate_variants') as guarded:
                result=cli.predict('what did one',path,'cpu')
            guarded.assert_not_called();old.assert_called_once()
            self.assertEqual(result['text'],candidate['text'])
            self.assertFalse(result['external_word_vectors'])

    def test_promotion_requires_copy_and_coherence_conditions_independently(self):
        baseline=dict(direct_copy_free_coherent_count=1,coherent_count=8,direct_copy_count=10,strict_novel_coherent_count=0)
        good=dict(direct_copy_free_coherent_count=2,coherent_count=6,direct_copy_count=9,strict_novel_coherent_count=0)
        self.assertTrue(finalizer.qualifies(good,baseline))
        for change in ({'direct_copy_free_coherent_count':1},{'coherent_count':5},{'direct_copy_count':10}):
            self.assertFalse(finalizer.qualifies(dict(good,**change),baseline))
        stricter=dict(baseline,strict_novel_coherent_count=1)
        self.assertFalse(finalizer.qualifies(good,stricter))

    def test_finalizer_requires_every_fixed_prompt_even_when_all_abstain(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            model_spec=dict(name='unit')
            rows=[dict(candidate='unit__direct_union',model_spec=model_spec,mode='direct_union',prompt_index=i,
                       prompt='what did one',seen_prefix=False,blind_id='A001',text='',result=None) for i in range(16)]
            def save(current):
                mapping=[dict(id=r['blind_id'],candidate=r['candidate'],prompt_index=r['prompt_index']) for r in current]
                files={'outputs.json':current,'blind_mapping.json':mapping,
                    'blind_review.json':[dict(id='A001',text='',no_output=True)],
                    'coherence_ratings.json':[dict(id='A001',score=0,reason='No output')],
                    'metrics.json':[dict(model='unit',loss=dict(loss=4.,count=3265))],
                    'candidate_pools.json':[],'provenance.json':{}}
                for name,value in files.items():(directory/name).write_text(json.dumps(value))
            save(rows)
            summary,_=finalizer.summarize(directory)
            self.assertEqual(summary[0]['count'],16)
            self.assertEqual(summary[0]['output_count'],0)
            save(rows[:-1])
            with self.assertRaises(ValueError):finalizer.summarize(directory)
            changed=copy.deepcopy(rows);changed[-1]['prompt_index']=16;save(changed)
            with self.assertRaises(ValueError):finalizer.summarize(directory)


if __name__=='__main__':unittest.main()
