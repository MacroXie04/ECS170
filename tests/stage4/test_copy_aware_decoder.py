import copy
import math
import unittest

import torch
from torch import nn

from local_code.stage_4_code.Copy_Aware_Decoder import CopySpanIndex, search_candidates
from local_code.stage_4_code.Dataset_Loader import Vocabulary
from local_code.stage_4_code.Generation_Ensemble import MixtureModel
from local_code.stage_4_code.Method_RNN import SequenceModel


class TableModel(nn.Module):
    task = 'generation'

    def __init__(self,vocab,table=None,default=None):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocab_size = len(vocab)
        self.table = table or {}
        self.default = default or {3:1.}

    def step(self,tokens,hidden=None):
        # Mutating this state deliberately catches branch-aliasing bugs.
        history = [] if hidden is None else hidden[0].tolist()
        if hidden is not None and int(hidden[1].item()) != len(history):
            raise AssertionError('A sibling beam mutated this branch state')
        logits = []
        for token in tokens[0].tolist():
            history.append(token)
            probabilities = self.table.get(tuple(history),self.default)
            scores = torch.full((self.vocab_size,),-torch.inf)
            for value,probability in probabilities.items(): scores[value] = math.log(probability)
            logits.append(scores)
        if hidden is not None: hidden[1].add_(1)
        return torch.stack(logits)[None,:,:], (torch.tensor(history),torch.tensor([len(history)]))


class CopyAwareTests(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary(['what','did','one','fox','bring','to','school','books','snacks','?','.'])
        self.prefix = self.vocab.encode('what did one',bos=True)
        self.records = [{'id':'train/1','text':'what did one fox bring to school books.'}]
        self.p = tuple(self.prefix)

    def tok(self,word): return self.vocab.stoi[word]

    def branching_model(self):
        table = {}; prefix = self.p
        for word in ('fox','bring','to','school'):
            token = self.tok(word); table[prefix] = {token:1.}; prefix += (token,)
        table[prefix] = {self.tok('books'):.8,self.tok('snacks'):.2}
        for word in ('books','snacks'): table[prefix+(self.tok(word),)] = {3:1.}
        return TableModel(self.vocab,table)

    def test_soft_penalty_changes_branch_but_retains_original_likelihood(self):
        model = self.branching_model()
        kwargs = dict(beam_width=2,num_return_sequences=2,max_new_tokens=6,min_match_words=5)
        ordinary = search_candidates(model,self.vocab,'what did one',self.records,copy_penalty=0,**kwargs)
        aware = search_candidates(model,self.vocab,'what did one',self.records,copy_penalty=3,**kwargs)
        self.assertIn('books',ordinary[0]['text'])
        self.assertIn('snacks',aware[0]['text'])
        self.assertEqual(len(aware),2)
        by_word = {r['continuation'].split()[-1]:r for r in aware}
        self.assertAlmostEqual(by_word['books']['log_probability'],math.log(.8),places=6)
        self.assertAlmostEqual(by_word['snacks']['log_probability'],math.log(.2),places=6)
        self.assertEqual(by_word['books']['copy_penalty_cost'],3.)
        self.assertEqual(by_word['snacks']['copy_penalty_cost'],0.)
        event = by_word['books']['copy_events'][0]
        self.assertEqual(event['lexical_ngram'],['fox','bring','to','school','books'])
        self.assertEqual(event['generated_word_index'],4)
        for row in aware:
            self.assertAlmostEqual(row['search_score'],row['search_lm_log_probability']-row['copy_penalty_cost'])
            self.assertTrue(row['ended_with_eos'])
        # Copies are penalized, not silently dropped; all returned sequences are model-generated.
        self.assertTrue(by_word['books']['novelty']['full_text']['exact_copy'])

    def test_index_ignores_case_punctuation_and_exempts_common_boilerplate(self):
        index = CopySpanIndex([{'id':'a','text':'WHAT did one fox, bring to school books!'}],5)
        matches = index.next_words(('fox','bring','to','school'))
        self.assertEqual(matches['books']['document_frequency'],1)
        repeated = [dict(id=str(i),text='what did one fox bring to school books') for i in range(3)]
        common = CopySpanIndex(repeated,5)
        self.assertEqual(common.next_words(('fox','bring','to','school')),{})
        self.assertGreater(common.ignored_boilerplate_count,0)
        self.assertEqual(CopySpanIndex(repeated[:2],5).next_words(('fox','bring','to','school'))['books']['raw_cost'],1/math.sqrt(2))

    def test_prompt_not_counted_as_generated_copy_and_prebuilt_index(self):
        row = search_candidates(self.branching_model(),self.vocab,'what did one',CopySpanIndex(self.records,5),
            beam_width=1,num_return_sequences=1,max_new_tokens=6,copy_penalty=0)[0]
        self.assertEqual(len(row['copy_events']),1)
        self.assertEqual(row['new_word_count'],5)
        self.assertEqual(row['copy_events'][0]['lexical_ngram'][0],'fox')
        with self.assertRaises(ValueError):
            search_candidates(self.branching_model(),self.vocab,'what did one',CopySpanIndex(self.records,4))

    def test_original_probability_keeps_blocked_special_token_mass(self):
        sequence = ('fox','bring','to','school')
        table = {}; prefix = self.p
        for word in sequence:
            token=self.tok(word);table[prefix]={0:.1,1:.2,2:.3,token:.4};prefix+=(token,)
        table[prefix] = {3:1.}
        row=search_candidates(TableModel(self.vocab,table),self.vocab,'what did one',self.records,
            beam_width=1,num_return_sequences=1,max_new_tokens=5)[0]
        self.assertAlmostEqual(row['log_probability'],4*math.log(.4),places=6)
        self.assertFalse(set(row['ids'][len(self.prefix):]) & {0,1,2})

    def test_large_copy_penalty_cannot_choose_implausible_escape(self):
        model=self.branching_model()
        ending=self.p+tuple(self.tok(word) for word in ('fox','bring','to','school'))
        model.table[ending]={self.tok('books'):.999,self.tok('snacks'):.001}
        rows=search_candidates(model,self.vocab,'what did one',self.records,
            beam_width=2,num_return_sequences=2,max_new_tokens=6,copy_penalty=100)
        self.assertEqual(len(rows),1)
        self.assertIn('books',rows[0]['text'])
        self.assertEqual(rows[0]['copy_penalty_cost'],100.)
        # Top-k is also applied before copying costs, independently of the floor.
        model=self.branching_model()
        top_one=search_candidates(model,self.vocab,'what did one',self.records,
            beam_width=2,num_return_sequences=2,max_new_tokens=6,copy_penalty=100,top_k=1)
        self.assertEqual(len(top_one),1)
        self.assertIn('books',top_one[0]['text'])

    def test_completed_punctuation_variants_do_not_fill_candidate_pool(self):
        model=self.branching_model()
        ending=self.p+tuple(self.tok(word) for word in ('fox','bring','to','school'))
        model.table[ending]={3:.4,self.tok('.'):.3,self.tok('books'):.3}
        rows=search_candidates(model,self.vocab,'what did one',self.records,
            beam_width=3,num_return_sequences=3,max_new_tokens=6,copy_penalty=0)
        self.assertEqual(len(rows),2)
        self.assertEqual(sorted(row['equivalent_token_sequence_count'] for row in rows),[1,2])

    def test_deterministic_preserves_rng_all_modes_and_inputs(self):
        model=self.branching_model().train()
        model.extra=nn.Dropout(.4).eval()
        before_records=copy.deepcopy(self.records);state=torch.get_rng_state().clone()
        first=search_candidates(model,self.vocab,'what did one',self.records,beam_width=2,num_return_sequences=2,max_new_tokens=6)
        second=search_candidates(model,self.vocab,'what did one',self.records,beam_width=2,num_return_sequences=2,max_new_tokens=6)
        self.assertEqual(first,second)
        self.assertEqual(self.records,before_records)
        torch.testing.assert_close(state,torch.get_rng_state(),rtol=0,atol=0)
        self.assertTrue(model.training);self.assertFalse(model.extra.training)

    def test_incomplete_budget_does_not_fabricate_eos(self):
        row=search_candidates(self.branching_model(),self.vocab,'what did one',self.records,
            beam_width=1,num_return_sequences=1,max_new_tokens=2)[0]
        self.assertEqual(row['generated_token_count'],2)
        self.assertFalse(row['ended_with_eos']);self.assertEqual(row['finish_reason'],'max_new_tokens')
        empty=search_candidates(self.branching_model(),self.vocab,'what did one',self.records,
            max_new_tokens=0)[0]
        self.assertEqual(empty['ids'],self.prefix)
        self.assertIsNone(empty['average_log_probability'])

    def test_exhausted_repeated_trigrams_are_explicit(self):
        model=TableModel(self.vocab,default={self.tok('one'):1.})
        row=search_candidates(model,self.vocab,'what did one',self.records,
            beam_width=1,num_return_sequences=1,max_new_tokens=12)[0]
        self.assertFalse(row['ended_with_eos'])
        self.assertEqual(row['finish_reason'],'constraints_exhausted')
        self.assertEqual(row['repeated_trigrams'],0)

    def test_real_lstm_and_nested_mixture_hidden_states(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(43)
            models=[SequenceModel(len(self.vocab),task='generation',cell=cell,
                embedding_dim=4,hidden_dim=6,dropout=.2) for cell in ('lstm','gru')]
            mixture=MixtureModel(models,late_weight=.25).train()
        rows=search_candidates(mixture,self.vocab,'what did one',self.records,
            beam_width=3,num_return_sequences=2,max_new_tokens=7)
        self.assertTrue(rows)
        self.assertTrue(mixture.training)
        for row in rows:
            self.assertTrue(math.isfinite(row['log_probability']))
            self.assertTrue(math.isfinite(row['normalized_search_score']))
            self.assertEqual(row['ids'][:len(self.prefix)],self.prefix)

    def test_rejects_invalid_arguments_and_restores_mode_on_error(self):
        model=self.branching_model()
        for kwargs in ({'copy_penalty':-1},{'copy_penalty':float('nan')},{'min_match_words':2},
                       {'min_match_words':True},{'beam_width':0},{'num_return_sequences':13},
                       {'max_new_tokens':-1},{'seed':True},{'copy_penalty':'two'},
                       {'top_k':0},{'min_relative_probability':0},{'min_relative_probability':float('nan')}):
            with self.assertRaises(ValueError): search_candidates(model,self.vocab,'what did one',self.records,**kwargs)
        with self.assertRaises(ValueError):
            search_candidates(model,self.vocab,'what did one',[dict(id='v',text='what did one fox',split='validation')])
        class Broken(TableModel):
            def step(self,tokens,hidden=None): raise RuntimeError('deliberate')
        broken=Broken(self.vocab).train()
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            search_candidates(broken,self.vocab,'what did one',self.records)
        self.assertTrue(broken.training)


if __name__=='__main__': unittest.main()
