#!/usr/bin/env python3
"""Audit copy-aware generation evidence without dumping or rating its texts."""
from __future__ import annotations
import argparse
import copy
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from torch.utils.data import DataLoader
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Experiment_Runner import collate,run_epoch,write_json
from local_code.stage_4_code.Generation_Optimization import validate_prefix,format_completion
from local_code.stage_4_code.Generation_Decoder import _scores
from local_code.stage_4_code.Novel_Generation import TrainingNoveltyIndex,lexical_tokens,select_candidate
from local_code.stage_4_code.Alternative_Generation import load_model
from verify_generalization import (
    require,digest,repo_path,verify_sources,verify_panel,rows_digest,read_json,
    repeated_trigrams,verify_pool,verify_likelihoods,
)

BASE=ROOT/'result/stage_4_result/generation_alternatives'
FILES=('provenance.json','metrics.json','outputs.json','candidate_pools.json','blind_review.json','blind_mapping.json')
REQUIRED_SOURCES={'local_code/stage_4_code/Alternative_Generation.py','local_code/stage_4_code/Copy_Aware_Decoder.py',
    'local_code/stage_4_code/Novel_Generation.py','local_code/stage_4_code/Generation_Decoder.py',
    'local_code/stage_4_code/Generation_Optimization.py','local_code/stage_4_code/Dataset_Loader.py',
    'local_code/stage_4_code/Experiment_Runner.py','script/stage_4_script/compare_alternatives.py',
    'script/stage_4_script/compare_generation.py','script/stage_4_script/predict_novel_generation.py'}


def close(a,b,label,tolerance=1e-9):
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tolerance,label)


def build_ngrams(index,width):
    documents=defaultdict(list)
    for identifier,words,_ in index.rows:
        for gram in set(tuple(words[j:j+width]) for j in range(len(words)-width+1)):
            documents[gram].append(identifier)
    threshold=max(3,math.ceil(.02*len(index.rows)))
    ignored=sum(len(ids)>=threshold for ids in documents.values())
    return {gram:sorted(ids) for gram,ids in documents.items() if len(ids)<threshold},threshold,ignored


def check_candidate(candidate,prompt,vocab,index,cache,settings,penalty,ngrams,threshold,ignored):
    prefix=validate_prefix(prompt,vocab);ids=candidate['ids'];new=ids[len(prefix):]
    require(candidate['prompt']==prompt and ids[:len(prefix)]==prefix,'Candidate prefix mismatch')
    require(all(type(i) is int and 0<=i<len(vocab) for i in ids),'Invalid candidate token ID')
    require(len(new)<=settings['max_new_tokens'] and not any(i<3 for i in new),'Candidate token budget/special-token violation')
    require(3 not in new[:-1],'Internal EOS')
    ended=bool(new and new[-1]==3)
    require(candidate['ended_with_eos']==ended and (candidate['finish_reason']=='eos')==ended,'EOS metadata mismatch')
    require(candidate['finish_reason'] in ('eos','max_new_tokens','constraints_exhausted'),'Unknown finish reason')
    require(candidate['generated_token_count']==len(new),'Generated-token count mismatch')
    continuation=vocab.decode(new);text=format_completion(prompt,ids,len(prefix),vocab)
    require(candidate['continuation']==continuation and candidate['text']==text,'Candidate text/ID mismatch')
    words=lexical_tokens(continuation)
    require(candidate['new_word_count']==len(words),'Word count mismatch')
    require(candidate['repeated_trigrams']==repeated_trigrams(lexical_tokens(text)),'Lexical repetition mismatch')
    require(candidate['continuation_repeated_trigrams']==repeated_trigrams(words),'Continuation repetition mismatch')
    expected_average=candidate['log_probability']/len(new) if new else None
    require(candidate['average_log_probability']==expected_average,'Average likelihood mismatch')
    key=(text,continuation)
    if key not in cache:cache[key]=index.assess(text,continuation)
    require(candidate['novelty']==cache[key],'TRAIN novelty mismatch')
    require(len(candidate['sources'])==1,'Search candidate has unexpected source count')
    source=candidate['sources'][0];decode=source['decoding']
    require(source['strategy']=='copy_aware_beam' and source['seed']==settings['seed'],'Search source mismatch')
    required_settings={key:value for key,value in settings.items() if key!='copy_penalties'}
    required_settings.update(copy_penalty=penalty,strategy='copy_aware_beam',
        min_new_tokens=min(4,settings['max_new_tokens']),length_penalty=.7,repetition_penalty=1.1,no_repeat_ngram_size=3,
        boilerplate_document_frequency=threshold,boilerplate_ngram_count=ignored)
    require(all(decode.get(key)==value for key,value in required_settings.items()),'Search settings mismatch')
    require(type(candidate['equivalent_token_sequence_count']) is int and candidate['equivalent_token_sequence_count']>=1,'Invalid lexical equivalence count')
    generated_words=[];events=[];raw=0.;width=settings['min_match_words']
    for position,token in enumerate(new):
        token_words=lexical_tokens(vocab.tokens[token]) if token>=4 else ()
        require(len(token_words)<=1,'Vocabulary is not word-tokenized')
        if token_words:
            gram=tuple((generated_words+list(token_words))[-width:])
            if len(gram)==width and gram in ngrams:
                identifiers=ngrams[gram];cost=1/math.sqrt(len(identifiers));raw+=cost
                events.append(dict(lexical_ngram=list(gram),document_frequency=len(identifiers),training_ids=identifiers,
                    raw_cost=cost,generated_token_index=position,generated_word_index=len(generated_words),weighted_cost=penalty*cost))
            generated_words.extend(token_words)
    require(candidate['copy_events']==events,'Copy penalty events do not match TRAIN n-grams')
    close(candidate['raw_copy_cost'],raw,'Raw copy cost mismatch')
    close(candidate['copy_penalty_cost'],penalty*raw,'Weighted copy cost mismatch')
    score=candidate['search_lm_log_probability']-penalty*raw
    close(candidate['search_score'],score,'Search objective mismatch')
    normalized=score/(((5+max(1,len(new)))/6)**.7)
    close(candidate['normalized_search_score'],normalized,'Normalized search objective mismatch')
    return len(prefix)


@torch.no_grad()
def replay_search_paths(model,candidates,prefix_size):
    maximum=0.
    for start in range(0,len(candidates),32):
        batch=candidates[start:start+32]
        sequences=[torch.tensor(row['ids'],dtype=torch.long) for row in batch]
        tokens=torch.nn.utils.rnn.pad_sequence(sequences,batch_first=True)
        logits=model(tokens[:,:-1])
        for index,candidate in enumerate(batch):
            ids=candidate['ids'];settings=candidate['sources'][0]['decoding'];total=0.
            for emitted,token in enumerate(ids[prefix_size:]):
                original,scores=_scores(logits[index,prefix_size+emitted-1],ids[:prefix_size+emitted],
                    emitted,settings['min_new_tokens'],3,1.1,1.)
                require(torch.isfinite(scores[token]).item(),'Recorded path violates repetition or EOS constraints')
                plausible=original.clone();plausible[:3]=-torch.inf
                if emitted<settings['min_new_tokens']:plausible[3]=-torch.inf
                ranked=torch.sort(plausible,descending=True,stable=True).values
                kth=ranked[min(settings['top_k'],len(ranked))-1]
                epsilon=2e-4
                require(original[token]>=kth-epsilon,'Recorded path violates original-model top-k bound')
                require(original[token]>=plausible.max()+math.log(settings['min_relative_probability'])-epsilon,
                    'Recorded path violates original-model relative-probability bound')
                total+=float(scores[token])
            error=abs(total-candidate['search_lm_log_probability'])
            require(error<=2e-4*max(1,len(ids)-prefix_size),'CPU search likelihood replay mismatch')
            maximum=max(maximum,error)
    return maximum


def direct_selection(pool):
    eligible=[];rejected=[]
    for candidate in pool:
        reasons=[]
        if not candidate['ended_with_eos']:reasons.append('incomplete')
        if candidate['new_word_count']<5:reasons.append('too_short')
        if candidate['repeated_trigrams']:reasons.append('repeated_trigrams')
        full,tail=candidate['novelty']['full_text'],candidate['novelty']['continuation']
        if full['exact_copy']:reasons.append('exact_full_copy')
        if tail['exact_copy'] or tail['exact_training_span']:reasons.append('copied_continuation')
        if '?' in candidate['text'] and not lexical_tokens(candidate['text'].split('?',1)[1]):reasons.append('question_without_answer')
        score=candidate['average_log_probability']
        if score is None or not math.isfinite(score):reasons.append('invalid_likelihood')
        if reasons:rejected.append(dict(candidate_id=candidate['candidate_id'],reasons=reasons))
        else:eligible.append((score-.5*max(0.,tail['nearest_similarity']-.65),candidate))
    eligible.sort(key=lambda item:(-item[0],item[1]['candidate_id']))
    return dict(selected=eligible[0][1] if eligible else None,status='selected' if eligible else 'no_eligible_candidate',
        selection_score=eligible[0][0] if eligible else None,eligible_count=len(eligible),rejections=rejected,
        method='mean_original_likelihood_minus_soft_similarity_cost')


def verify_blind(directory,rows):
    blind=read_json(directory/'blind_review.json');mapping=read_json(directory/'blind_mapping.json')
    unique={row['text']:row['prompt'] for row in rows};texts=list(unique);random.Random(170451).shuffle(texts)
    identifiers={text:f'A{j:03d}' for j,text in enumerate(texts,1)}
    expected=[dict(id=identifiers[text],prompt=unique[text],text=text,no_output=not bool(text)) for text in texts]
    require(blind==expected,'Deduplicated blind-review content/order mismatch')
    require(len(mapping)==len(rows),'Blind mapping count mismatch')
    for row,item in zip(rows,mapping):
        require(row['blind_id']==identifiers[row['text']],'Output blind ID mismatch')
        require(item==dict(id=row['blind_id'],candidate=row['candidate'],prompt_index=row['prompt_index']),'Many-to-one blind mapping mismatch')
    reviews={}
    for name in ('coherence_ratings.json','confirmatory_ratings.json'):
        path=directory/name
        if not path.exists():continue
        ratings=read_json(path)
        require(len(ratings)==len(blind) and {r['id'] for r in ratings}=={b['id'] for b in blind},'Rating ID coverage/uniqueness mismatch')
        scores={r['id']:r['score'] for r in ratings}
        require(all(type(score) is int and score in (0,1,2) for score in scores.values()),'Invalid rating score')
        require(all(scores[row['id']]==0 for row in blind if row['no_output']),'Missing output has nonzero rating')
        reviews[name]=dict(sha256=digest(path),unique_ratings=len(ratings),mapped_outputs=len(rows))
    return dict(unique_texts=len(blind),mapped_outputs=len(rows),reviews=reviews)


def audit(directory,spec_path,panel_path,data):
    missing=[name for name in FILES if not (directory/name).exists()]
    if missing:return dict(status='skipped_incomplete',directory=str(directory.relative_to(ROOT)),missing=missing)
    spec=read_json(spec_path);manifest=read_json(panel_path);provenance=read_json(directory/'provenance.json')
    require(digest(spec_path)==provenance['spec_sha256'] and digest(panel_path)==provenance['panel_sha256'],'Frozen spec/panel file changed')
    require(provenance['source_sha256']==spec['source_sha256'],'Comparison/source specification mismatch')
    verify_sources(spec['source_sha256'],REQUIRED_SOURCES)
    require(provenance['data_source_sha256']==data['metadata']['source_sha256'],'Raw data hash mismatch')
    split=provenance['split'];require(split in ('validation','test'),'Invalid evaluation split')
    panel=verify_panel(manifest,data,split)
    prior_prefixes=set()
    for name,expected in manifest.get('prior_panel_files_sha256',{}).items():
        path=repo_path(name);require(digest(path)==expected,'Previously frozen panel changed')
        prior_prefixes.update(item['prompt'] for item in read_json(path)['prompts'])
    require(not prior_prefixes.intersection(item['prompt'] for item in panel),'Panel reuses an excluded old prefix')
    rows=read_json(directory/'outputs.json');pools=read_json(directory/'candidate_pools.json');metrics=read_json(directory/'metrics.json')
    require(len({s['name'] for s in spec['models']})==len(spec['models']),'Duplicate model names')
    require([m['model'] for m in metrics]==[s['name'] for s in spec['models']],'Metric model order mismatch')
    require([(p['model'],p['prompt_index']) for p in pools]==[(s['name'],j) for s in spec['models'] for j in range(len(panel))],'Pool membership/order mismatch')
    lookup={(p['model'],p['prompt_index']):p for p in pools};index=TrainingNoveltyIndex(data['train'])
    ngrams,threshold,ignored=build_ngrams(index,spec['search']['min_match_words'])
    cache={};expected_outputs=[];checks=[];candidate_count=0;maximum_raw=0.;maximum_search=0.
    for model_spec,recorded in zip(spec['models'],metrics):
        model,vocab,components=load_model(model_spec,'cpu',data['metadata'])
        require(vocab.to_dict()==data['vocab'].to_dict(),'Model vocabulary mismatch')
        require(components==recorded['components'],'Checkpoint component binding mismatch')
        for component in model_spec['components']:
            config=read_json(repo_path(component['checkpoint']).parent.parent/'config.json')
            require(config['data_metadata']==data['metadata'],'Model training partition/config metadata mismatch')
        replay=run_epoch(model,DataLoader(data[split],batch_size=128,shuffle=False,collate_fn=collate),'generation',torch.device('cpu'))
        require(replay['count']==recorded['loss']['count'],'Split target-count mismatch')
        close(replay['loss'],recorded['loss']['loss'],'CPU split-NLL mismatch',5e-4)
        close(replay['accuracy'],recorded['loss']['accuracy'],'CPU token accuracy mismatch',.5/max(1,replay['count']))
        for j,item in enumerate(panel):
            saved=lookup[(model_spec['name'],j)];union=[];by_ids={};all_candidates=[]
            expected_keys=[str(p) for p in spec['search']['copy_penalties']]
            if model_spec.get('include_old_beam',False):expected_keys.append('old_beam5')
            require(list(saved['pools'])==expected_keys,'Search pool settings/membership mismatch')
            for penalty in spec['search']['copy_penalties']:
                pool=saved['pools'][str(penalty)]
                require(len(pool)<=spec['search']['num_return_sequences'],'Too many returned search candidates')
                require(len({tuple(lexical_tokens(r['continuation'])) for r in pool})==len(pool),'Search pool lexical deduplication mismatch')
                ranks=[]
                for k,candidate in enumerate(pool):
                    require(candidate['candidate_id']==k,'Search candidate ID/order mismatch')
                    prefix_size=check_candidate(candidate,item['prompt'],vocab,index,cache,spec['search'],penalty,ngrams,threshold,ignored)
                    candidate_count+=1;all_candidates.append(candidate);ranks.append((-candidate['normalized_search_score'],tuple(candidate['ids'])))
                    key=tuple(candidate['ids']);path=dict(copy_penalty=penalty,normalized_search_score=candidate['normalized_search_score'])
                    if key in by_ids:union[by_ids[key]]['search_paths'].append(path)
                    else:
                        value=copy.deepcopy(candidate);value['candidate_id']=len(union);value['search_paths']=[path]
                        by_ids[key]=len(union);union.append(value)
                require(ranks==sorted(ranks),'Search output score ordering mismatch')
            require(saved['union']==union,'Candidate union/deduplication/path attribution mismatch')
            if all_candidates:
                maximum_raw=max(maximum_raw,verify_likelihoods(model,all_candidates,len(validate_prefix(item['prompt'],vocab))))
                maximum_search=max(maximum_search,replay_search_paths(model,all_candidates,len(validate_prefix(item['prompt'],vocab))))
            def first(pool):return dict(selected=pool[0] if pool else None,status='selected' if pool else 'no_eligible_candidate')
            variants=dict(beam12=first(saved['pools']['0.0']),soft075=first(saved['pools']['0.75']),
                direct_union=direct_selection(union),strict_union=select_candidate(union,max_similarity=.8,min_content_words=5))
            if model_spec.get('include_old_beam',False):
                old=saved['pools']['old_beam5'];prefix_size=verify_pool(dict(candidates=old),item['prompt'],vocab,index,
                    dict(num_samples=0,seed=spec['search']['seed'],max_new_tokens=spec['search']['max_new_tokens']))
                candidate_count+=len(old);maximum_raw=max(maximum_raw,verify_likelihoods(model,old,prefix_size))
                variants['old_beam5']=first(old)
            for mode in model_spec.get('modes',list(variants)):
                selection=variants[mode];selected=selection.pop('selected')
                expected_outputs.append(dict(candidate=model_spec['name']+'__'+mode,model_spec=model_spec,mode=mode,
                    prompt_index=j,prompt=item['prompt'],seen_prefix=item['prefix_seen_in_train'],selection=selection,
                    result=selected,text=selected['text'] if selected else ''))
        checks.append(dict(model=model_spec['name'],components=components,recorded_loss=recorded['loss'],cpu_replay=replay))
        del model
        print('Verified '+model_spec['name'],flush=True)
    require(len(rows)==len(expected_outputs),'Output count mismatch')
    for actual,expected in zip(rows,expected_outputs):
        require({key:value for key,value in actual.items() if key!='blind_id'}==expected,'Selected output differs from recorded fixed profile')
    review=verify_blind(directory,rows)
    return dict(status='verified',directory=str(directory.relative_to(ROOT)),split=split,prompt_count=len(panel),
        source_count=len(spec['source_sha256']),candidate_path_count=candidate_count,unique_novelty_recomputations=len(cache),
        output_count=len(rows),blind_review=review,models=checks,maximum_raw_likelihood_error=maximum_raw,
        maximum_search_likelihood_error=maximum_search,likelihood_tolerance_per_generated_token=.0002,
        artifacts_sha256={name:digest(directory/name) for name in FILES},spec_sha256=digest(spec_path),panel_sha256=digest(panel_path),
        train_rows_sha256=rows_digest(data['train']),evaluation_rows_sha256=rows_digest(data[split]),
        limitations=['Scores/path constraints/copy events/returned selection reconstructed; entire beam pruning not rerun.',
            'Equivalent-token-sequence count checked as positive, not regenerated from all hidden beam hypotheses.',
            'Ratings audited for complete unique-ID coverage and types, not semantic correctness.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',nargs=3,action='append',metavar=('COMPARISON','SPEC','PANEL'))
    parser.add_argument('--output',type=Path,default=BASE/'verification_v1.json')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Audits are immutable; use a new output path')
    torch.set_num_threads(4)
    data=Dataset_Loader(ROOT/'data/stage_4_data',seed=1704).load_generation()
    jobs=args.job or [(BASE/'dev_comparison',BASE/'experiment_spec.json',BASE/'dev_panel.json')]
    result=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=digest(__file__),
        helper_sha256=digest(ROOT/'script/stage_4_script/verify_generalization.py'),torch=str(torch.__version__),device='cpu',audits=[])
    try:
        for directory,spec,panel in jobs:result['audits'].append(audit(Path(directory),Path(spec),Path(panel),data))
        result['status']='verified_available_comparisons'
        write_json(args.output,result);print(json.dumps(dict(status=result['status'],output=str(args.output))))
    except Exception as error:
        result['status']='failed';result['error']=str(error);write_json(args.output,result);raise


if __name__=='__main__':main()
