"""Bounded alternatives to post-filtering: semantic priors and copy-aware search."""
from __future__ import annotations
import copy
import hashlib
import math
from pathlib import Path
from local_code.stage_4_code.Generation_Optimization import load_generation_checkpoint
from local_code.stage_4_code.Generation_Ensemble import MixtureModel
from local_code.stage_4_code.Regularized_Generation import load_regularized_checkpoint
from local_code.stage_4_code.Semantic_Generation import load_semantic_checkpoint
from local_code.stage_4_code.Semantic_Input_Generation import load_semantic_input_checkpoint
from local_code.stage_4_code.Copy_Aware_Decoder import search_candidates,CopySpanIndex
from local_code.stage_4_code.Novel_Generation import generate_candidates,select_candidate,lexical_tokens

ROOT=Path(__file__).resolve().parents[2]
LOADERS={'original':load_generation_checkpoint,'regularized':load_regularized_checkpoint,
         'semantic_tied':load_semantic_checkpoint,'semantic_input':load_semantic_input_checkpoint}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(spec,device='cpu',metadata=None):
    components=spec['components']
    if len(components) not in (1,2):raise ValueError('Expected one or two components')
    models=[];vocab=None;provenance=[]
    for component in components:
        checkpoint=(ROOT/component['checkpoint']).resolve()
        if not checkpoint.is_relative_to(ROOT):raise ValueError('Checkpoint outside repository')
        if sha(checkpoint)!=component['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        loader=component.get('loader','original')
        if loader not in LOADERS:raise ValueError('Unknown generation architecture')
        model,current,meta=LOADERS[loader](checkpoint,device)
        if metadata is not None:
            for key in ('source_sha256','vocabulary_sha256','split_id_sha256'):
                if meta['config']['data_metadata'][key]!=metadata[key]:raise ValueError('Checkpoint data changed: '+key)
        if vocab is not None and current.to_dict()!=vocab.to_dict():raise ValueError('Component vocabularies differ')
        models.append(model);vocab=current
        provenance.append(dict(checkpoint=component['checkpoint'],checkpoint_sha256=sha(checkpoint),epoch=meta['epoch']))
    model=models[0] if len(models)==1 else MixtureModel(models,late_weight=spec['late_weight']).to(device).eval()
    return model,vocab,provenance


def direct_copy_reasons(candidate):
    """Reject full copies and copied whole continuations, including changed setups."""
    reasons=[]
    if not candidate['ended_with_eos']:reasons.append('incomplete')
    if candidate['new_word_count']<5:reasons.append('too_short')
    if candidate['repeated_trigrams']:reasons.append('repeated_trigrams')
    full,tail=candidate['novelty']['full_text'],candidate['novelty']['continuation']
    if full['exact_copy']:reasons.append('exact_full_copy')
    if tail['exact_copy'] or tail['exact_training_span']:reasons.append('copied_continuation')
    if '?' in candidate['text'] and not lexical_tokens(candidate['text'].split('?',1)[1]):
        reasons.append('question_without_answer')
    score=candidate['average_log_probability']
    if score is None or not math.isfinite(score):reasons.append('invalid_likelihood')
    return reasons


def choose_direct(candidates):
    eligible=[];rejections=[]
    for candidate in candidates:
        reasons=direct_copy_reasons(candidate)
        if reasons:rejections.append(dict(candidate_id=candidate['candidate_id'],reasons=reasons));continue
        similarity=candidate['novelty']['continuation']['nearest_similarity']
        # A small fixed soft novelty cost; it is not a learned coherence score.
        score=candidate['average_log_probability']-.5*max(0.,similarity-.65)
        eligible.append((score,candidate))
    eligible.sort(key=lambda row:(-row[0],row[1]['candidate_id']))
    return dict(selected=eligible[0][1] if eligible else None,status='selected' if eligible else 'no_eligible_candidate',
                selection_score=eligible[0][0] if eligible else None,eligible_count=len(eligible),
                rejections=rejections,method='mean_original_likelihood_minus_soft_similarity_cost')


def generate_variants(model,vocab,prompt,training_records,settings,include_old_beam=False):
    index=training_records if isinstance(training_records,CopySpanIndex) else CopySpanIndex(training_records,settings['min_match_words'])
    union=[];by_ids={};search_pools={}
    for penalty in settings['copy_penalties']:
        kwargs={key:value for key,value in settings.items() if key!='copy_penalties'}
        pool=search_candidates(model,vocab,prompt,index,copy_penalty=penalty,**kwargs)
        search_pools[str(penalty)]=pool
        for row in pool:
            key=tuple(row['ids'])
            if key in by_ids:
                union[by_ids[key]]['search_paths'].append(dict(copy_penalty=penalty,normalized_search_score=row['normalized_search_score']))
            else:
                item=copy.deepcopy(row);item['candidate_id']=len(union)
                item['search_paths']=[dict(copy_penalty=penalty,normalized_search_score=row['normalized_search_score'])]
                by_ids[key]=len(union);union.append(item)
    def first(pool):return dict(selected=pool[0] if pool else None,status='selected' if pool else 'no_eligible_candidate')
    variants=dict(beam12=first(search_pools['0.0']),soft075=first(search_pools['0.75']),
                  direct_union=choose_direct(union),strict_union=select_candidate(union,max_similarity=.8,min_content_words=5))
    if include_old_beam:
        old=generate_candidates(model,vocab,prompt,index.novelty,num_samples=0,seed=settings['seed'],max_new_tokens=settings['max_new_tokens'])
        search_pools['old_beam5']=old;variants['old_beam5']=first(old)
    return dict(pools=search_pools,union=union,variants=variants)
