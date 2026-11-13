#!/usr/bin/env python3
"""Aggregate blind ratings and lock an alternative before new test prefixes."""
from __future__ import annotations
import json
import math
import sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Experiment_Runner import write_json
from local_code.stage_4_code.Alternative_Generation import direct_copy_reasons,sha
from local_code.stage_4_code.Novel_Generation import select_candidate
from predict_alternative_generation import REQUIRED_SOURCES,verify_profile
from predict_novel_generation import training_digest
BASE=ROOT/'result/stage_4_result/generation_alternatives'


def read(path):return json.loads(Path(path).read_text())


def summarize(directory,split='validation',rating_file='coherence_ratings.json'):
    rows=read(directory/'outputs.json');blind=read(directory/'blind_review.json');mapping=read(directory/'blind_mapping.json')
    ratings=read(directory/rating_file);ids={r['id'] for r in blind}
    if len(ids)!=len(blind) or len(ids)!=len(ratings) or {r['id'] for r in ratings}!=ids:raise ValueError('Incomplete/duplicate blind ratings')
    if any(type(r['score']) is not int or r['score'] not in (0,1,2) for r in ratings):raise ValueError('Invalid rating')
    score={r['id']:r['score'] for r in ratings};text={r['id']:r['text'] for r in blind}
    if len(mapping)!=len(rows):raise ValueError('Mapping length mismatch')
    by_key={(r['candidate'],r['prompt_index']):r['id'] for r in mapping}
    if len(by_key)!=len(rows):raise ValueError('Duplicate output mapping')
    losses={r['model']:r['loss'] for r in read(directory/'metrics.json')}
    groups=defaultdict(list)
    for row in rows:
        if by_key[(row['candidate'],row['prompt_index'])]!=row['blind_id'] or row['text']!=text[row['blind_id']]:raise ValueError('Blind text mapping mismatch')
        if not row['text'] and score[row['blind_id']]!=0:raise ValueError('No-output must score zero')
        groups[row['candidate']].append(row)
    result=[]
    for name,subset in groups.items():
        model_spec=subset[0]['model_spec'];loss=losses[model_spec['name']]
        summary=dict(candidate=name,model_spec=model_spec,mode=subset[0]['mode'],count=len(subset),output_count=0,
            coherent_count=0,direct_copy_free_coherent_count=0,strict_novel_coherent_count=0,exact_full_copies=0,
            copied_continuations=0,direct_copy_count=0,high_similarity_count=0,seen_coherent=0,unseen_coherent=0,
            score_counts={str(i):0 for i in (0,1,2)},loss=loss['loss'],target_count=loss['count'],perplexity=math.exp(loss['loss']),split=split)
        for row in subset:
            coherent=score[row['blind_id']]==2
            summary['score_counts'][str(score[row['blind_id']])]+=1;summary['coherent_count']+=coherent
            summary['seen_coherent' if row['seen_prefix'] else 'unseen_coherent']+=coherent
            value=row['result']
            if value is None:continue
            summary['output_count']+=1
            full,tail=value['novelty']['full_text'],value['novelty']['continuation']
            copied=tail['exact_copy'] or tail['exact_training_span']
            summary['exact_full_copies']+=full['exact_copy'];summary['copied_continuations']+=copied
            summary['direct_copy_count']+=full['exact_copy'] or copied
            summary['high_similarity_count']+=(full['exact_copy'] or copied or tail['nearest_similarity']>.8 or tail['shared_fraction']>.8)
            summary['direct_copy_free_coherent_count']+=coherent and not direct_copy_reasons(value)
            summary['strict_novel_coherent_count']+=coherent and select_candidate([value],max_similarity=.8,min_content_words=5)['selected'] is not None
        result.append(summary)
    binding={name:sha(directory/name) for name in ['outputs.json','candidate_pools.json','blind_review.json','blind_mapping.json',rating_file,'metrics.json','provenance.json']}
    return result,binding


def ranking(row):
    return (-row['strict_novel_coherent_count'],-row['direct_copy_free_coherent_count'],-row['coherent_count'],row['direct_copy_count'],row['loss'],row['candidate'])


def qualifies(row,baseline):
    return (row['direct_copy_free_coherent_count']>baseline['direct_copy_free_coherent_count'] and
        row['coherent_count']>=baseline['coherent_count']-2 and row['direct_copy_count']<baseline['direct_copy_count'] and
        row['strict_novel_coherent_count']>=baseline['strict_novel_coherent_count'])


def main():
    if (BASE/'decision.json').exists():raise FileExistsError('Decision is already frozen')
    summaries,binding=summarize(BASE/'dev_comparison')
    baseline=next(r for r in summaries if r['candidate']=='previous_late__old_beam5')
    eligible=[r for r in summaries if qualifies(r,baseline)]
    best=min(summaries,key=ranking);selected=min(eligible,key=ranking) if eligible else baseline
    experimental=min((r for r in summaries if r['candidate']!=baseline['candidate']),key=ranking)
    spec=read(BASE/'experiment_spec.json');data=Dataset_Loader(ROOT/'data/stage_4_data',seed=1704).load_generation()
    data_binding={k:data['metadata'][k] for k in ('source_sha256','vocabulary_sha256','split_id_sha256')}
    data_binding['train_rows_sha256']=training_digest(data['train'])
    profiles={}
    for filename,row,status in [('recommendation.json',selected,'promoted_on_development' if eligible else 'previous_default_retained'),
        ('recommendation_experimental.json',experimental,'experimental_best_alternative')]:
        profile=dict(schema_version=1,candidate=row['candidate'],model_spec=row['model_spec'],mode=row['mode'],search=spec['search'],
            data_binding=data_binding,source_sha256={name:sha(ROOT/name) for name in sorted(REQUIRED_SOURCES)},
            status=status,dev_summary=row,selection_rule=spec['selection_rule'])
        verify_profile(profile,data);write_json(BASE/filename,profile);profiles[filename]=sha(BASE/filename)
    write_json(BASE/'dev_summary.json',summaries)
    decision=dict(locked_at_utc=datetime.now(timezone.utc).isoformat(),promoted=bool(eligible),selected=selected['candidate'],
        experimental=experimental['candidate'],best_overall=best['candidate'],eligible_candidates=[r['candidate'] for r in eligible],
        selection_rule=spec['selection_rule'],selection_implementation='Apply predeclared promotion constraints, then rank eligible candidates; if none qualify, retain previous default.',
        spec_sha256=sha(BASE/'experiment_spec.json'),panel_sha256=sha(BASE/'dev_panel.json'),review_bindings=binding,
        profile_sha256=profiles,finalizer_sha256=sha(__file__),test_evaluated_at_selection=False,
        caveat='Direct-copy-free is weaker than the unchanged strict .80 novelty diagnostic; neither proves semantic originality.')
    write_json(BASE/'decision.json',decision)
    print(json.dumps(dict(selected=selected['candidate'],promoted=bool(eligible),experimental=experimental['candidate']),indent=2))


if __name__=='__main__':main()
