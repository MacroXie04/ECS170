#!/usr/bin/env python3
"""Independently audit the frozen alternative decision and final review summaries.

No model generations, semantic ratings, or saved experiment outputs are changed.
Use a new --output path to rerun; default evidence is immutable.
"""
from __future__ import annotations
import ast
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import argparse
import random
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'script/stage_4_script'))
import torch
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Experiment_Runner import write_json
from local_code.stage_4_code.Generation_Optimization import validate_prefix
from local_code.stage_4_code.Novel_Generation import select_candidate
from local_code.stage_4_code.Alternative_Generation import load_model
from predict_alternative_generation import verify_profile
from verify_generalization import first_three
import verify_alternatives as audit


def verify_guard():
    torch.set_num_threads(4)
    base=ROOT/'result/stage_4_result/generation_alternatives'
    read=audit.read_json;sha=audit.digest;require=audit.require
    previous=read(base/'selection_guard_update/decision_before_guard_patch.json')
    decision=read(base/'decision.json');spec=read(base/'experiment_spec.json')
    old_source=base/'selection_guard_update/finalizer_before_guard_patch.py'
    new_source=ROOT/'script/stage_4_script/finalize_alternatives.py'
    require(sha(old_source)==previous['finalizer_sha256'],'Archived pre-guard finalizer does not match prior decision')
    require(sha(new_source)==decision['finalizer_sha256'],'Guarded finalizer does not match current decision')
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    old_functions,new_functions=functions(old_source),functions(new_source)
    for name in ('ranking','qualifies'):
        require(old_functions[name]==new_functions[name],'Guard patch changes selection logic: '+name)
    for key,value in previous.items():
        if key not in ('locked_at_utc','finalizer_sha256'):
            require(decision.get(key)==value,'Guard patch changed substantive decision field: '+key)
    require(decision['test_evaluated_at_selection'] is False,'Selection occurred after test evaluation')
    require(decision['spec_sha256']==sha(base/'experiment_spec.json'),'Decision spec hash mismatch')
    require(decision['panel_sha256']==sha(base/'dev_panel.json'),'Decision panel hash mismatch')
    require(decision['selection_rule']==spec['selection_rule'],'Decision selection criterion changed')
    data=Dataset_Loader(ROOT/'data/stage_4_data',seed=1704).load_generation()
    manifest=read(base/'dev_panel.json');panel=audit.verify_panel(manifest,data,'validation')
    v1=read(base/'verification_v1.json')
    require(v1['auditor_sha256']==sha(ROOT/'script/stage_4_script/verify_alternatives.py'),'Frozen auditor source changed')
    require(v1['helper_sha256']==sha(ROOT/'script/stage_4_script/verify_generalization.py'),'Frozen audit helper changed')
    require(v1['status']=='verified_available_comparisons' and len(v1['audits'])==1,'Expected successful development path audit')
    for name,expected in v1['audits'][0]['artifacts_sha256'].items():
        require(sha(base/'dev_comparison'/name)==expected,'Previously verified generated artifact changed: '+name)
    for name,expected in decision['review_bindings'].items():
        require(sha(base/'dev_comparison'/name)==expected,'Decision review binding changed: '+name)
    audit.verify_sources(spec['source_sha256'],audit.REQUIRED_SOURCES)
    rows=read(base/'dev_comparison/outputs.json')
    review=audit.verify_blind(base/'dev_comparison',rows)
    ratings={r['id']:r['score'] for r in read(base/'dev_comparison/coherence_ratings.json')}
    metrics={m['model']:m['loss'] for m in read(base/'dev_comparison/metrics.json')}
    groups=defaultdict(list)
    for row in rows:groups[row['candidate']].append(row)
    expected_names=set()
    for model in spec['models']:
        modes=model.get('modes',['beam12','soft075','direct_union','strict_union']+(['old_beam5'] if model.get('include_old_beam') else []))
        expected_names.update(model['name']+'__'+mode for mode in modes)
    require(set(groups)==expected_names,'Missing or extra complete candidate families')
    summaries=[]
    for name,subset in groups.items():
        require(len(subset)==len(panel) and {r['prompt_index'] for r in subset}==set(range(len(panel))),'Incomplete candidate prompt panel')
        model_spec=subset[0]['model_spec'];loss=metrics[model_spec['name']]
        value=dict(candidate=name,model_spec=model_spec,mode=subset[0]['mode'],count=len(subset),output_count=0,
            coherent_count=0,direct_copy_free_coherent_count=0,strict_novel_coherent_count=0,exact_full_copies=0,
            copied_continuations=0,direct_copy_count=0,high_similarity_count=0,seen_coherent=0,unseen_coherent=0,
            score_counts={str(i):0 for i in (0,1,2)},loss=loss['loss'],target_count=loss['count'],perplexity=math.exp(loss['loss']),split='validation')
        for row in subset:
            score=ratings[row['blind_id']];coherent=score==2
            value['score_counts'][str(score)]+=1;value['coherent_count']+=coherent
            value['seen_coherent' if row['seen_prefix'] else 'unseen_coherent']+=coherent
            candidate=row['result']
            if candidate is None:continue
            value['output_count']+=1
            full,tail=candidate['novelty']['full_text'],candidate['novelty']['continuation']
            copied=tail['exact_copy'] or tail['exact_training_span']
            value['exact_full_copies']+=full['exact_copy'];value['copied_continuations']+=copied
            value['direct_copy_count']+=full['exact_copy'] or copied
            value['high_similarity_count']+=full['exact_copy'] or copied or tail['nearest_similarity']>.8 or tail['shared_fraction']>.8
            value['direct_copy_free_coherent_count']+=coherent and audit.direct_selection([candidate])['selected'] is not None
            value['strict_novel_coherent_count']+=coherent and select_candidate([candidate],max_similarity=.8,min_content_words=5)['selected'] is not None
        summaries.append(value)
    require(summaries==read(base/'dev_summary.json'),'Development summary fields differ from independently recomputed ratings')
    baseline=next(row for row in summaries if row['candidate']=='previous_late__old_beam5')
    def rank(row):return (-row['strict_novel_coherent_count'],-row['direct_copy_free_coherent_count'],-row['coherent_count'],row['direct_copy_count'],row['loss'],row['candidate'])
    eligible=[r for r in summaries if r['direct_copy_free_coherent_count']>baseline['direct_copy_free_coherent_count'] and
        r['coherent_count']>=baseline['coherent_count']-2 and r['direct_copy_count']<baseline['direct_copy_count'] and
        r['strict_novel_coherent_count']>=baseline['strict_novel_coherent_count']]
    selected=min(eligible,key=rank) if eligible else baseline
    experimental=min((r for r in summaries if r['candidate']!=baseline['candidate']),key=rank)
    require(decision['selected']==selected['candidate'] and decision['experimental']==experimental['candidate'],'Selected profiles violate predeclared ranking')
    require(decision['promoted']==bool(eligible) and decision['eligible_candidates']==[r['candidate'] for r in eligible],'Promotion eligibility mismatch')
    require(decision['best_overall']==min(summaries,key=rank)['candidate'],'Best overall candidate mismatch')
    profiles={}
    for filename,expected in decision['profile_sha256'].items():
        path=base/filename;require(sha(path)==expected==previous['profile_sha256'][filename],'Profile bytes changed during guard patch')
        profile=read(path);verify_profile(profile,data)
        chosen=selected if filename=='recommendation.json' else experimental
        require(profile['candidate']==chosen['candidate'] and profile['dev_summary']==chosen,'Profile summary or candidate mismatch')
        require(profile['model_spec']==chosen['model_spec'] and profile['mode']==chosen['mode'],'Profile model or mode mismatch')
        require(profile['search']==spec['search'] and profile['selection_rule']==spec['selection_rule'],'Profile search or criterion mismatch')
        model,vocab,components=load_model(profile['model_spec'],'cpu',data['metadata'])
        require(vocab.to_dict()==data['vocab'].to_dict(),'Profile model vocabulary mismatch')
        profiles[filename]=dict(sha256=expected,candidate=profile['candidate'],status=profile['status'],components=components)
        del model
    record=dict(status='verified',created_at_utc=datetime.now(timezone.utc).isoformat(),audit_script_sha256=sha(__file__),
        archived_decision_sha256=sha(base/'selection_guard_update/decision_before_guard_patch.json'),
        archived_finalizer_sha256=sha(old_source),current_decision_sha256=sha(base/'decision.json'),
        current_finalizer_sha256=sha(new_source),auditor_source_sha256=sha(ROOT/'script/stage_4_script/verify_alternatives.py'),
        allowed_decision_changes=['locked_at_utc','finalizer_sha256'],ranking_and_qualifies_ast_unchanged=True,
        complete_candidate_families=len(groups),mapped_outputs=len(rows),review=review,promoted=decision['promoted'],
        selected=selected['candidate'],experimental=experimental['candidate'],profiles=profiles,
        baseline_counts={k:baseline[k] for k in ('coherent_count','direct_copy_free_coherent_count','strict_novel_coherent_count','direct_copy_count')},
        experimental_counts={k:experimental[k] for k in ('coherent_count','direct_copy_free_coherent_count','strict_novel_coherent_count','direct_copy_count')},
        note='No semantic ratings or model generations changed; this audit checks bindings and independently recomputes the recorded decision.')
    return record


def summary(directory,split,ratings_file,expected_count):
    read=audit.read_json;require=audit.require
    rows=read(directory/'outputs.json');scores={r['id']:r['score'] for r in read(directory/ratings_file)}
    metrics={r['model']:r['loss'] for r in read(directory/'metrics.json')};groups=defaultdict(list)
    for row in rows:groups[row['candidate']].append(row)
    result=[]
    for name,subset in groups.items():
        require(len(subset)==expected_count and {r['prompt_index'] for r in subset}==set(range(expected_count)),'Incomplete summary panel')
        model=subset[0]['model_spec'];loss=metrics[model['name']]
        value=dict(candidate=name,model_spec=model,mode=subset[0]['mode'],count=len(subset),output_count=0,
            coherent_count=0,direct_copy_free_coherent_count=0,strict_novel_coherent_count=0,exact_full_copies=0,
            copied_continuations=0,direct_copy_count=0,high_similarity_count=0,seen_coherent=0,unseen_coherent=0,
            score_counts={str(i):0 for i in (0,1,2)},loss=loss['loss'],target_count=loss['count'],perplexity=math.exp(loss['loss']),split=split)
        for row in subset:
            score=scores[row['blind_id']];coherent=score==2
            value['score_counts'][str(score)]+=1;value['coherent_count']+=coherent
            value['seen_coherent' if row['seen_prefix'] else 'unseen_coherent']+=coherent
            candidate=row['result']
            if candidate is None:continue
            value['output_count']+=1
            full,tail=candidate['novelty']['full_text'],candidate['novelty']['continuation']
            copied=tail['exact_copy'] or tail['exact_training_span']
            value['exact_full_copies']+=full['exact_copy'];value['copied_continuations']+=copied
            value['direct_copy_count']+=full['exact_copy'] or copied
            value['high_similarity_count']+=full['exact_copy'] or copied or tail['nearest_similarity']>.8 or tail['shared_fraction']>.8
            value['direct_copy_free_coherent_count']+=coherent and audit.direct_selection([candidate])['selected'] is not None
            value['strict_novel_coherent_count']+=coherent and select_candidate([candidate],max_similarity=.8,min_content_words=5)['selected'] is not None
        result.append(value)
    return result,scores


def verify_final(guard):
    read=audit.read_json;sha=audit.digest;require=audit.require
    base=ROOT/'result/stage_4_result/generation_alternatives'
    report=read(base/'final_summary.json');decision=read(base/'decision.json')
    require(report['decision_sha256']==sha(base/'decision.json')==guard['current_decision_sha256'],'Decision changed after guard verification')
    require(report['profile_selection_changed'] is False,'Final evaluation changed selection')
    require(report['profile_sha256']==decision['profile_sha256'],'Final profile binding mismatch')
    for name,expected in report['profile_sha256'].items():require(sha(base/name)==expected,'Locked profile changed')
    audit.verify_sources(report['source_sha256'],{'script/stage_4_script/summarize_alternatives_test.py','script/stage_4_script/finalize_alternatives.py'})
    paths=read(base/'verification_final_paths.json')
    require(paths['status']=='verified_available_comparisons' and len(paths['audits'])==2,'Expected both successful path audits')
    require(paths['auditor_sha256']==sha(ROOT/'script/stage_4_script/verify_alternatives.py'),'Frozen path auditor source changed')
    require(paths['helper_sha256']==sha(ROOT/'script/stage_4_script/verify_generalization.py'),'Frozen path audit helper changed')
    data=Dataset_Loader(ROOT/'data/stage_4_data',seed=1704).load_generation()
    prior_paths=[ROOT/'result/stage_4_result/generation_optimization/comparison/panel.json',
        ROOT/'result/stage_4_result/generation_generalization/dev_panel.json',
        ROOT/'result/stage_4_result/generation_generalization/test_panel.json',base/'dev_panel.json']
    dev=read(base/'dev_panel.json')
    for name,expected in dev['prior_panel_files_sha256'].items():require(sha(ROOT/name)==expected,'Previously exposed panel changed')
    excluded={tuple(first_three(r['prompt'])) for path in prior_paths for r in read(path)['prompts']}
    frequencies=Counter(tuple(first_three(r['text'])) for r in data['train'])
    seen=[];unseen=[];unique=set()
    for row in sorted(data['test'],key=lambda r:r['id']):
        prefix=tuple(first_three(row['text']));prompt=' '.join(prefix)
        if prefix in excluded or prefix in unique:continue
        try:validate_prefix(prompt,data['vocab'])
        except ValueError:continue
        unique.add(prefix)
        item=dict(id=row['id'],prompt=prompt,train_prefix_count=frequencies[prefix],prefix_seen_in_train=frequencies[prefix]>0,source_split='test')
        (seen if item['prefix_seen_in_train'] else unseen).append(item)
    rng=random.Random(170455);rng.shuffle(seen);rng.shuffle(unseen);expected_test=seen[:8]+unseen[:8]
    rest=seen[8:]+unseen[8:];rng.shuffle(rest);expected_test+=rest[:16-len(expected_test)]
    require(read(base/'test_panel.json')['prompts']==expected_test,'Fresh test-prefix seeded selection differs')
    require(read(base/'regression_panel.json')['prompts']==read(prior_paths[0])['prompts'],'Regression is not the original12prefix panel')
    final_spec=read(base/'final_spec.json')
    require(final_spec['decision_sha256']==sha(base/'decision.json') and final_spec['profile_sha256']==decision['profile_sha256'],'Final comparison spec not bound to frozen decision')
    checks=[]
    for name,split,panel_file,path_audit in zip(('final_test','regression'),('test','validation'),('test_panel.json','regression_panel.json'),paths['audits']):
        directory=base/name;panel=read(base/panel_file);recorded=report['panels'][name]
        require(path_audit['status']=='verified' and path_audit['directory']==str(directory.relative_to(ROOT)),'Wrong audited comparison')
        for filename,expected in path_audit['artifacts_sha256'].items():require(sha(directory/filename)==expected,'Audited final artifact changed')
        require(path_audit['spec_sha256']==sha(base/'final_spec.json') and path_audit['panel_sha256']==sha(base/panel_file),'Audited final spec/panel changed')
        audit.verify_panel(panel,data,split)
        require(panel['decision_sha256']==sha(base/'decision.json'),'Final panel not bound to decision')
        require(datetime.fromisoformat(panel['locked_at_utc'])>=datetime.fromisoformat(decision['locked_at_utc']),'Final prefix panel predates decision')
        require(recorded['panel_sha256']==sha(base/panel_file),'Final summary panel hash mismatch')
        coverage=dict(seen=sum(r['prefix_seen_in_train'] for r in panel['prompts']),unseen=sum(not r['prefix_seen_in_train'] for r in panel['prompts']))
        require(recorded['coverage']==panel['coverage']==coverage,'Panel coverage mismatch')
        rows=read(directory/'outputs.json');reviews=audit.verify_blind(directory,rows)
        primary,first_scores=summary(directory,split,'coherence_ratings.json',len(panel['prompts']))
        secondary,second_scores=summary(directory,split,'confirmatory_ratings.json',len(panel['prompts']))
        secondary={r['candidate']:r for r in secondary}
        for row in primary:
            second=secondary[row['candidate']]
            row['confirmatory']={key:second[key] for key in ('coherent_count','direct_copy_free_coherent_count','strict_novel_coherent_count','score_counts','seen_coherent','unseen_coherent')}
            row['reviewer_score_disagreements']=sum(first_scores[r['blind_id']]!=second_scores[r['blind_id']] for r in rows if r['candidate']==row['candidate'])
        require(primary==recorded['summaries'],'Final summary metrics differ from independent computation')
        for key in ('primary_bindings','confirmatory_bindings'):
            for filename,expected in recorded[key].items():require(sha(directory/filename)==expected,'Final summary review binding mismatch')
        checks.append(dict(comparison=name,prompt_count=len(panel['prompts']),reviews=reviews,
            candidate_counts=[{key:row[key] for key in ('candidate','count','coherent_count','direct_copy_free_coherent_count','strict_novel_coherent_count','exact_full_copies','copied_continuations','direct_copy_count','confirmatory','reviewer_score_disagreements')} for row in primary]))
    result=dict(status='verified',created_at_utc=datetime.now(timezone.utc).isoformat(),audit_script_sha256=sha(__file__),
        decision_sha256=sha(base/'decision.json'),final_summary_sha256=sha(base/'final_summary.json'),
        final_paths_audit_sha256=sha(base/'verification_final_paths.json'),selection_guard_audit_sha256=sha(base/'verification_selection_guard.json'),
        profile_sha256=decision['profile_sha256'],profile_selection_changed=False,seeded_test_prefix_selection_verified=True,
        original12_regression_panel_verified=True,panels=checks,note='No model replay or semantic rerating; all summaries and bindings independently checked.')
    result['guard_integrity']=guard
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'result/stage_4_result/generation_alternatives/verification_final_summary.json')
    parser.add_argument('--guard-only',action='store_true',help='Audit the preserved development decision and profiles only')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Audits are immutable; use a new --output path')
    guard=verify_guard()
    result=guard if args.guard_only else verify_final(guard)
    write_json(args.output,result)
    print(json.dumps(dict(status=result['status'],output=str(args.output),guard_only=args.guard_only)))


if __name__=='__main__':main()
