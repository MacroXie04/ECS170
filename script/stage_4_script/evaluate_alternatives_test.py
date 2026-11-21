#!/usr/bin/env python3
"""Evaluate frozen alternatives on fresh test prefixes and the original regression panel."""
from __future__ import annotations
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Generation_Optimization import validate_prefix
from local_code.stage_4_code.Experiment_Runner import write_json
from local_code.stage_4_code.Alternative_Generation import sha
from compare_generation import first_three_words
from predict_novel_generation import training_digest
from predict_alternative_generation import verify_profile


def read(path):return json.loads(Path(path).read_text())


def main():
    base=ROOT/'result/stage_4_result/generation_alternatives'
    if (base/'test_panel.json').exists():raise FileExistsError('Final evaluation already frozen')
    decision=read(base/'decision.json');experiment=read(base/'experiment_spec.json')
    if sha(base/'experiment_spec.json')!=decision['spec_sha256']:raise ValueError('Experiment spec changed')
    data=Dataset_Loader(ROOT/'data/stage_4_data',seed=1704).load_generation();profiles=[]
    for name,expected in decision['profile_sha256'].items():
        if sha(base/name)!=expected:raise ValueError('Locked profile changed')
        profile=read(base/name);verify_profile(profile,data);profiles.append(profile)
    prior_paths=[ROOT/'result/stage_4_result/generation_optimization/comparison/panel.json',
        ROOT/'result/stage_4_result/generation_generalization/dev_panel.json',
        ROOT/'result/stage_4_result/generation_generalization/test_panel.json',base/'dev_panel.json']
    excluded={tuple(first_three_words(row['prompt'])) for path in prior_paths for row in read(path)['prompts']}
    frequencies=Counter(tuple(first_three_words(row['text'])) for row in data['train'])
    seen=[];unseen=[];unique=set()
    for row in sorted(data['test'],key=lambda r:r['id']):
        prefix=tuple(first_three_words(row['text']));prompt=' '.join(prefix)
        if prefix in excluded or prefix in unique:continue
        try:validate_prefix(prompt,data['vocab'])
        except ValueError:continue
        unique.add(prefix)
        item=dict(id=row['id'],prompt=prompt,train_prefix_count=frequencies[prefix],prefix_seen_in_train=frequencies[prefix]>0,source_split='test')
        (seen if item['prefix_seen_in_train'] else unseen).append(item)
    rng=random.Random(170455);rng.shuffle(seen);rng.shuffle(unseen);panel=seen[:8]+unseen[:8]
    rest=seen[8:]+unseen[8:];rng.shuffle(rest);panel+=rest[:16-len(panel)]
    if len(panel)!=16:raise ValueError('Too few fresh test prefixes')
    def manifest(prompts,split,description):
        return dict(prompts=prompts,sha256=hashlib.sha256(json.dumps(prompts,sort_keys=True).encode()).hexdigest(),
            locked_at_utc=datetime.now(timezone.utc).isoformat(),decision_sha256=sha(base/'decision.json'),
            train_rows_sha256=training_digest(data['train']),**{split+'_rows_sha256':training_digest(data[split])},
            vocabulary_sha256=data['metadata']['vocabulary_sha256'],
            source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),'local_code/stage_4_code/Dataset_Loader.py':sha(ROOT/'local_code/stage_4_code/Dataset_Loader.py')},
            selection=description,coverage=dict(seen=sum(r['prefix_seen_in_train'] for r in prompts),unseen=sum(not r['prefix_seen_in_train'] for r in prompts)),
            limitation='No tuning follows these outputs. Test corpus loss has been used previously; fresh prefix panel is not a wholly new dataset.')
    write_json(base/'test_panel.json',manifest(panel,'test','Seed170455; target8seen+8unseen unique in-vocabulary prefixes, excluding all4previous panels; fill anyshortfallfrom remainingeligible prefixes.'))
    original=read(prior_paths[0])['prompts']
    write_json(base/'regression_panel.json',manifest(original,'validation','Original12 development prefixes reused ONLY as a regression check after locking; not fresh validation/test evidence.'))
    models={}
    baseline=json.loads(json.dumps(experiment['models'][0]));baseline['modes']=['old_beam5'];models[baseline['name']]=baseline
    for profile in profiles:
        model=json.loads(json.dumps(profile['model_spec']));name=model['name']
        if name not in models:model['modes']=[];models[name]=model
        if profile['mode'] not in models[name]['modes']:models[name]['modes'].append(profile['mode'])
    spec=dict(experiment);spec['models']=list(models.values());spec['selection_rule']='No selection or tuning: preserve locked decision and profiles.'
    spec['decision_sha256']=sha(base/'decision.json');spec['profile_sha256']=decision['profile_sha256']
    write_json(base/'final_spec.json',spec)
    for split,panel_name,directory in [('test','test_panel.json','final_test'),('validation','regression_panel.json','regression')]:
        subprocess.run([sys.executable,str(ROOT/'script/stage_4_script/compare_alternatives.py'),
            '--spec',str(base/'final_spec.json'),'--panel',str(base/panel_name),'--output',str(base/directory),'--split',split],cwd=ROOT,check=True)
    for name,expected in decision['profile_sha256'].items():
        if sha(base/name)!=expected:raise ValueError('Locked profile changed during evaluation')


if __name__=='__main__':main()
