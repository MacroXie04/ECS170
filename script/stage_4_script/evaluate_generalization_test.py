#!/usr/bin/env python3
"""Select new test prefixes only after the generation profile decision is locked."""
from __future__ import annotations
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(Path(__file__).resolve().parent))
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Generation_Optimization import validate_prefix
from local_code.stage_4_code.Experiment_Runner import write_json
from compare_generation import first_three_words
from predict_novel_generation import training_digest, verify_profile


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def main():
    base = ROOT/'result/stage_4_result/generation_generalization'
    decision = read(base/'decision.json')
    if (base/'test_panel.json').exists() or (base/'final_test').exists():
        raise FileExistsError('Final test evaluation is immutable; do not regenerate or tune on it')
    data = Dataset_Loader(ROOT/'data/stage_4_data',seed=1704).load_generation()
    profiles = []
    for name, expected in decision['profile_sha256'].items():
        if sha(base/name) != expected: raise ValueError('Locked profile changed')
        profile = read(base/name); verify_profile(profile,data); profiles.append(profile)
    excluded = {tuple(first_three_words(row['prompt'])) for path in [base/'dev_panel.json',
        ROOT/'result/stage_4_result/generation_optimization/comparison/panel.json'] for row in read(path)['prompts']}
    frequencies = Counter(tuple(first_three_words(row['text'])) for row in data['train'])
    seen, unseen, unique = [], [], set()
    for row in sorted(data['test'],key=lambda row:row['id']):
        words = tuple(first_three_words(row['text']))
        if words in excluded or words in unique: continue
        prompt = ' '.join(words)
        try: validate_prefix(prompt,data['vocab'])
        except ValueError: continue
        unique.add(words)
        item = dict(id=row['id'],prompt=prompt,train_prefix_count=frequencies[words],
                    prefix_seen_in_train=frequencies[words]>0,source_split='test')
        (seen if item['prefix_seen_in_train'] else unseen).append(item)
    rng = random.Random(170499); rng.shuffle(seen); rng.shuffle(unseen)
    panel = seen[:8]+unseen[:8]
    if len(panel) < 16:
        remainder = seen[8:]+unseen[8:]; rng.shuffle(remainder); panel.extend(remainder[:16-len(panel)])
    if len(panel) != 16: raise ValueError('Too few eligible fresh test prefixes')
    manifest = dict(schema_version=1,seed=170499,prompts=panel,
        sha256=hashlib.sha256(json.dumps(panel,sort_keys=True).encode()).hexdigest(),
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),decision_sha256=sha(base/'decision.json'),
        train_rows_sha256=training_digest(data['train']),test_rows_sha256=training_digest(data['test']),
        vocabulary_sha256=data['metadata']['vocabulary_sha256'],
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
                       'local_code/stage_4_code/Dataset_Loader.py':sha(ROOT/'local_code/stage_4_code/Dataset_Loader.py')},
        coverage=dict(eligible_seen=len(seen),eligible_unseen=len(unseen),selected_seen=sum(i['prefix_seen_in_train'] for i in panel),
                      selected_unseen=sum(not i['prefix_seen_in_train'] for i in panel)),
        selection='Seeded unique in-vocabulary test prefixes, targeting8 seen+8 unseen; exclude all old12 and new16 development prefixes. No reference completions supplied to models/reviewer.',
        limitation='New qualitative test-prefix evaluation after profile lock. The same dataset test loss was reported in previous coursework experiments; this is not an entirely new dataset. No tuning follows these results.')
    write_json(base/'test_panel.json',manifest)
    by_model = {}
    for profile in profiles:
        spec = json.loads(json.dumps(profile['model_spec']))
        name = spec['name']
        if name not in by_model: spec['modes']=[]; by_model[name]=spec
        if profile['mode'] not in by_model[name]['modes']: by_model[name]['modes'].append(profile['mode'])
    initial = read(base/'initial_spec.json')
    final = dict(models=list(by_model.values()),sampling=initial['sampling'],screening=initial['screening'],
        selection_rule='No test-based selection: preserve decision.json and all locked profiles.',
        decision_sha256=sha(base/'decision.json'),profile_sha256=decision['profile_sha256'])
    write_json(base/'final_spec.json',final)
    subprocess.run([sys.executable,str(ROOT/'script/stage_4_script/compare_generalization.py'),
        '--spec',str(base/'final_spec.json'),'--panel',str(base/'test_panel.json'),
        '--output',str(base/'final_test'),'--split','test','--device','cuda'],check=True,cwd=ROOT)
    for name, expected in decision['profile_sha256'].items():
        if sha(base/name) != expected: raise ValueError('Locked profile changed during evaluation')


if __name__=='__main__': main()
