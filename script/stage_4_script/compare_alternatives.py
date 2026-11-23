#!/usr/bin/env python3
"""Frozen-prefix comparison of predeclared copy-aware and semantic alternatives."""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader,tokenize
from local_code.stage_4_code.Experiment_Runner import collate,run_epoch,write_json
from local_code.stage_4_code.Copy_Aware_Decoder import CopySpanIndex
from local_code.stage_4_code.Alternative_Generation import sha,load_model,generate_variants
from compare_generation import first_three_words
from predict_novel_generation import training_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--split',choices=['validation','test'],default='validation')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Comparison output already exists')
    spec=json.loads(args.spec.read_text());manifest=json.loads(args.panel.read_text());panel=manifest['prompts']
    import hashlib
    if hashlib.sha256(json.dumps(panel,sort_keys=True).encode()).hexdigest()!=manifest['sha256']:raise ValueError('Panel hash changed')
    data=Dataset_Loader(ROOT/'data/stage_4_data',seed=1704).load_generation()
    if training_digest(data['train'])!=manifest['train_rows_sha256'] or training_digest(data[args.split])!=manifest[args.split+'_rows_sha256']:
        raise ValueError('Panel data rows changed')
    if data['metadata']['vocabulary_sha256']!=manifest['vocabulary_sha256']:raise ValueError('Vocabulary changed')
    for name,expected in manifest['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Panel source changed: '+name)
    records={row['id']:row for row in data[args.split]}
    for item in panel:
        if item['source_split']!=args.split or first_three_words(records[item['id']]['text'])!=tokenize(item['prompt']):raise ValueError('Panel prompt binding mismatch')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    args.output.mkdir(parents=True)
    runtime_sources=spec['source_sha256']
    for name,expected in runtime_sources.items():
        if sha(ROOT/name)!=expected:raise ValueError('Experiment source changed: '+name)
    write_json(args.output/'provenance.json',dict(spec_sha256=sha(args.spec),panel_sha256=sha(args.panel),
        source_sha256=runtime_sources,data_source_sha256=data['metadata']['source_sha256'],
        split=args.split,torch=str(torch.__version__),device='cuda'))
    index=CopySpanIndex(data['train'],spec['search']['min_match_words'])
    outputs=[];all_pools=[];metrics=[]
    for model_spec in spec['models']:
        model,vocab,provenance=load_model(model_spec,'cuda',data['metadata'])
        if vocab.to_dict()!=data['vocab'].to_dict():raise ValueError('Model vocabulary mismatch')
        loss=run_epoch(model,DataLoader(data[args.split],batch_size=128,shuffle=False,collate_fn=collate),'generation',torch.device('cuda'))
        metrics.append(dict(model=model_spec['name'],components=provenance,loss=loss))
        for j,item in enumerate(panel):
            generated=generate_variants(model,vocab,item['prompt'],index,spec['search'],model_spec.get('include_old_beam',False))
            all_pools.append(dict(model=model_spec['name'],prompt_index=j,**{k:v for k,v in generated.items() if k!='variants'}))
            wanted=model_spec.get('modes',list(generated['variants']))
            for name in wanted:
                selection=dict(generated['variants'][name]);candidate=selection.pop('selected')
                outputs.append(dict(candidate=model_spec['name']+'__'+name,model_spec=model_spec,mode=name,prompt_index=j,
                    prompt=item['prompt'],seen_prefix=item['prefix_seen_in_train'],selection=selection,result=candidate,
                    text=candidate['text'] if candidate else ''))
            print(json.dumps(dict(model=model_spec['name'],prompt_index=j)),flush=True)
        del model;torch.cuda.empty_cache()
    # Review each distinct text once. Repeated candidates map to the same blind ID.
    unique={row['text']:row['prompt'] for row in outputs};texts=list(unique);random.Random(170451).shuffle(texts)
    identifiers={text:f'A{i:03d}' for i,text in enumerate(texts,1)}
    blind=[dict(id=identifiers[text],prompt=unique[text],text=text,no_output=not bool(text)) for text in texts]
    mapping=[]
    for row in outputs:
        row['blind_id']=identifiers[row['text']]
        mapping.append(dict(id=row['blind_id'],candidate=row['candidate'],prompt_index=row['prompt_index']))
    for name,value in [('metrics.json',metrics),('outputs.json',outputs),('candidate_pools.json',all_pools),('blind_review.json',blind),('blind_mapping.json',mapping)]:write_json(args.output/name,value)
    print(f'Completed {len(outputs)} outputs, {len(blind)} distinct texts',flush=True)


if __name__=='__main__':main()
