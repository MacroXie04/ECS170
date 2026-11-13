#!/usr/bin/env python3
"""Run a frozen alternative generation profile; no hidden copied fallback."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Alternative_Generation import load_model,generate_variants,sha
from local_code.stage_4_code.Novel_Generation import generate_candidates
from predict_novel_generation import training_digest

DEFAULT_PROFILE=ROOT/'result/stage_4_result/generation_alternatives/recommendation.json'
REQUIRED_SOURCES={
    'script/stage_4_script/predict_alternative_generation.py','script/stage_4_script/predict_novel_generation.py',
    *['local_code/stage_4_code/'+name+'.py' for name in ['Alternative_Generation','Copy_Aware_Decoder','Novel_Generation',
        'Generation_Decoder','Generation_Optimization','Generation_Ensemble','Method_RNN','Dataset_Loader',
        'Regularized_Generation','Semantic_Generation','Semantic_Input_Generation']]}


def verify_profile(profile,data):
    if profile.get('schema_version')!=1 or profile['mode'] not in {'old_beam5','beam12','soft075','direct_union','strict_union'}:
        raise ValueError('Unsupported alternative profile')
    if not REQUIRED_SOURCES.issubset(profile['source_sha256']):raise ValueError('Missing required source bindings')
    for name,expected in profile['source_sha256'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path)!=expected:raise ValueError('Profile source changed: '+name)
    for key in ('source_sha256','vocabulary_sha256','split_id_sha256'):
        if profile['data_binding'][key]!=data['metadata'][key]:raise ValueError('Profile data changed: '+key)
    if profile['data_binding']['train_rows_sha256']!=training_digest(data['train']):raise ValueError('Training-copy reference changed')


def predict(prompt,profile=DEFAULT_PROFILE,device='auto'):
    path=Path(profile);settings=json.loads(path.read_text())
    data=Dataset_Loader(ROOT/'data/stage_4_data',seed=1704).load_generation();verify_profile(settings,data)
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    model,vocab,provenance=load_model(settings['model_spec'],device,data['metadata'])
    if vocab.to_dict()!=data['vocab'].to_dict():raise ValueError('Vocabulary differs from training reference')
    mode=settings['mode']
    if mode=='old_beam5':
        pool=generate_candidates(model,vocab,prompt,data['train'],num_samples=0,seed=settings['search']['seed'],max_new_tokens=settings['search']['max_new_tokens'])
        selection=dict(selected=pool[0],status='selected')
    else:
        values=generate_variants(model,vocab,prompt,data['train'],settings['search'])
        selection=dict(values['variants'][mode])
    selected=selection.pop('selected')
    return dict(profile=settings['candidate'],profile_sha256=sha(path),status=selection['status'],
        text=selected['text'] if selected else None,result=selected,selection=selection,checkpoints=provenance,
        notice='Copy diagnostics are lexical checks, not a guarantee of coherence or original humor.',
        external_word_vectors=any(c.get('loader','original').startswith('semantic') for c in settings['model_spec']['components']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--text');group.add_argument('--text-file',type=Path)
    parser.add_argument('--profile',type=Path,default=DEFAULT_PROFILE);parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args();torch.set_num_threads(4)
    prompt=args.text if args.text is not None else args.text_file.read_text(encoding='utf-8').strip()
    try:result=predict(prompt,args.profile,args.device)
    except (ValueError,KeyError,FileNotFoundError) as error:parser.exit(2,str(error)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
