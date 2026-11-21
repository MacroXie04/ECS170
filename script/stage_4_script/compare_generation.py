#!/usr/bin/env python3
"""Frozen validation-prefix comparison, including copy/loop diagnostics and blind review."""
from __future__ import annotations

import argparse
from collections import Counter
import difflib
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader, tokenize
from local_code.stage_4_code.Experiment_Runner import collate, run_epoch, write_json
from local_code.stage_4_code.Method_RNN import generate
from local_code.stage_4_code.Generation_Decoder import decode
from local_code.stage_4_code.Generation_Optimization import load_generation_checkpoint, validate_prefix

PANEL = [
    ('joke/940','what do you',153),('joke/1183','why did the',73),('joke/648','what did the',64),
    ('joke/867','how do you',25),('joke/786','what kind of',17),('joke/550','what does a',11),
    ('joke/1169','where did the',6),( 'joke/1537',"i'm reading a",1),
    ('joke/858','what side of',0),('joke/89','time flies like',0),
    ('joke/1279','a guy walks',0),('joke/1262','i was wondering',0),
]


def first_three_words(text):
    result=[]
    count=0
    for token in tokenize(text):
        result.append(token)
        count += any(char.isalnum() for char in token)
        if count == 3:
            return result
    return result


def normalized(text):
    return tuple(tokenize(text))


def repetition(tokens):
    grams = [tuple(tokens[i:i+3]) for i in range(max(0,len(tokens)-2))]
    return sum(count-1 for count in Counter(grams).values())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'result/stage_4_result/generation_optimization/comparison')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    data=Dataset_Loader(ROOT/'data/stage_4_data',seed=1704).load_generation()
    val_by_id={r['id']:r for r in data['validation']}
    panel=[]
    for identifier,prompt,frequency in PANEL:
        assert first_three_words(val_by_id[identifier]['text']) == tokenize(prompt)
        observed=sum(first_three_words(r['text'])==tokenize(prompt) for r in data['train'])
        assert observed==frequency,(prompt,observed,frequency)
        validate_prefix(prompt,data['vocab'])
        panel.append(dict(id=identifier,prompt=prompt,train_prefix_count=frequency,
                          prefix_seen_in_train=frequency>0,source_split='validation'))
    panel_hash=hashlib.sha256(json.dumps(panel,sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=args.output/'panel.json'
    if manifest.exists() and json.loads(manifest.read_text())['prompts']!=panel:
        raise ValueError('Cannot replace frozen prompt panel')
    write_json(manifest,dict(prompts=panel,sha256=panel_hash,seed=1704,
                            selection='Frozen from validation texts before optimized generations were inspected'))
    base=ROOT/'result/stage_4_result/runs'
    opt=ROOT/'result/stage_4_result/generation_optimization/runs'
    candidates=[
        ('baseline_gru_greedy',base/'generation_gru_baseline','best.pt','original'),
        ('baseline_gru_beam',base/'generation_gru_baseline','best.pt','beam'),
        ('compact_gru_best_greedy',opt/'generation_gru_compact256','best.pt','original'),
        ('compact_gru_best_beam',opt/'generation_gru_compact256','best.pt','beam'),
        ('compact_gru_late_greedy',opt/'generation_gru_compact256','last.pt','original'),
        ('compact_gru_late_beam',opt/'generation_gru_compact256','last.pt','beam'),
        ('wide_gru_best_beam',opt/'generation_gru_wide256','best.pt','beam'),
        ('wide_gru_late_beam',opt/'generation_gru_wide256','last.pt','beam'),
        ('compact_lstm_late_beam',opt/'generation_lstm_compact256','last.pt','beam'),
    ]
    train_tokens=[normalized(r['text']) for r in data['train']]
    exact=set(train_tokens)
    rows=[]; summaries=[]; nll_cache={}
    for candidate,directory,filename,strategy in candidates:
        checkpoint=directory/'checkpoints'/filename
        model,vocab,provenance=load_generation_checkpoint(checkpoint,args.device)
        assert vocab.to_dict()==data['vocab'].to_dict()
        if str(checkpoint) not in nll_cache:
            val_loader=DataLoader(data['validation'],batch_size=128,shuffle=False,collate_fn=collate)
            nll_cache[str(checkpoint)]=run_epoch(model,val_loader,'generation',torch.device(args.device))['loss']
        candidate_rows=[]
        for index,p in enumerate(panel):
            prefix=validate_prefix(p['prompt'],vocab)
            if strategy=='original':
                ids=generate(model,prefix,max_new_tokens=48,temperature=0,seed=1704)
                settings={'strategy':'original_greedy','max_new_tokens':48}
            else:
                settings={'strategy':'beam','max_new_tokens':48,'min_new_tokens':4,
                          'beam_width':5,'length_penalty':.7,'no_repeat_ngram_size':3,
                          'repetition_penalty':1.1,'seed':1704}
                generated=decode(model,prefix,**settings)
                ids=generated['ids']
            text=vocab.decode(ids)
            tokens=normalized(text)
            similarities=[difflib.SequenceMatcher(None,tokens,reference,autojunk=False).ratio() for reference in train_tokens]
            nearest=max(range(len(similarities)),key=similarities.__getitem__)
            row=dict(candidate=candidate,prompt_index=index,prompt=p['prompt'],text=text,
                     seen_prefix=p['prefix_seen_in_train'],ended_with_eos=ids[-1]==3,
                     new_token_count=len(ids)-len(prefix),repeated_trigrams=repetition(tokens),
                     exact_training_copy=tokens in exact,nearest_training_similarity=similarities[nearest],
                     nearest_training_id=data['train'][nearest]['id'],settings=settings)
            candidate_rows.append(row);rows.append(row)
        summaries.append(dict(candidate=candidate,checkpoint=str(checkpoint.relative_to(ROOT)),
            checkpoint_sha256=provenance['checkpoint_sha256'],epoch=provenance['epoch'],
            validation_loss=nll_cache[str(checkpoint)],panel_size=len(panel),
            eos_rate=sum(r['ended_with_eos'] for r in candidate_rows)/len(panel),
            repeated_trigram_outputs=sum(r['repeated_trigrams']>0 for r in candidate_rows),
            exact_training_copies=sum(r['exact_training_copy'] for r in candidate_rows),
            near_training_copies=sum(r['nearest_training_similarity']>=.9 for r in candidate_rows)))
        print(json.dumps(summaries[-1]),flush=True)
    # Blinded IDs are randomized, do not encode a candidate or model name.
    indices=list(range(len(rows)));random.Random(170412).shuffle(indices)
    blind=[];mapping=[]
    for order,index in enumerate(indices,1):
        identifier=f'G{order:03d}'
        rows[index]['blind_id']=identifier
        blind.append({'id':identifier,'text':rows[index]['text']})
        mapping.append({'id':identifier,'candidate':rows[index]['candidate'],'prompt_index':rows[index]['prompt_index']})
    write_json(args.output/'outputs.json',rows)
    write_json(args.output/'summary.json',summaries)
    write_json(args.output/'blind_review.json',blind)
    write_json(args.output/'blind_mapping.json',mapping)
    write_json(args.output/'provenance.json',dict(panel_sha256=panel_hash,
        dataset_source_sha256=data['metadata']['source_sha256'],vocabulary_sha256=data['metadata']['vocabulary_sha256'],
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                      [Path(__file__).resolve(),ROOT/'local_code/stage_4_code/Generation_Decoder.py',
                       ROOT/'local_code/stage_4_code/Generation_Optimization.py']},
        purpose='Validation-only qualitative selection; test texts never used in this comparison',
        copy_metric='SequenceMatcher token ratio >=0.90 is a near copy; exact normalized token equality separately'))


if __name__=='__main__':
    main()
