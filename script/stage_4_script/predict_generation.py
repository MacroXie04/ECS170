#!/usr/bin/env python3
"""Continue exactly three words with the validated Stage 4 generation profile."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from local_code.stage_4_code.Generation_Decoder import decode
from local_code.stage_4_code.Generation_Optimization import load_generation_checkpoint, validate_prefix, format_completion

DEFAULT_PROFILE=ROOT/'result/stage_4_result/generation_optimization/recommendation.json'


def predict(prompt, *, profile=DEFAULT_PROFILE, checkpoint=None, device='auto', strategy=None,
            temperature=None, top_p=None, seed=1704, max_new_tokens=None):
    if device not in ('auto','cuda','cpu'):
        raise ValueError('Device must be auto, cuda, or cpu')
    if not isinstance(prompt,str) or not prompt.strip():
        raise ValueError('Supply a nonempty three-word prompt')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    if device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu')
    if checkpoint is None:
        settings=json.loads(Path(profile).read_text())
        checkpoint=ROOT/settings['checkpoint']
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=settings['checkpoint_sha256']:
            raise ValueError('Recommended checkpoint has changed; rerun verification before using this profile')
        for name,expected in settings.get('source_sha256',{}).items():
            source=(ROOT/name).resolve()
            if not source.is_relative_to(ROOT) or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
                raise ValueError('Generation source changed; revalidate the profile: '+name)
        decoding=dict(settings['decoding'])
        profile_name=settings['candidate']
    else:
        checkpoint=Path(checkpoint).expanduser().resolve()
        decoding=dict(strategy='beam',max_new_tokens=48,min_new_tokens=4,beam_width=5,
                      length_penalty=.7,no_repeat_ngram_size=3,repetition_penalty=1.1,seed=seed)
        profile_name='explicit_checkpoint'
    decoding['seed']=seed
    for key,value in [('strategy',strategy),('temperature',temperature),('top_p',top_p),('max_new_tokens',max_new_tokens)]:
        if value is not None:
            decoding[key]=value
    model,vocab,provenance=load_generation_checkpoint(checkpoint,device)
    ids=validate_prefix(prompt,vocab)
    result=decode(model,ids,**decoding)
    return dict(prompt=prompt.strip(),text=format_completion(prompt,result['ids'],len(ids),vocab),
                profile=profile_name,cell=provenance['config']['model']['cell'],epoch=provenance['epoch'],
                device=device,ended_with_eos=result['ended_with_eos'],finish_reason=result['finish_reason'],
                generated_tokens=result['generated_token_count'],decoding=result['decoding'],
                checkpoint_sha256=provenance['checkpoint_sha256'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    text=parser.add_mutually_exclusive_group(required=True)
    text.add_argument('--text');text.add_argument('--text-file',type=Path)
    parser.add_argument('--profile',type=Path,default=DEFAULT_PROFILE)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    parser.add_argument('--strategy',choices=['beam','sample','greedy'])
    parser.add_argument('--temperature',type=float)
    parser.add_argument('--top-p',type=float)
    parser.add_argument('--seed',type=int,default=1704)
    parser.add_argument('--max-new-tokens',type=int)
    args=parser.parse_args(argv)
    torch.set_num_threads(4)
    try:
        prompt=args.text if args.text is not None else args.text_file.read_text()
        result=predict(prompt,profile=args.profile,checkpoint=args.checkpoint,device=args.device,
                       strategy=args.strategy,temperature=args.temperature,top_p=args.top_p,seed=args.seed,
                       max_new_tokens=args.max_new_tokens)
    except (ValueError,KeyError,OSError,RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    main()
