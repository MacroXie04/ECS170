#!/usr/bin/env python3
"""Two fixed own-GRU experiments with disclosed external pretrained word vectors."""
from __future__ import annotations
import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Experiment_Runner import collate,run_epoch,write_json,save_checkpoint
from local_code.stage_4_code.Semantic_Generation import (
    SemanticGenerationGRU,canonical_digest,file_digest,load_word_prior,load_semantic_checkpoint)

EXPERIMENTS={'glove100_finetune':0.,'glove100_anchor01':.1}
SOURCES=['local_code/stage_4_code/Semantic_Generation.py','local_code/stage_4_code/Dataset_Loader.py',
         'local_code/stage_4_code/Experiment_Runner.py','script/stage_4_script/train_semantic_generation.py']


def train_epoch(model,loader,optimizer,device,coefficient):
    model.train();total_nll,total_objective,correct,count=0.,0.,0,0
    for tokens,_,_,_ in loader:
        tokens=tokens.to(device);targets=tokens[:,1:]
        logits=model(tokens[:,:-1]);mask=targets.ne(0);n=int(mask.sum())
        nll=nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten(),ignore_index=0,reduction='sum')
        objective=nll/n+coefficient*model.anchor_penalty()
        if not torch.isfinite(objective):raise RuntimeError('Non-finite training loss')
        optimizer.zero_grad(set_to_none=True);objective.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if torch.count_nonzero(model.embedding.weight[0]):raise RuntimeError('PAD embedding drift')
        total_nll+=float(nll.detach());total_objective+=float(objective.detach())*n
        correct+=int((logits.argmax(-1).eq(targets)&mask).sum());count+=n
    return dict(loss=total_nll/count,objective=total_objective/count,accuracy=correct/count,count=count)


def experiment(name,anchor,data,prior,matched,pretrained,args,device):
    directory=Path(args.output)/name;checkpoints=directory/'checkpoints';checkpoints.mkdir(parents=True,exist_ok=True)
    kwargs=dict(vocab_size=len(data['vocab']),embedding_dim=100,hidden_dim=256,dropout=.2)
    source_hashes={name:file_digest(ROOT/name) for name in SOURCES}
    config=dict(architecture='semantic_prior_gru_v1',model=kwargs,seed=1704,max_epochs=args.max_epochs,patience=15,
        batch_size=64,learning_rate=.001,optimizer='AdamW',weight_decay=.01,gradient_clip_norm=1.,anchor_coefficient=anchor,
        anchor_definition='Mean across GloVe-matched types of the squared L2 embedding distance to the normalized pretrained vector.',
        selection='Lowest ordinary validation NLL; earliest tie. No qualitative outputs or test evaluation.',
        initialization='Matched input/output embedding vectors unitL2 normalized. Unmatched random normalstd.1; zeroPAD. Own GRU/projection random; outputbiaszero.',
        data_metadata=data['metadata'],source_sha256=source_hashes,pretrained=pretrained)
    digest=canonical_digest(config)
    if (directory/'config.json').exists() and json.loads((directory/'config.json').read_text())!=config:
        raise ValueError('Existing run differs; use a new output directory')
    write_json(directory/'config.json',config)
    if (directory/'metrics.json').exists():
        load_semantic_checkpoint(checkpoints/'best.pt',device);print('SKIP '+name,flush=True);return
    write_json(directory/'environment.json',dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,
        platform=platform.platform(),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,command=sys.argv,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=source_hashes,
        pretrained_disclosure='Uses external GloVe6B100d word vectors. Does not use external text-generation weights or retrieved jokes.'))
    random.seed(1704);np.random.seed(1704);torch.manual_seed(1704)
    if device.type=='cuda':torch.cuda.manual_seed_all(1704)
    model=SemanticGenerationGRU(**kwargs);model.initialize_prior(prior,matched);model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    generator=torch.Generator().manual_seed(1704)
    loaders={split:DataLoader(data[split],batch_size=64,shuffle=split=='train',
        generator=generator if split=='train' else None,collate_fn=collate,num_workers=0) for split in ('train','validation')}
    history,best_loss,best_epoch,seconds=[],float('inf'),0,0.
    last=checkpoints/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location=device,weights_only=True)
        if saved['config_hash']!=digest:raise ValueError('Resume configuration mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        history,best_loss,best_epoch,seconds=saved['history'],saved['best_loss'],saved['best_epoch'],saved['seconds']
        torch.set_rng_state(saved['rng'].cpu());generator.set_state(saved['loader_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    print('START '+name+' parameters='+str(model.parameter_count()),flush=True)
    for epoch in range(len(history)+1,args.max_epochs+1):
        if len(history)-best_epoch>=15:break
        started=time.monotonic();train=train_epoch(model,loaders['train'],optimizer,device,anchor)
        validation=run_epoch(model,loaders['validation'],'generation',device);seconds+=time.monotonic()-started
        row=dict(epoch=epoch,train_loss=train['loss'],train_objective=train['objective'],train_accuracy=train['accuracy'],
            validation_loss=validation['loss'],validation_accuracy=validation['accuracy'],anchor_penalty=float(model.anchor_penalty().detach()),
            seconds=time.monotonic()-started)
        history.append(row)
        common=dict(model=model.state_dict(),vocab=data['vocab'].to_dict(),epoch=epoch,config_hash=digest)
        if validation['loss']<best_loss:
            best_loss,best_epoch=validation['loss'],epoch;save_checkpoint(checkpoints/'best.pt',common)
        write_json(directory/'history.json',history)
        save_checkpoint(last,dict(**common,optimizer=optimizer.state_dict(),history=history,best_loss=best_loss,best_epoch=best_epoch,
            seconds=seconds,rng=torch.get_rng_state(),loader_rng=generator.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
        print(json.dumps(dict(run=name,**row,best_epoch=best_epoch)),flush=True)
    selected,_,provenance=load_semantic_checkpoint(checkpoints/'best.pt',device)
    replay=run_epoch(selected,loaders['validation'],'generation',device)
    if abs(replay['loss']-best_loss)>1e-6:raise RuntimeError('Selected validation replay mismatch')
    result=dict(name=name,parameters=model.parameter_count(),best_epoch=best_epoch,epochs_completed=len(history),validation=replay,
        last_validation=history[-1]['validation_loss'],train_seconds=seconds,checkpoint_sha256=provenance['checkpoint_sha256'],
        config_hash=digest,test_evaluated=False,external_word_embeddings=True,matched_types=int(matched.sum()),
        matched_type_fraction=pretrained['matched_type_fraction'])
    write_json(directory/'metrics.json',result);print('COMPLETE '+json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/stage_4_data')
    parser.add_argument('--output',type=Path,default=ROOT/'result/stage_4_result/generation_alternatives/semantic_runs')
    parser.add_argument('--vector-file',type=Path,default=ROOT/'data/stage_4_data/external_embeddings/glove.6B.100d.txt')
    parser.add_argument('--manifest-file',type=Path,default=ROOT/'data/stage_4_data/external_embeddings/manifest.json')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--max-epochs',type=int,default=120)
    parser.add_argument('--experiments',nargs='+',choices=list(EXPERIMENTS),default=list(EXPERIMENTS))
    args=parser.parse_args()
    if args.max_epochs<1:parser.error('--max-epochs must be positive')
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    data=Dataset_Loader(args.data_dir,seed=1704).load_generation();del data['test']
    prior,matched,pretrained=load_word_prior(data['vocab'],args.vector_file,args.manifest_file)
    pretrained.update(vector_file=str(args.vector_file.resolve().relative_to(ROOT)),manifest_file=str(args.manifest_file.resolve().relative_to(ROOT)))
    ordinary=[token for row in data['train'] for token in row['tokens'] if token>=4]
    pretrained['matched_training_token_fraction']=sum(bool(matched[token]) for token in ordinary)/len(ordinary)
    for name in args.experiments:experiment(name,EXPERIMENTS[name],data,prior,matched,pretrained,args,torch.device(args.device))


if __name__=='__main__':main()
