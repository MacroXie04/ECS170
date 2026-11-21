#!/usr/bin/env python3
"""Train a compatibility reranker on training joke pairs; no test evaluation."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Experiment_Runner import write_json,save_checkpoint
from local_code.stage_4_code.Coherence_Ranker import PairRanker,make_pairs,digest,canonical


def main():
    torch.set_num_threads(4); torch.manual_seed(1704)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    base=ROOT/'result/stage_4_result/generation_alternatives/coherence_ranker'
    if (base/'metrics.json').exists(): raise FileExistsError('Ranker run is immutable')
    (base/'checkpoints').mkdir(parents=True,exist_ok=True)
    data=Dataset_Loader(ROOT/'data/stage_4_data',seed=1704).load_generation();del data['test']
    vocab=data['vocab'];matrix=torch.zeros(len(vocab),100);matched=[]
    external=ROOT/'data/stage_4_data/external_embeddings'
    manifest=json.loads((external/'manifest.json').read_text())
    vectors=external/manifest['member']
    if digest(vectors)!=manifest['member_sha256']:raise ValueError('External vectors changed')
    with vectors.open() as handle:
        for line in handle:
            token,_,values=line.partition(' ')
            if token in vocab.stoi and vocab.stoi[token]>=4:
                row=torch.tensor(np.fromstring(values,sep=' ',dtype=np.float32))
                if row.shape!=(100,) or not torch.isfinite(row).all():raise ValueError('Invalid external vector')
                index=vocab.stoi[token];matrix[index]=torch.nn.functional.normalize(row,dim=0);matched.append(index)
    pairs={split:make_pairs(data[split],1704 if split=='train' else 1705) for split in ('train','validation')}
    sources=['local_code/stage_4_code/Coherence_Ranker.py','script/stage_4_script/train_coherence_ranker.py','local_code/stage_4_code/Dataset_Loader.py']
    config=dict(architecture='frozen_glove_pair_mlp_v1',seed=1704,dimensions=100,hidden=64,dropout=.2,
        learning_rate=.001,weight_decay=.01,max_epochs=80,patience=10,selection='Lowest validation BCE on fixed balanced positive/shuffled-negative pairs',
        inference_gate='Use reranker only if held-out pair discrimination AUC>=0.65; this is not validated humor or semantic correctness.',
        data_metadata=data['metadata'],external_embeddings=manifest,matched_vocabulary_count=len(matched),
        pairs_sha256={split:canonical(rows) for split,rows in pairs.items()},
        source_sha256={name:digest(ROOT/name) for name in sources})
    write_json(base/'config.json',config);write_json(base/'pairs.json',pairs)
    model=PairRanker(matrix,vocab.tokens).cuda()
    features={split:model.features(rows) for split,rows in pairs.items()}
    labels={split:torch.tensor([row['label'] for row in rows],dtype=torch.float32,device='cuda') for split,rows in pairs.items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    history=[];best=float('inf');best_epoch=0
    generator=torch.Generator().manual_seed(1704)
    # Negatives are twice positives; equal class weights remove this prior imbalance.
    def loss(pred,target):
        weights=torch.where(target==1,2.,1.)
        return (torch.nn.functional.binary_cross_entropy_with_logits(pred,target,reduction='none')*weights).sum()/weights.sum()
    for epoch in range(1,81):
        model.train();order=torch.randperm(len(pairs['train']),generator=generator).tolist()
        for start in range(0,len(order),128):
            index=order[start:start+128];objective=loss(model(features['train'][index]),labels['train'][index])
            optimizer.zero_grad(set_to_none=True);objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions=model(features['validation']);val_loss=float(loss(predictions,labels['validation']))
            auc=float(roc_auc_score(labels['validation'].cpu(),predictions.cpu()))
        history.append(dict(epoch=epoch,validation_loss=val_loss,validation_auc=auc))
        if val_loss<best:
            best=val_loss;best_epoch=epoch
            save_checkpoint(base/'checkpoints/best.pt',dict(model=model.state_dict(),config_hash=canonical(config),vocab=vocab.to_dict(),epoch=epoch,validation_auc=auc))
        write_json(base/'history.json',history)
        if epoch-best_epoch>=10:break
    selected=history[best_epoch-1]
    write_json(base/'metrics.json',dict(**selected,epochs_completed=len(history),pair_counts={split:len(rows) for split,rows in pairs.items()},
        enabled=selected['validation_auc']>=.65,checkpoint_sha256=digest(base/'checkpoints/best.pt'),
        limitation='Synthetic shuffled negatives measure topic compatibility; they do not establish generated-joke coherence. No test data used.'))
    print(json.dumps(selected),flush=True)


if __name__=='__main__':main()
