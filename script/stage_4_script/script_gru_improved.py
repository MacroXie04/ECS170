import argparse
import hashlib
import csv
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Glove_Embedding import get_embeddings
from local_code.stage_4_code.Method_GRU_Improved import Method_GRU_Improved


def encode(texts,vocab,max_len):
    x=np.zeros((len(texts),max_len),dtype=np.int64)
    lengths=np.zeros(len(texts),dtype=np.int64)
    half=max_len//2
    for i,s in enumerate(texts):
        words=s[:half]+s[-(max_len-half):] if len(s)>max_len else s
        ids=[vocab.get(w,1) for w in words]
        if not ids:
            ids=[1]
        lengths[i]=len(ids)
        x[i,:len(ids)]=ids
    return x,lengths


def make_loader(x,y,n,batch,shuffle=False,seed=2):
    data=TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(n))
    generator=torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(data,batch_size=batch,shuffle=shuffle,generator=generator,num_workers=0)


def run_epoch(model,loader,criterion,optimizer=None):
    model.train(optimizer is not None)
    total=0
    correct=0
    loss_sum=0.0
    for x,y,n in loader:
        with torch.set_grad_enabled(optimizer is not None):
            logits=model(x,n)
            loss=criterion(logits,y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
        count=len(y)
        total+=count
        correct+=(logits.argmax(1)==y).sum().item()
        loss_sum+=loss.item()*count
    return loss_sum/total,correct/total


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--hidden-size',type=int,default=96)
    parser.add_argument('--max-len',type=int,default=200)
    parser.add_argument('--learning-rate',type=float,default=0.0015)
    parser.add_argument('--dropout',type=float,default=0.3)
    parser.add_argument('--seed',type=int,default=2)
    parser.add_argument('--patience',type=int,default=3)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--glove',type=str,default='')
    parser.add_argument('--run-name',default='classification_gru_improved')
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output=ROOT/'result'/'stage_4_result'/args.run_name
    output.mkdir(parents=True,exist_ok=True)

    loader=Dataset_Loader('IMDb','sentiment classification',ROOT/'data'/'stage_4_data','classification',args.max_len,20000)
    texts,y=loader._reviews('train',None)
    test,yt=loader._reviews('test',None)
    idx=np.arange(len(y))
    train_idx,val_idx=train_test_split(idx,test_size=0.1,random_state=args.seed,stratify=y)
    vocab=loader._vocab([texts[i] for i in train_idx])
    x,n=encode(texts,vocab,args.max_len)
    xt,nt=encode(test,vocab,args.max_len)
    train_loader=make_loader(x[train_idx],y[train_idx],n[train_idx],args.batch_size,True,args.seed)
    val_loader=make_loader(x[val_idx],y[val_idx],n[val_idx],args.batch_size)
    test_loader=make_loader(xt,yt,nt,args.batch_size)
    print(f'IMDb train={len(train_idx)} val={len(val_idx)} test={len(yt)} vocab={len(vocab)}',flush=True)

    cache=ROOT/'result'/'stage_4_result'/'embedding_cache'
    sig=hashlib.sha256('\n'.join(vocab).encode()).hexdigest()[:12]
    cached=cache/f'glove_{sig}.npy'
    if args.glove:
        embeddings=get_embeddings(vocab,args.glove,cache=cache)
    elif cached.exists():
        embeddings=torch.from_numpy(np.load(cached).copy())
        print('Loaded local GloVe embedding cache',flush=True)
    else:
        embeddings=None
        print('No GloVe vectors: using random embedding initialization',flush=True)
    model=Method_GRU_Improved('gru_improved','',len(vocab),args.hidden_size,embeddings,args.dropout)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    criterion=nn.CrossEntropyLoss()
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=1)
    best_loss=float('inf')
    best_epoch=0
    best_weights=None
    stale=0
    history=[]
    for epoch in range(1,args.epochs+1):
        start=time.time()
        tr_loss,tr_acc=run_epoch(model,train_loader,criterion,optimizer)
        val_loss,val_acc=run_epoch(model,val_loader,criterion)
        history.append({'epoch':epoch,'train_loss':tr_loss,'train_accuracy':tr_acc,'val_loss':val_loss,'val_accuracy':val_acc,'learning_rate':optimizer.param_groups[0]['lr']})
        print(f'Epoch {epoch}/{args.epochs} | train loss {tr_loss:.4f}, acc {tr_acc*100:.2f}% | val loss {val_loss:.4f}, acc {val_acc*100:.2f}% | {time.time()-start:.1f}s',flush=True)
        if val_loss<best_loss-1e-4:
            best_loss=val_loss
            best_epoch=epoch
            best_weights=deepcopy(model.state_dict())
            stale=0
        else:
            stale+=1
        scheduler.step(val_loss)
        if stale>=args.patience:
            print('Early stopping at',epoch,flush=True)
            break

    model.load_state_dict(best_weights)
    model.eval()
    preds=[]
    with torch.no_grad():
        for x,y,n in test_loader:
            preds.extend(model(x,n).argmax(1).tolist())
    p,r,f,_=precision_recall_fscore_support(yt,preds,average='macro',zero_division=0)
    metrics={'accuracy':float(accuracy_score(yt,preds)),'precision_macro':float(p),'recall_macro':float(r),'f1_macro':float(f),'best_epoch':best_epoch,'best_val_loss':best_loss,'train_size':len(train_idx),'validation_size':len(val_idx),'test_size':len(yt)}
    print('TEST METRICS',metrics,flush=True)
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (output/'run_config.json').write_text(json.dumps({**vars(args),'architecture':'bidirectional GRU + masked mean,max,final pooling','split_seed':args.seed,'gradient_clip_norm':1.0,'use_glove':bool(args.glove)},indent=2),encoding='utf-8')
    with (output/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (output/'classification_report.txt').write_text(classification_report(yt,preds,digits=4),encoding='utf-8')
    np.savetxt(output/'confusion_matrix.csv',confusion_matrix(yt,preds),delimiter=',',fmt='%d')
    np.savez(output/'predictions.npz',y_true=yt,y_pred=np.asarray(preds))
    np.savez(output/'split_indices.npz',train_idx=train_idx,val_idx=val_idx)
    torch.save({'weights':best_weights,'vocab':vocab,'config':vars(args)},output/'model_weights.pt')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for k in ('train','val'):
        axes[0].plot([h['epoch'] for h in history],[h[k+'_loss'] for h in history],label=k)
        axes[1].plot([h['epoch'] for h in history],[h[k+'_accuracy'] for h in history],label=k)
    axes[0].set(xlabel='Epoch',ylabel='Cross Entropy',title='Training and validation loss')
    axes[1].set(xlabel='Epoch',ylabel='Accuracy',title='Training and validation accuracy')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output/'learning_curve.png',dpi=160)
    plt.close(fig)

if __name__=='__main__':
    main()
