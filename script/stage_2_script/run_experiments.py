"""Run four prespecified 15-epoch comparisons; test evaluated only at the end.
Usage: python script/stage_2_script/run_experiments.py --device cpu
"""
from pathlib import Path
import argparse, hashlib, json, random, sys, platform
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Setting_Prepartitioned import Setting_Prepartitioned

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--device',default='cpu',choices=['cpu','mps','cuda'])
    args=parser.parse_args()
    torch.set_num_threads(4)
    out=ROOT/'results';out.mkdir(exist_ok=True)
    data=Dataset_Loader(ROOT/'data').load()
    meta={'python':platform.python_version(),'pytorch':torch.__version__,'device':args.device,
          'train_shape':list(data['train']['X'].shape),'test_shape':list(data['test']['X'].shape),
          'class_counts':{k:np.bincount(v['y'],minlength=10).tolist() for k,v in data.items()},
          'data_sha256':{n:hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest() for n in ['train.csv','test.csv']}}
    configs=[dict(name='shallow_adam_ce',hidden=[128],optimizer='Adam',lr=.001,loss='CE'),
             dict(name='deep_adam_ce',hidden=[256,128],optimizer='Adam',lr=.001,loss='CE'),
             dict(name='deep_sgd_ce',hidden=[256,128],optimizer='SGD',lr=.05,loss='CE'),
             dict(name='deep_adam_mse',hidden=[256,128],optimizer='Adam',lr=.001,loss='MSE')]
    results=[]
    for cfg in configs:
        cfg.update(seed=42,epochs=15,batch_size=256)
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
        model=Method_MLP(cfg,args.device)
        result=Setting_Prepartitioned(data,model).load_run_save_evaluate()
        np.savez_compressed(out/f"{cfg['name']}_predictions.npz",true_y=result.pop('true_y'),pred_y=result.pop('pred_y'))
        torch.save({'state_dict':model.state_dict(),'config':cfg},out/f"{cfg['name']}.pt")
        result.update(config=cfg,parameters=sum(p.numel() for p in model.parameters()))
        results.append(result)
        print(json.dumps(result['metrics']),flush=True)
        (out/'experiments.json').write_text(json.dumps({'environment':meta,'experiments':results},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(9,3.2),layout='constrained')
    for r in results:
        ax=axs[0 if r['config']['loss']=='CE' else 1]
        ax.plot([h['epoch'] for h in r['history']],[h['train_loss'] for h in r['history']],label=r['config']['name'],lw=1.8)
    for ax,title in zip(axs,['Cross-entropy objective','MSE objective (different scale)']):
        ax.set(xlabel='Epoch',ylabel='Mean training loss',title=title); ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'convergence.png',dpi=220);plt.close(fig)
    reference=results[1]
    fig,ax=plt.subplots(figsize=(4.6,4),layout='constrained')
    cm=np.array(reference['metrics']['confusion_matrix']); im=ax.imshow(cm,cmap='Blues')
    ax.set(xticks=range(10),yticks=range(10),xlabel='Predicted label',ylabel='True label',title='Deep Adam + CE: test confusion matrix')
    for i in range(10):
        for j in range(10):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=6,color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.savefig(out/'confusion.png',dpi=220);plt.close(fig)
if __name__=='__main__': main()
