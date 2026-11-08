"""First augmented batch from unchanged PR6 preprocessing, for a device parity check."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent))
from local_code.stage_3_code.Dataset_Loader import make_train_loader
from local_code.stage_3_code.Experiment_Config import CNNConfig
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
r=next(r for r in json.loads((ROOT/'protocol.json').read_text())['runs'] if r['id']==a.run);c=CNNConfig.from_dict(r['config'])
m=json.loads((ROOT/'data/manifest.json').read_text())[c.dataset];ch,h,w=m['input_shape'];n=m['optimization_samples']
x=np.fromfile(ROOT/f'data/{c.dataset}_train_images.bin',np.uint8).reshape(n,h,w,ch).transpose(0,3,1,2).copy();y=np.fromfile(ROOT/f'data/{c.dataset}_train_labels.bin',np.uint8).astype(np.int64)
loader=make_train_loader(x,y,m['mean'],m['std'],c,torch.Generator().manual_seed(r['seed']),torch.Generator().manual_seed(r['seed']+1))
images,labels=next(iter(loader));images.numpy().tofile(ROOT/'data'/r['id']/'expected_batch.bin')
print('REFERENCE',r['id'],list(images.shape))
