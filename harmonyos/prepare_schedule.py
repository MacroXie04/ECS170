"""Reproduce PR6 CPU sampling/augmentation choices, with pixels processed on device."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
 r=next(r for r in json.loads((ROOT/'protocol.json').read_text())['runs'] if r['id']==a.run);c=r['config']
 n=json.loads((ROOT/'data/manifest.json').read_text())[c['dataset']]['optimization_samples']
 gen=torch.Generator().manual_seed(r['seed']);aug=torch.Generator().manual_seed(r['seed']+1)
 loader=DataLoader(torch.arange(n),batch_size=c['batch_size'],shuffle=True,generator=gen)
 out=ROOT/'data'/r['id'];out.mkdir(exist_ok=True)
 for epoch in range(1,c['max_epoch']+1):
  parts=[]
  for ids in loader:
   size=len(ids);shift=c['aug_shift']
   dy=torch.randint(0,2*shift+1,(size,),generator=aug)-shift if shift else torch.zeros(size,dtype=torch.int64)
   dx=torch.randint(0,2*shift+1,(size,),generator=aug)-shift if shift else torch.zeros(size,dtype=torch.int64)
   flip=(torch.rand(size,generator=aug)<0.5) if c['aug_hflip'] else torch.zeros(size,dtype=torch.bool)
   parts.append(torch.stack([ids,dy,dx,flip],dim=1).numpy().astype('<i4'))
  schedule=np.concatenate(parts);assert len(np.unique(schedule[:,0]))==n
  schedule.tofile(out/f'epoch_{epoch}.bin')
 print('PREPARED',r['id'],c['max_epoch'],'epochs; no optimizer steps')
if __name__=='__main__':main()
