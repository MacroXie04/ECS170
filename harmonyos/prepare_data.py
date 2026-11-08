"""Use the unchanged PR6 loader, split and normalization; never optimize on host."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from local_code.stage_3_code.Dataset_Loader import (Dataset_Loader,DATASET_SPECS,
    make_train_validation_split,compute_channel_stats,split_summary)
from local_code.stage_3_code.Experiment_Config import SPLIT_SEED

def main():
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);a=p.parse_args()
 out=ROOT/'data';out.mkdir(exist_ok=True);manifest={}
 for key,spec in DATASET_SPECS.items():
  loader=Dataset_Loader(key,folder=a.source);data=loader.load()
  x,y=data['train']['X'],data['train']['y']
  ti,vi=make_train_validation_split(y,spec.validation_size,SPLIT_SEED)
  mean,std=compute_channel_stats(x[ti]);meta=split_summary(y,ti,vi,spec,SPLIT_SEED)
  meta.update(source_sha256=loader.verify_checksum(),mean=mean,std=std,input_shape=spec.input_shape,files={})
  for split,xx,yy in [('train',x[ti],y[ti]),('val',x[vi],y[vi]),('test',data['test']['X'],data['test']['y'])]:
   for kind,arr in [('images',xx.transpose(0,2,3,1)),('labels',yy.astype(np.uint8))]:
    dest=out/f'{key}_{split}_{kind}.bin';np.ascontiguousarray(arr).tofile(dest)
    meta['files'][dest.name]={'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
  np.save(out/f'{key}_train_indices.npy',ti);np.save(out/f'{key}_val_indices.npy',vi)
  manifest[key]=meta;print(key,len(ti),len(vi),spec.test_size,flush=True)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
