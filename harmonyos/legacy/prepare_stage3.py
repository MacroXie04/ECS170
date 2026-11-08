from pathlib import Path
import argparse,zipfile,pickle,json,hashlib
import numpy as np
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('archive');a=p.parse_args()
 configs=[];datasets=[]
 with zipfile.ZipFile(a.archive) as z:
  for name in ['MNIST','ORL','CIFAR']:
   data=Dataset_Loader(a.archive,name).load()
   mapping=sorted({int(r['label']) for rows in data.values() for r in rows})
   manifest={'name':name,'classes':len(mapping),'original_labels':mapping,'splits':{}}
   for split,rows in data.items():
    x=np.stack([r['image'] for r in rows])
    if name=='ORL':
     assert np.array_equal(x[:,:,:,0],x[:,:,:,1]) and np.array_equal(x[:,:,:,0],x[:,:,:,2])
     # 2x2 area downsample of identical-channel grayscale, 112x92 -> 56x46.
     x=x[:,:,:,0].astype(np.float32).reshape(len(x),56,2,46,2).mean((2,4)).round().astype(np.uint8)[:,:,:,None]
    elif name=='MNIST':x=x.astype(np.uint8)[:,:,:,None]
    x=x.astype(np.uint8)
    y=np.array([mapping.index(int(r['label'])) for r in rows],np.uint8)
    for kind,arr in [('images',x),('labels',y)]:
     f=ROOT/'data'/f'{name}_{split}_{kind}.bin';arr.tofile(f)
    manifest['splits'][split]={'count':len(x),'shape':list(x.shape),'labels_per_class':np.bincount(y,minlength=len(mapping)).tolist()}
    if split=='train':
     sums=x.astype(np.float64).sum(axis=(0,1,2));sumsq=np.square(x.astype(np.float64)).sum(axis=(0,1,2));n=np.prod(x.shape[:3]);mean=sums/n/255;std=np.sqrt(sumsq/n-(sums/n)**2)/255
     manifest['mean']=mean.tolist();manifest['std']=std.tolist()
   h,w,c=manifest['splits']['train']['shape'][1:]
   for variant,channels,hidden in [('baseline',[8,16],64),('wide',[16,32],128),('deep',[16,32,64],128)]:
    configs.append({'id':f'{name.lower()}_{variant}','dataset':name,'variant':variant,'channels':channels,'hidden':hidden,'height':h,'width':w,'inputChannels':c,'classes':len(mapping),'batchSize':8 if name=='ORL' else 50,'epochs':{'MNIST':10,'ORL':60,'CIFAR':20}[name],'learningRate':{'MNIST':0.03,'ORL':0.01,'CIFAR':0.02}[name],'trainCount':len(data['train']),'testCount':len(data['test']),'mean':manifest['mean'],'std':manifest['std'],'seed':170,'augmentation':name=='CIFAR'})
   datasets.append(manifest)
   print(name,manifest['splits'],flush=True)
 (ROOT/'plan.json').write_text(json.dumps(configs,indent=2))
 (ROOT/'data/manifest.json').write_text(json.dumps({'archive_sha256':hashlib.sha256(Path(a.archive).read_bytes()).hexdigest(),'datasets':datasets,'protocol':'All official training examples every epoch; all test examples evaluated once after fixed final epoch. No host optimization. Three prespecified architectures per dataset.'},indent=2))
if __name__=='__main__':main()
