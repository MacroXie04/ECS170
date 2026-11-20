import hashlib
import zipfile
from pathlib import Path
import numpy as np
import torch


def get_embeddings(vocab,filename,dim=50,cache=None):
    filename=Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f'GloVe file not found: {filename}')
    sig=hashlib.sha256('\n'.join(vocab).encode()).hexdigest()[:12]
    cached=Path(cache)/f'glove_{sig}.npy' if cache else None
    if cached and cached.exists():
        return torch.from_numpy(np.load(cached).copy())
    table=np.random.default_rng(2).normal(0,0.05,(len(vocab),dim)).astype(np.float32)
    table[0]=0
    wanted={w.encode('utf-8'):i for w,i in vocab.items() if i>2}
    found=0
    with zipfile.ZipFile(filename) as z:
        with z.open(z.namelist()[0]) as f:
            for line in f:
                key,sep,values=line.partition(b' ')
                i=wanted.get(key)
                if i is None:
                    continue
                vec=np.fromstring(values.decode('utf-8'),sep=' ',dtype=np.float32)
                if len(vec)==dim:
                    table[i]=vec
                    found+=1
    print(f'GloVe found: {found}/{len(vocab)-3} words')
    if cached:
        cached.parent.mkdir(parents=True,exist_ok=True)
        np.save(cached,table)
    return torch.from_numpy(table)
