import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from torch.utils.data import DataLoader,TensorDataset

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_GRU_Improved import Method_GRU_Improved
from script.stage_4_script.script_gru_improved import encode


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-name',default='gru_conv_pool_glove')
    parser.add_argument('--batch-size',type=int,default=128)
    args=parser.parse_args()
    torch.set_num_threads(4)
    path=ROOT/'result'/'stage_4_result'/args.run_name/'model_weights.pt'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    config=saved['config']
    vocab=saved['vocab']
    loader=Dataset_Loader('IMDb','test',ROOT/'data'/'stage_4_data','classification',config['max_len'])
    texts,y=loader._reviews('test',None)
    x,n=encode(texts,vocab,config['max_len'])
    data=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(n)),batch_size=args.batch_size)
    model=Method_GRU_Improved('gru','',len(vocab),config['hidden_size'],dropout=config['dropout'])
    model.load_state_dict(saved['weights'])
    model.eval()
    preds=[]
    with torch.no_grad():
        for x,n in data:
            preds.extend(model(x,n).argmax(1).tolist())
    p,r,f,_=precision_recall_fscore_support(y,preds,average='macro',zero_division=0)
    print('Test samples:',len(y))
    print(f'Accuracy: {accuracy_score(y,preds)*100:.2f}%')
    print(f'Macro Precision: {p*100:.2f}%')
    print(f'Macro Recall: {r*100:.2f}%')
    print(f'Macro F1: {f*100:.2f}%')

if __name__=='__main__':
    main()
