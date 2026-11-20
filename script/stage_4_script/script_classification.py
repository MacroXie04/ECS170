import argparse
import csv
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from local_code.stage_4_code.Glove_Embedding import get_embeddings
from local_code.stage_4_code.Evaluate_Stage4 import Evaluate_Stage4
from local_code.stage_4_code.Setting_Stage4 import Setting_Stage4
from local_code.stage_4_code.Result_Saver import Result_Saver


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['rnn','lstm','gru'],default='rnn')
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--hidden-size',type=int,default=128)
    parser.add_argument('--max-len',type=int,default=180)
    parser.add_argument('--vocab-size',type=int,default=20000)
    parser.add_argument('--learning-rate',type=float,default=0.001)
    parser.add_argument('--seed',type=int,default=2)
    parser.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto')
    parser.add_argument('--glove',type=str,default='')
    parser.add_argument('--train-limit',type=int)
    parser.add_argument('--test-limit',type=int)
    parser.add_argument('--run-name',type=str,default='')
    args=parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device
    if device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is not available')
    output=ROOT/'result'/'stage_4_result'/(args.run_name or 'classification_'+args.model)
    output.mkdir(parents=True,exist_ok=True)
    data_obj=Dataset_Loader('IMDb','sentiment classification',ROOT/'data'/'stage_4_data','classification',args.max_len,args.vocab_size,args.train_limit,args.test_limit)
    data=data_obj.load()
    embeddings=get_embeddings(data['vocab'],args.glove,cache=ROOT/'result'/'stage_4_result'/'embedding_cache') if args.glove else None
    model=Method_RNN_Classification(args.model,'',len(data['vocab']),args.model,50,args.hidden_size,args.epochs,args.batch_size,args.learning_rate,device,embeddings)
    save=Result_Saver('save','')
    save.result_destination_folder_path=output
    save.result_destination_file_name='predictions.pkl'
    evaluator=Evaluate_Stage4('classification metrics','')
    setting=Setting_Stage4('provided train/test','')
    setting.prepare(data_obj,model,save,evaluator)
    model.data=data
    metrics,_=setting.load_run_save_evaluate()
    print('Test metrics:',metrics)
    with (output/'metrics.json').open('w',encoding='utf-8') as f:
        json.dump(metrics,f,indent=2)
    with (output/'run_config.json').open('w',encoding='utf-8') as f:
        json.dump(vars(args),f,indent=2)
    with (output/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','train_loss','train_accuracy'])
        writer.writeheader()
        writer.writerows(model.history)
    (output/'classification_report.txt').write_text(evaluator.report(),encoding='utf-8')
    np.savetxt(output/'confusion_matrix.csv',evaluator.confusion(),delimiter=',',fmt='%d')
    torch.save({'weights':model.state_dict(),'vocab':data['vocab'],'config':vars(args)},output/'model_weights.pt')
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([s['epoch'] for s in model.history],[s['train_loss'] for s in model.history],marker='o')
    ax.set(xlabel='Epoch',ylabel='Training Loss',title='IMDb '+args.model.upper()+' Training Convergence')
    fig.tight_layout()
    fig.savefig(output/'learning_curve.png',dpi=150)
    plt.close(fig)
    print('Saved:',output)


if __name__=='__main__':
    main()
