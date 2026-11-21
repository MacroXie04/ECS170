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
from local_code.stage_4_code.xiaoyu.Dataset_Loader import Dataset_Loader,tokenize
from local_code.stage_4_code.xiaoyu.Method_RNN_Generation import Method_RNN_Generation
from local_code.stage_4_code.xiaoyu.Glove_Embedding import get_embeddings
from local_code.stage_4_code.xiaoyu.Evaluate_Stage4 import Evaluate_Stage4
from local_code.stage_4_code.xiaoyu.Result_Saver import Result_Saver
from local_code.stage_4_code.xiaoyu.Setting_Stage4 import Setting_Stage4


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['rnn','lstm','gru'],default='rnn')
    parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--hidden-size',type=int,default=128)
    parser.add_argument('--max-len',type=int,default=70)
    parser.add_argument('--vocab-size',type=int,default=10000)
    parser.add_argument('--learning-rate',type=float,default=0.001)
    parser.add_argument('--seed',type=int,default=2)
    parser.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto')
    parser.add_argument('--glove',type=str,default='')
    parser.add_argument('--start-words',type=str,default='what did the')
    parser.add_argument('--temperature',type=float,default=0.8)
    parser.add_argument('--run-name',type=str,default='')
    args=parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device
    if device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is not available')
    output=ROOT/'result'/'stage_4_result'/(args.run_name or 'generation_'+args.model)
    output.mkdir(parents=True,exist_ok=True)
    loader=Dataset_Loader('jokes','text generation',ROOT/'data'/'stage_4_data','generation',args.max_len,args.vocab_size)
    data=loader.load()
    embeddings=get_embeddings(data['vocab'],args.glove,cache=ROOT/'result'/'stage_4_result'/'embedding_cache') if args.glove else None
    model=Method_RNN_Generation(args.model,'',data['vocab'],args.model,50,args.hidden_size,args.epochs,args.batch_size,args.learning_rate,device,embeddings)
    saver=Result_Saver('save','')
    saver.result_destination_folder_path=output
    saver.result_destination_file_name='training_result.pkl'
    evaluator=Evaluate_Stage4('generation loss','')
    setting=Setting_Stage4('text generation','')
    setting.prepare(loader,model,saver,evaluator)
    model.data=data
    metrics,_=setting.load_run_save_evaluate()
    print('Final loss:',metrics['train_loss'])
    prefixes=[args.start_words,'why did the','how many people']
    rows=[]
    for words in prefixes:
        try:
            generated=model.generate(words,40,args.temperature)
        except ValueError:
            continue
        target=next((j for j in data['jokes'] if tokenize(j)[:3]==tokenize(words)),None)
        rows.append({'starting_words':words,'training_example':target or '(not found)','generated':generated})
        print('START:',words,'\nGENERATED:',generated,'\nTRAINING EXAMPLE:',target or '(none)',flush=True)
    with (output/'generated_examples.txt').open('w',encoding='utf-8') as f:
        for row in rows:
            f.write('Input: '+row['starting_words']+'\nTraining example: '+row['training_example']+'\nGenerated: '+row['generated']+'\n\n')
    with (output/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','train_loss'])
        writer.writeheader()
        writer.writerows(model.history)
    with (output/'run_config.json').open('w',encoding='utf-8') as f:
        json.dump(vars(args),f,indent=2)
    torch.save({'weights':model.state_dict(),'vocab':data['vocab'],'config':vars(args)},output/'model_weights.pt')
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([s['epoch'] for s in model.history],[s['train_loss'] for s in model.history],marker='o')
    ax.set(xlabel='Epoch',ylabel='Training Loss',title='Joke '+args.model.upper()+' Training Convergence')
    fig.tight_layout()
    fig.savefig(output/'learning_curve.png',dpi=150)
    plt.close(fig)
    print('Saved:',output)


if __name__=='__main__':
    main()
