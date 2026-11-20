import argparse
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=['classification','generation','both'],default='both')
    p.add_argument('--classification-epochs',type=int,default=5)
    p.add_argument('--generation-epochs',type=int,default=15)
    p.add_argument('--glove',type=str,default='')
    args=p.parse_args()
    for mode in ('rnn','lstm','gru'):
        if args.task in ('classification','both'):
            command=[sys.executable,str(HERE/'script_classification.py'),'--model',mode,'--epochs',str(args.classification_epochs)]
            if args.glove:
                command+=['--glove',args.glove]
            subprocess.run(command,check=True)
        if args.task in ('generation','both'):
            command=[sys.executable,str(HERE/'script_generation.py'),'--model',mode,'--epochs',str(args.generation_epochs)]
            if args.glove:
                command+=['--glove',args.glove]
            subprocess.run(command,check=True)


if __name__=='__main__':
    main()
