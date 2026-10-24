import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy


def run_experiment(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.epochs<=0 or args.batch_size<=0 or args.learning_rate<=0:
        raise ValueError('epochs, batch size and learning rate must be positive')
    if not args.hidden_dims or any(n<=0 for n in args.hidden_dims):
        raise ValueError('Provide one or more positive hidden layer sizes')
    if not args.run_name or not all(c.isalnum() or c in '-_' for c in args.run_name):
        raise ValueError('run-name can only contain letters, digits, hyphens, underscores')

    save_path=PROJECT_ROOT / 'result' / 'stage_2_result' / args.run_name
    save_path.mkdir(parents=True, exist_ok=True)


    data_obj=Dataset_Loader('stage_2_digits', 'provided 28x28 digit vectors')
    data_obj.dataset_source_folder_path=PROJECT_ROOT / 'data' / 'stage_2_data'
    data_obj.train_limit=args.train_limit
    data_obj.test_limit=args.test_limit

    method_obj=Method_MLP('multi-layer perceptron', '',
                            hidden_dims=args.hidden_dims, max_epoch=args.epochs,
                            learning_rate=args.learning_rate, batch_size=args.batch_size,
                            optimizer_name=args.optimizer, loss_name=args.loss,
                            device=args.device)
    result_obj=Result_Saver('saver', '')
    result_obj.result_destination_folder_path=save_path
    result_obj.result_destination_file_name='predictions.pkl'
    setting_obj=Setting_Train_Test_Split('provided train/test partition', '')
    evaluate_obj=Evaluate_Accuracy('multiclass metrics', '')


    print('************ Stage 2 Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    scores, _=setting_obj.load_run_save_evaluate()
    print('************ Test Performance ************')
    for name, value in scores.items():
        print(f'{name}: {value:.4f}')


    history=method_obj.history
    with(save_path / 'training_history.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=('epoch', 'train_loss', 'train_accuracy'))
        writer.writeheader()
        writer.writerows(history)

    fig, axes=plt.subplots(1, 2, figsize=(11, 4))
    epochs=[item['epoch'] for item in history]
    axes[0].plot(epochs, [item['train_loss'] for item in history], marker='o')
    axes[0].set(xlabel='Epoch', ylabel='Training Loss', title='Learning Convergence: Loss')
    axes[1].plot(epochs, [item['train_accuracy'] for item in history], marker='o')
    axes[1].set(xlabel='Epoch', ylabel='Training Accuracy', title='Learning Convergence: Accuracy')
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path / 'learning_curve.png', dpi=160)
    plt.close(fig)

    with(save_path / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(scores, f, indent=2)
    with(save_path / 'classification_report.txt').open('w', encoding='utf-8') as f:
        f.write(evaluate_obj.detailed_report())
    np.savetxt(save_path / 'confusion_matrix.csv', evaluate_obj.confusion_matrix(),
               fmt='%d', delimiter=',')

    config=dict(run_name=args.run_name, epochs=args.epochs,
                  hidden_dims=list(args.hidden_dims), batch_size=args.batch_size,
                  optimizer=args.optimizer, loss=args.loss,
                  learning_rate=args.learning_rate, device=str(method_obj.device),
                  seed=args.seed, train_limit=args.train_limit, test_limit=args.test_limit)
    with(save_path / 'run_config.json').open('w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    torch.save(method_obj.state_dict(), save_path / 'mlp_weights.pt')
    print(f'Generated report files in: {save_path}')
    print('************ Stage 2 Finish ************')
    return scores


def parse_args():
    p=argparse.ArgumentParser(description='ECS 170 Stage 2 PyTorch MLP')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--hidden-dims', nargs='+', type=int, default=[128])
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    p.add_argument('--loss', choices=['cross_entropy', 'nll', 'mse'], default='cross_entropy')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    p.add_argument('--run-name', default='baseline')
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--train-limit', type=int, default=None,
                   help='Temporary debug subset; do NOT use for full reported scores.')
    p.add_argument('--test-limit', type=int, default=None,
                   help='Temporary debug subset; do NOT use for full reported scores.')
    return p.parse_args()


if __name__=='__main__':
    run_experiment(parse_args())
