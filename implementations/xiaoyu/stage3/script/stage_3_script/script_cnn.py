import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import torch
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split


def run_experiment(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    name=args.dataset.upper()
    epochs=args.epochs if args.epochs is not None else {'MNIST': 5, 'ORL': 20, 'CIFAR': 10}[name]
    batch_size=args.batch_size if args.batch_size is not None else (32 if name=='ORL' else 256)
    channels=args.channels or ([32, 64] if name=='CIFAR' else [16, 32])
    if epochs<=0 or batch_size<=0 or args.learning_rate<=0 or any(x<=0 for x in channels):
        raise ValueError('Training sizes and learning rate must be positive')
    if not args.run_name or any(not(c.isalnum() or c in '-_') for c in args.run_name):
        raise ValueError('Invalid run name')
    save_path=PROJECT_ROOT/'result/stage_3_result'/name.lower()/args.run_name
    save_path.mkdir(parents=True, exist_ok=True)
    data_obj=Dataset_Loader(name, PROJECT_ROOT/'data/stage_3_data')
    cls={'MNIST': Method_CNN_MNIST, 'ORL': Method_CNN_ORL, 'CIFAR': Method_CNN_CIFAR}[name]
    model=cls(channels=channels, kernel_size=args.kernel_size, pooling=args.pooling,
              hidden_dim=args.hidden_dim, max_epoch=epochs, learning_rate=args.learning_rate,
              batch_size=batch_size, device=args.device, seed=args.seed)
    evaluator=Evaluate_Accuracy(40 if name=='ORL' else 10)
    saver=Result_Saver('predictions', '')
    saver.result_destination_folder_path=save_path
    saver.result_destination_file_name='predictions.pkl'
    setting_obj=Setting_Train_Test_Split('provided split', '')
    setting_obj.prepare(data_obj, model, saver, evaluator)
    start=time.perf_counter()
    scores, _=setting_obj.load_run_save_evaluate()
    config=vars(args).copy()
    config.update(epochs=epochs, batch_size=batch_size, channels=channels, device=str(model.device),
                  torch_version=torch.__version__, numpy_version=np.__version__,
                  train_count=len(model.data['train']['y']), test_count=len(model.data['test']['y']),
                  seconds=time.perf_counter()-start, parameters=sum(p.numel() for p in model.parameters()))
    for file, data in [('metrics.json', scores), ('run_config.json', config)]:
        (save_path/file).write_text(json.dumps(data, indent=2), encoding='utf-8')
    with (save_path/'training_history.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(model.history[0]))
        writer.writeheader()
        writer.writerows(model.history)
    np.savetxt(save_path/'confusion_matrix.csv', evaluator.confusion_matrix(), fmt='%d', delimiter=',')
    np.savetxt(save_path/'classification_report.csv', evaluator.per_class, delimiter=',',
               header='precision,recall,f1,support', comments='')
    torch.save(model.state_dict(), save_path/'cnn_weights.pt')
    print(name, args.run_name, scores, flush=True)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes=plt.subplots(1, 2, figsize=(9, 3))
        for ax, key in zip(axes, ['train_loss', 'train_accuracy']):
            ax.plot([r['epoch'] for r in model.history], [r[key] for r in model.history])
            ax.set(xlabel='Epoch', ylabel=key, title=name)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path/'learning_curve.png', dpi=160)
        plt.close(fig)
    except ImportError:
        print('Install matplotlib to generate plots; training history is saved.', flush=True)
    return scores


def parse_args(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['mnist', 'orl', 'cifar'], default='mnist')
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--channels', type=int, nargs='+')
    p.add_argument('--kernel-size', type=int, choices=[3, 5], default=3)
    p.add_argument('--pooling', choices=['max', 'avg'], default='max')
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--run-name', default='baseline')
    return p.parse_args(argv)


if __name__=='__main__':
    run_experiment(parse_args())
