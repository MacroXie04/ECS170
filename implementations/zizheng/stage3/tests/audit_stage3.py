"""Read-only verification of recorded Stage 3 splits, metrics and final checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Evaluate import Evaluate
from local_code.stage_3_code.Setting import MODELS, code_hash

ROOT = Path(__file__).resolve().parents[1]


def audit(output, data_folder, datasets=('MNIST', 'ORL', 'CIFAR')):
    for dataset in datasets:
        folder = output / dataset
        selection = json.loads((folder/'selection.json').read_text())
        loader = Dataset_Loader(data_folder,dataset)
        train_data = loader.load()
        reference = None
        records = []
        for name in selection['candidates'] + ['final']:
            run = folder/name
            record = json.loads((run/'record.json').read_text())
            assert record['source_sha256'] == code_hash(), (dataset,name,'source changed')
            with np.load(run/'predictions.npz') as saved:
                evaluator = Evaluate(record['classes'])
                for split in (['train','test'] if name=='final' else ['train','validation']):
                    assert evaluator.score(saved[split+'_true'], saved[split+'_pred']) == record[split]
            history = list(csv.DictReader((run/'history.csv').open()))
            assert [int(r['epoch']) for r in history] == list(range(1,len(history)+1))
            assert 1 <= record['best_epoch'] <= len(history)
            if name != 'final':
                with np.load(run/'split.npz') as split:
                    train, val = split['train'], split['validation']
                assert not np.intersect1d(train,val).size
                np.testing.assert_array_equal(np.sort(np.r_[train,val]),np.arange(record['train_source']['rows']))
                if reference is None:
                    reference = train,val
                else:
                    np.testing.assert_array_equal(train,reference[0])
                    np.testing.assert_array_equal(val,reference[1])
                checkpoint = torch.load(run/'model.pt',map_location='cpu',weights_only=True)
                model = MODELS[dataset](num_classes=len(checkpoint['classes']),config=checkpoint['config'])
                model.load_state_dict(checkpoint['state_dict'])
                with np.load(run/'predictions.npz') as saved:
                    np.testing.assert_array_equal(saved['train_true'],train_data['y'][train].numpy())
                    np.testing.assert_array_equal(saved['validation_true'],train_data['y'][val].numpy())
                    np.testing.assert_array_equal(model.test(train_data['X'][val]),saved['validation_pred'])
                row = history[record['best_epoch']-1]
                assert abs(float(row['val_accuracy'])-record['validation']['accuracy']) < 1e-7
                assert float(row['val_loss']) <= min(float(r['val_loss']) for r in history)+1e-6
                records.append(record)
        best = sorted(records,key=lambda r:(-r['validation']['accuracy'],-r['validation']['macro']['f1'],r['name']))[0]
        assert selection['selected'] == best['name']
        assert selection['config'] == dict(best['config'],epochs=best['best_epoch'])
        assert json.loads((folder/'final/locked_selection.json').read_text()) == selection
        checkpoint = torch.load(folder/'final/model.pt',map_location='cpu',weights_only=True)
        model = MODELS[dataset](num_classes=len(checkpoint['classes']),config=checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        loader = Dataset_Loader(data_folder,dataset)
        train = loader.load()
        test = loader.load('test',np.array(checkpoint['classes']))
        final_record = json.loads((folder/'final/record.json').read_text())
        assert train['source'] == final_record['train_source']
        assert test['source'] == final_record['test_source']
        np.testing.assert_array_equal(train['classes'],checkpoint['classes'])
        with np.load(folder/'final/predictions.npz') as saved:
            np.testing.assert_array_equal(test['y'].numpy(),saved['test_true'])
            np.testing.assert_array_equal(model.test(test['X']),saved['test_pred'])
        print(dataset, 'PASS: five comparisons, fixed split, selection, metrics and checkpoint')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'result/stage_3_result')
    parser.add_argument('--data',type=Path,default=ROOT/'data/stage_3_data')
    parser.add_argument('--dataset',choices=['MNIST','ORL','CIFAR','all'],default='all')
    args=parser.parse_args()
    torch.set_num_threads(2)
    audit(args.output,args.data,['MNIST','ORL','CIFAR'] if args.dataset=='all' else [args.dataset])
