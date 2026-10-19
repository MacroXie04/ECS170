"""Audit saved real experiments; this does not train or select another model."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Evaluate import Evaluate
from local_code.stage_2_code.Setting import code_hash


def main():
    torch.set_num_threads(2)
    output = ROOT / 'result/stage_2_result'
    selection = json.loads((output / 'selection.json').read_text())
    reference = None
    for name in selection['candidates']:
        folder = output / name
        record = json.loads((folder / 'record.json').read_text())
        assert record['source_sha256'] == code_hash(), name
        split = np.load(folder / 'split.npz')
        train, val = split['train'], split['validation']
        assert len(train) == 54000 and len(val) == 6000
        assert not np.intersect1d(train, val).size
        np.testing.assert_array_equal(np.sort(np.concatenate([train, val])), np.arange(60000))
        if reference is None:
            reference = train, val
        else:
            np.testing.assert_array_equal(train, reference[0])
            np.testing.assert_array_equal(val, reference[1])
        pred = np.load(folder / 'predictions.npz')
        for label in ['train', 'validation']:
            actual = Evaluate(record['classes']).score(pred[f'{label}_true'], pred[f'{label}_pred'])
            assert actual == record[label], (name, label)
        rows = list(csv.DictReader((folder / 'history.csv').open()))
        best_row = min(rows, key=lambda row: float(row['val_loss']))
        assert int(best_row['epoch']) == record['best_epoch']
        assert len(rows) == record['completed_epochs']
    folder = output / 'final'
    final = json.loads((folder / 'record.json').read_text())
    assert final['config'] == selection['config']
    assert final['source_sha256'] == code_hash()
    checkpoint = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
    model = Method_MLP(checkpoint['input_dim'], len(checkpoint['classes']), checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    test = Dataset_Loader(ROOT / 'data/stage_2_data').load_file('test.csv', checkpoint['classes'])
    pred = np.load(folder / 'predictions.npz')
    np.testing.assert_array_equal(pred['test_true'], test['y'])
    np.testing.assert_array_equal(model.test(test['X']), pred['test_pred'])
    assert Evaluate(final['classes']).score(pred['test_true'], pred['test_pred']) == final['test']
    audit = {'status': 'passed', 'validation_runs': len(selection['candidates']),
             'same_disjoint_complete_split': True, 'saved_metrics_recomputed': True,
             'best_epoch_matches_history': True, 'source_hash_matches': True,
             'final_checkpoint_reproduces_predictions': True,
             'test_correct': int((pred['test_true'] == pred['test_pred']).sum()),
             'test_total': len(test['y'])}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
