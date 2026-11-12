import csv
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


PROJECT_ROOT=Path(__file__).resolve().parents[2]


if __name__=='__main__':
    count=0
    for path in (PROJECT_ROOT/'result/stage_3_result').glob('*/*/metrics.json'):
        metrics=json.loads(path.read_text())
        config=json.loads((path.parent/'run_config.json').read_text())
        with (path.parent/'predictions.pkl').open('rb') as f:
            data=pickle.load(f)
        p, r, f1, _=precision_recall_fscore_support(data['true_y'], data['pred_y'], average='macro', zero_division=0)
        expected=[accuracy_score(data['true_y'], data['pred_y']), p, r, f1]
        actual=[metrics[k] for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']]
        assert np.allclose(actual, expected), path
        assert len(data['true_y'])==config['test_count']
        matrix=np.loadtxt(path.parent/'confusion_matrix.csv', delimiter=',')
        classes=40 if config['dataset']=='orl' else 10
        assert matrix.shape==(classes, classes)
        assert matrix.sum()==config['test_count']
        with (path.parent/'training_history.csv').open() as f:
            assert len(list(csv.DictReader(f)))==config['epochs']
        print(path.parent.relative_to(PROJECT_ROOT), 'OK')
        count+=1
    assert count>=9, 'Expected all nine experiments'
