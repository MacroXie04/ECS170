'''
Stage 4 (4-5): can more regularisation close the train/validation gap?  (set GLOVE=/path/glove.6B.50d.txt to add GloVe rows)

    python script_text_regularization.py

Every row is trained from scratch with the same seed. The best epoch AND the best configuration are chosen by validation
accuracy; test accuracy is only reported (never used to choose). -> result/stage_4_result/text_regularization.csv
'''
import csv
from stage4_common import *

BI = {'bidirectional': True}
experiments = [
    ('GRU (baseline, dropout 0.3)',                   {}, {}),
    ('GRU dropout 0.5',                               {}, {'dropout': 0.5}),
    ('GRU dropout 0.5 + embedding dropout 0.3',       {}, {'dropout': 0.5, 'embedding_dropout': 0.3}),
    ('GRU dropout 0.5 + emb. dropout 0.3 + L2 1e-5',  {}, {'dropout': 0.5, 'embedding_dropout': 0.3, 'weight_decay': 1e-5}),
    ('BiGRU + dropout 0.5 + emb. dropout 0.3',        {}, dict(BI, dropout=0.5, embedding_dropout=0.3)),
    ('BiGRU + same, length 400',                      {'max_len': 400}, dict(BI, dropout=0.5, embedding_dropout=0.3)),
]
if os.environ.get('GLOVE'):
    g = os.environ['GLOVE']
    experiments += [('BiGRU + same + GloVe',                {}, dict(BI, dropout=0.5, embedding_dropout=0.3, glove_path=g)),
                    ('BiGRU + same + GloVe, length 400',    {'max_len': 400}, dict(BI, dropout=0.5, embedding_dropout=0.3, glove_path=g))]

rows = []
for label, lo, ov in experiments:
    print('\n=== %s ===' % label)
    scores, m = run_classifier('gru', seed=2, loader_options=lo, verbose=False, max_epoch=15, **ov)
    rows.append({'config': label, 'best_val_acc': max(m.val_acc_history), 'best_epoch': m.best_epoch,
                 'test_accuracy': scores['accuracy'], 'train_acc_at_best': m.train_acc_history[m.best_epoch - 1],
                 'train_time_s': round(m.method_training_time, 1)})
    r = rows[-1]
    print('  val %.4f (epoch %d) | train %.4f | test %.4f | %.1fs' %
          (r['best_val_acc'], r['best_epoch'], r['train_acc_at_best'], r['test_accuracy'], r['train_time_s']))

best = max(rows, key=lambda r: r['best_val_acc'])
print('\nbest configuration by VALIDATION accuracy: %s (val %.4f, test %.4f)' % (best['config'], best['best_val_acc'], best['test_accuracy']))
with open(RESULT_DIR + 'text_regularization.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('saved', RESULT_DIR + 'text_regularization.csv')
