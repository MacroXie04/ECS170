'''
Stage 4 (4-5): last hidden state vs mean pooling over all hidden states.

    GLOVE=/path/to/50d_file.txt python script_text_pooling.py       (without GLOVE the two GloVe rows are skipped)

Same data, seed and settings as script_text_regularization.py (15 epochs). Only the pooling changes, so each row can be
compared with its "last hidden state" row there. Best epoch and best configuration are chosen on VALIDATION accuracy;
test accuracy is only reported. -> result/stage_4_result/text_pooling.csv
'''
import csv
from stage4_common import *

BI = {'bidirectional': True, 'dropout': 0.5, 'embedding_dropout': 0.3}
experiments = [
    ('GRU, mean pooling (baseline settings)',        {}, {'pooling': 'mean'}),
    ('BiGRU + regularization, mean pooling',         {}, dict(BI, pooling='mean')),
]
if os.environ.get('GLOVE'):
    g = os.environ['GLOVE']
    experiments += [('BiGRU + regularization + GloVe, mean pooling',               {}, dict(BI, pooling='mean', glove_path=g)),
                    ('BiGRU + regularization + GloVe, mean pooling, length 400',   {'max_len': 400}, dict(BI, pooling='mean', glove_path=g))]
else:
    print('GLOVE is not set: skipping the two GloVe rows')

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
print('for comparison, last-hidden-state rows (script_text_regularization.py): GRU val 0.8512 test 0.8485 | BiGRU+reg val 0.8784 test 0.8677 | '
      '+GloVe val 0.8880 test 0.8747 | +GloVe length 400 val 0.8912 test 0.8869')
with open(RESULT_DIR + 'text_pooling.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('saved', RESULT_DIR + 'text_pooling.csv')
