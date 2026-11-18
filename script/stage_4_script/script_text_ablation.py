'''
Stage 4 (4-5): compare RNN / LSTM / GRU and change other architecture settings, one at a time.

    python script_text_ablation.py                 (set GLOVE=/path/to/glove.6B.50d.txt to also test GloVe)

Every row trains from scratch with the same seed; the best epoch is picked on the validation split and the test set is
scored once. -> result/stage_4_result/text_ablation.csv and text_ablation.png
'''
import csv
from stage4_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# (group, label, rnn_type, loader options, method overrides)
experiments = [('cell', 'RNN', 'rnn', {}, {}),
               ('cell', 'LSTM', 'lstm', {}, {}),
               ('cell', 'GRU', 'gru', {}, {}),
               ('architecture', 'GRU bidirectional', 'gru', {}, {'bidirectional': True}),
               ('architecture', 'GRU 2 layers', 'gru', {}, {'num_layers': 2}),
               ('architecture', 'GRU hidden 128', 'gru', {}, {'hidden_size': 128}),
               ('architecture', 'GRU embedding 100', 'gru', {}, {'embedding_dim': 100}),
               ('input', 'GRU length 100', 'gru', {'max_len': 100}, {}),
               ('input', 'GRU length 400', 'gru', {'max_len': 400}, {}),
               ('input', 'GRU keep stop words', 'gru', {'remove_stopwords': False}, {})]
if os.environ.get('GLOVE'):
    experiments.append(('embedding', 'GRU + GloVe', 'gru', {}, {'glove_path': os.environ['GLOVE']}))

rows = []
for group, label, cell, lo, ov in experiments:
    print('\n=== %s | %s ===' % (group, label))
    scores, m = run_classifier(cell, seed=2, loader_options=lo, verbose=False, **ov)
    rows.append({'group': group, 'config': label, 'test_accuracy': scores['accuracy'], 'f1_macro': scores['f1_macro'],
                 'best_val_acc': max(m.val_acc_history), 'best_epoch': m.best_epoch,
                 'final_train_acc': m.train_acc_history[-1], 'params': m.num_parameters(),
                 'train_time_s': round(m.method_training_time, 1)})
    print('  test acc %.4f  best val %.4f (epoch %d)  train acc %.4f  %.1fs' %
          (scores['accuracy'], rows[-1]['best_val_acc'], m.best_epoch, rows[-1]['final_train_acc'], m.method_training_time))

with open(RESULT_DIR + 'text_ablation.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

fig, ax = plt.subplots(figsize=(7, 0.38 * len(rows) + 1.4))
y = np.arange(len(rows))
acc = np.array([r['test_accuracy'] for r in rows])
ax.scatter(acc, y, color='#1f77b4', s=30, zorder=2)
ax.set_yticks(y); ax.set_yticklabels([r['config'] for r in rows]); ax.invert_yaxis()
ax.set_xlabel('Test accuracy'); ax.set_title('Text classification: RNN variants'); ax.grid(axis='x', alpha=0.3)
for yi, a in zip(y, acc):
    ax.text(a + 0.002, yi, '%.2f%%' % (100 * a), va='center', fontsize=8)
ax.set_xlim(max(0.5, acc.min() - 0.03), min(1.0, acc.max() + 0.04))
plt.tight_layout()
plt.savefig(RESULT_DIR + 'text_ablation.png', dpi=150)
print('\nsaved', RESULT_DIR + 'text_ablation.csv')
