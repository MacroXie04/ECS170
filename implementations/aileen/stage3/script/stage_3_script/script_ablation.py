'''
Stage 3 (3-5): change one CNN setting at a time and record the effect.

    python script_ablation.py MNIST [epochs]        (or ORL, CIFAR)

Every row trains from scratch with the SAME seed and reports the FINAL-epoch test accuracy (no best-epoch picking).
Results -> result/stage_3_result/ablation_<DATASET>.csv and ablation_<DATASET>.png
'''
import sys, csv
from stage3_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

name = sys.argv[1].upper() if len(sys.argv) > 1 else 'MNIST'
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else None
assert name in METHODS

base = METHODS[name]
d = base.conv_channels
c0 = d[0]
# (group, label, overrides)  -- the first entry of each group is the default configuration
experiments = [('default', 'default', {})]
experiments += [('depth', '1 conv layer', {'conv_channels': [c0]}),
                ('depth', '%d conv layers' % (len(d) + 1), {'conv_channels': d + [d[-1] * 2]}),
                ('kernel', 'kernel 5', {'kernel_size': 5, 'padding': 2}),
                ('kernel', 'kernel 7', {'kernel_size': 7, 'padding': 3}),
                ('padding', 'padding 0', {'padding': 0}),
                ('stride', 'stride 2', {'stride': 2}),
                ('pooling', 'avg pool', {'pool': 'avg'}),
                ('pooling', 'no pool', {'pool': None}),
                ('hidden dim', 'hidden 32', {'hidden_dim': 32}),
                ('hidden dim', 'hidden 512', {'hidden_dim': 512}),
                ('hidden dim', 'no hidden layer', {'hidden_dim': 0}),
                ('loss', 'MSE loss', {'loss_name': 'mse'}),
                ('loss', 'label smoothing', {'loss_name': 'ce_ls'}),
                ('batch norm', 'no batch norm', {'batch_norm': False})]
if name == 'CIFAR':
    experiments += [('augmentation', 'no augmentation', {'augment': False})]

rows = []
for group, label, ov in experiments:
    if epochs:
        ov = dict(ov, max_epoch=epochs)
    print('\n=== %s | %s ===' % (group, label))
    try:
        scores, m = run_one(name, seed=2, verbose=False, **ov)
    except Exception as e:                              # e.g. a setting that shrinks the image to nothing
        print('  skipped:', e)
        continue
    rows.append({'group': group, 'config': label, 'accuracy': scores['accuracy'], 'f1_macro': scores['f1_macro'],
                 'final_train_acc': m.train_acc_history[-1], 'params': m.num_parameters(),
                 'train_time_s': round(m.method_training_time, 1)})
    print('  acc %.4f  train acc %.4f  params %d  %.1fs' % (scores['accuracy'], m.train_acc_history[-1],
                                                           m.num_parameters(), m.method_training_time))

csv_path = RESULT_DIR + 'ablation_%s.csv' % name
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

fig, ax = plt.subplots(figsize=(8, 0.35 * len(rows) + 1.5))
labels = [r['config'] for r in rows]
ax.barh(range(len(rows)), [r['accuracy'] for r in rows], color=['tab:orange' if r['group'] == 'default' else 'tab:blue' for r in rows])
ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels); ax.invert_yaxis()
lo = min(r['accuracy'] for r in rows)
ax.set_xlim(max(0, lo - 0.05), 1.0)
ax.set_xlabel('Test accuracy (final epoch)'); ax.set_title('Configuration study: %s' % name)
for i, r in enumerate(rows):
    ax.text(r['accuracy'] + 0.001, i, '%.2f%%' % (100 * r['accuracy']), va='center', fontsize=8)
plt.tight_layout()
plt.savefig(RESULT_DIR + 'ablation_%s.png' % name, dpi=150)
print('\nsaved', csv_path)
