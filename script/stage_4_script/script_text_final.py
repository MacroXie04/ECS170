'''
Stage 4 (4-5): the selected model vs the plain GRU baseline - metrics and figures for the report.

    GLOVE=/path/to/50d_file.txt python script_text_final.py

Trains (same seed, 15 epochs, best epoch chosen on validation accuracy):
    baseline : GRU, last hidden state, dropout 0.3, learned embedding, length 200
    final    : bidirectional GRU, mean pooling, GloVe 50d, dropout 0.5, embedding dropout 0.3, length 400
Saves result/stage_4_result/learning_curve_classification_final.png (loss + accuracy per epoch of the final model) and
text_baseline_vs_final.png (train and validation accuracy of both models).
'''
from stage4_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

assert os.environ.get('GLOVE'), 'set GLOVE=<path to the 50d GloVe text file> first'

runs = {}
for name, lo, ov in [('baseline', {}, {}),
                     ('final', {'max_len': 400}, dict(bidirectional=True, pooling='mean', dropout=0.5, embedding_dropout=0.3,
                                                      glove_path=os.environ['GLOVE']))]:
    print('\n************ %s ************' % name)
    scores, m = run_classifier('gru', seed=2, loader_options=lo, verbose=False, max_epoch=15, **ov)
    runs[name] = (scores, m)
    print('  best epoch %d | val %.4f | train %.4f | test accuracy %.4f | macro F1 %.4f | %.1fs' %
          (m.best_epoch, max(m.val_acc_history), m.train_acc_history[m.best_epoch - 1], scores['accuracy'],
           scores['f1_macro'], m.method_training_time))


def save_figures(runs, result_dir):
    m = runs['final'][1]
    ep = range(1, len(m.loss_history) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, m.loss_history, marker='o', ms=3)
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Training loss'); ax[0].set_title('Learning convergence (selected model)')
    ax[0].grid(alpha=0.3)
    ax[1].plot(ep, m.train_acc_history, marker='o', ms=3, label='train')
    ax[1].plot(ep, m.val_acc_history, marker='s', ms=3, label='validation')
    ax[1].axvline(m.best_epoch, color='gray', ls=':', label='best epoch (%d)' % m.best_epoch)
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Accuracy'); ax[1].set_title('Accuracy per epoch (selected model)')
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(result_dir + 'learning_curve_classification_final.png', dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    style = {'baseline': ('#7f7f7f', 'GRU baseline'), 'final': ('#1f77b4', 'selected model')}
    for name, (scores, mm) in runs.items():
        color, label = style[name]
        e = range(1, len(mm.val_acc_history) + 1)
        ax.plot(e, mm.train_acc_history, color=color, ls='--', alpha=0.7, label=label + ' - train')
        ax.plot(e, mm.val_acc_history, color=color, label=label + ' - validation (test %.2f%%)' % (100 * scores['accuracy']))
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_title('Baseline vs selected model')
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout(); plt.savefig(result_dir + 'text_baseline_vs_final.png', dpi=150); plt.close(fig)


save_figures(runs, RESULT_DIR)
print('\nsaved', RESULT_DIR + 'learning_curve_classification_final.png', 'and', RESULT_DIR + 'text_baseline_vs_final.png')
