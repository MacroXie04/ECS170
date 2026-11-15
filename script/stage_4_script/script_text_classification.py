'''
Stage 4 (4-3 / 4-5): train a recurrent text classifier on the movie reviews, print metrics, save learning curves.

    python script_text_classification.py rnn        (or lstm, gru)

Environment switches: DEVICE=cpu (force CPU), EPOCHS=3 (shorter run).
'''
import sys
from stage4_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rnn_type = sys.argv[1].lower() if len(sys.argv) > 1 else 'rnn'
assert rnn_type in ('rnn', 'lstm', 'gru')

print('************ Start (%s) ************' % rnn_type.upper())
scores, m = run_classifier(rnn_type, seed=2)
print('************ Overall Performance (test set) ************')
for k, v in scores.items():
    print('%-20s %.4f' % (k, v))
print('training time: %.1fs on %s, parameters: %d, best epoch: %s' %
      (m.method_training_time, m.device, m.num_parameters(), m.best_epoch))

ep = range(1, len(m.loss_history) + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ep, m.loss_history, marker='o', ms=3)
ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Training loss'); ax[0].set_title('Learning convergence (%s)' % rnn_type.upper())
ax[0].grid(alpha=0.3)
ax[1].plot(ep, m.train_acc_history, marker='o', ms=3, label='train')
ax[1].plot(ep, m.val_acc_history, marker='s', ms=3, label='validation')
ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Accuracy'); ax[1].set_title('Accuracy per epoch (%s)' % rnn_type.upper())
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
out = RESULT_DIR + 'learning_curve_classification_%s.png' % rnn_type
plt.savefig(out, dpi=150)
print('saved', out)
print('************ Finish ************')
