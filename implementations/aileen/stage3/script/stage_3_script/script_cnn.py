'''
Stage 3 (3-3 / 3-4): train the default CNN on one dataset, print the metrics and save the learning curves.

    python script_cnn.py MNIST        (or ORL, CIFAR)
'''
import sys
from stage3_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

name = sys.argv[1].upper() if len(sys.argv) > 1 else 'MNIST'
assert name in METHODS, 'dataset must be one of ' + str(list(METHODS))

print('************ Start (%s) ************' % name)
scores, m = run_one(name, seed=2)
print('************ Overall Performance ************')
for k, v in scores.items():
    print('%-20s %.4f' % (k, v))
print('training time: %.1fs on %s, parameters: %d' % (m.method_training_time, m.device, m.num_parameters()))

# learning curves: training loss, training accuracy, test accuracy per epoch
ep = range(1, len(m.loss_history) + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ep, m.loss_history, marker='o', ms=3)
ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Training loss'); ax[0].set_title('Learning convergence (%s)' % name)
ax[0].grid(alpha=0.3)
ax[1].plot(ep, m.train_acc_history, marker='o', ms=3, label='train')
if m.test_acc_history:
    ax[1].plot(ep, m.test_acc_history, marker='s', ms=3, label='test')
ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Accuracy'); ax[1].set_title('Accuracy per epoch (%s)' % name)
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
out = RESULT_DIR + 'learning_curve_%s.png' % name
plt.savefig(out, dpi=150)
print('saved', out)
print('************ Finish ************')
