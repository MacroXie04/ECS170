'''
Stage 4 (4-4 / 4-5): train a recurrent language model on the jokes and generate a joke from three start words.

    python script_text_generation.py lstm        (or rnn, gru)
    GLOVE=/path/to/50d_file.txt python script_text_generation.py gru glove      (embedding initialised from GloVe)

Environment switches: DEVICE=cpu, EPOCHS=10.
Output: learning curve PNG, generation_examples_<type>.txt, and the comparison metrics on screen.
'''
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import random
import numpy as np
import torch
from local_code.stage_4_code.azheng.Dataset_Loader_Jokes import Dataset_Loader_Jokes, detokenize
from local_code.stage_4_code.azheng.Method_RNN_Generator import Method_RNN_Generator
from local_code.stage_4_code.azheng.Evaluate_Generation import Evaluate_Generation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_DIR = '../../result/stage_4_result/'
os.makedirs(RESULT_DIR, exist_ok=True)
rnn_type = sys.argv[1].lower() if len(sys.argv) > 1 else 'lstm'
assert rnn_type in ('rnn', 'lstm', 'gru')
use_glove = len(sys.argv) > 2 and sys.argv[2].lower() == 'glove'
suffix = rnn_type + ('_glove' if use_glove else '')
if use_glove:
    assert os.environ.get('GLOVE'), 'set GLOVE=<path to the 50d GloVe text file> first'

data_obj = Dataset_Loader_Jokes('jokes', '')
folder = '../../data/stage_4_data/text_generation/'
data_obj.dataset_source_folder_path = folder if os.path.exists(folder + 'data') else '../../data/stage_4_data/'
data_obj.dataset_source_file_name = 'data'

method_obj = Method_RNN_Generator('recurrent language model', '')
method_obj.rnn_type = rnn_type
if use_glove:
    method_obj.glove_path = os.environ['GLOVE']
method_obj.use_gpu = os.environ.get('DEVICE') != 'cpu'
if os.environ.get('EPOCHS'):
    method_obj.max_epoch = int(os.environ['EPOCHS'])

print('************ Start (%s%s) ************' % (rnn_type.upper(), ' + GloVe' if use_glove else ''))
method_obj.data = data_obj.load()
method_obj.run()
print('training time: %.1fs on %s' % (method_obj.method_training_time, method_obj.device))

# ---- generation examples ----
train_tokens, val_tokens = method_obj.data['train_tokens'], method_obj.data['val_tokens']
vocab = method_obj.data['vocab']
lines = []
def show(prefix, mode, **kw):
    toks = method_obj.generate(prefix, **kw)
    lines.append('[%s] %s -> %s' % (mode, ' '.join(prefix), detokenize(toks)))
    print(lines[-1])

print('\n--- start words taken from the dataset (greedy, then sampled) ---')
for p in (['what', 'did', 'the'], ['why', 'did', 'the'], ['how', 'many', 'people'], ['my', 'girlfriend', 'told']):
    show(p, 'greedy')
    show(p, 'sampled T=0.8', temperature=0.8, top_k=10, seed=1)
print('\n--- three start words that are not in any joke ---')
for p in (['purple', 'robots', 'danced'], ['i', 'love', 'pizza']):
    show(p, 'greedy')
    show(p, 'sampled T=0.8', temperature=0.8, top_k=10, seed=1)
open(RESULT_DIR + 'generation_examples_%s.txt' % suffix, 'w').write('\n'.join(lines) + '\n')

# ---- compare with the original jokes: prefix = first three words of each reference joke ----
random.seed(0)
for name, refs in (('training jokes', random.sample(train_tokens, min(200, len(train_tokens)))),
                   ('validation jokes (never trained on)', val_tokens)):
    gen = [method_obj.generate(r[:3]) for r in refs]
    ref = [r[:-1] for r in refs]                                    # drop <eos>
    ev = Evaluate_Generation('generation vs original', '')
    ev.data = {'generated': gen, 'reference': ref, 'training': [t[:-1] for t in train_tokens]}
    print('\n************ Generated vs original: %s (greedy, n=%d) ************' % (name, len(refs)))
    for k, v in ev.evaluate().items():
        print('%-24s %.4f' % (k, v))

ep = range(1, len(method_obj.loss_history) + 1)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ep, method_obj.loss_history, marker='o', ms=3, label='train')
ax.plot(ep, method_obj.val_loss_history, marker='s', ms=3, label='validation')
ax.axvline(method_obj.best_epoch, color='#888', ls=':', label='selected epoch')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (per token)'); ax.set_title('Joke generation (%s)' % suffix.upper())
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR + 'learning_curve_generation_%s.png' % suffix, dpi=150)
print('\nsaved learning curve and examples in', RESULT_DIR)
print('************ Finish ************')
