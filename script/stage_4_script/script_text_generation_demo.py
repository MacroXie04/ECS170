'''
Stage 4 (4-4): readable-sentence check for the joke generators.

    python script_text_generation_demo.py gru        (or lstm, rnn)
    GLOVE=/path/to/50d_file.txt python script_text_generation_demo.py gru glove

Trains the same language model as script_text_generation.py (same seed), then compares two decoding settings:
    default  : greedy, as in script_text_generation.py
    improved : greedy with a minimum length of 8 tokens and no repeated word pairs (deterministic)
    sampled  : temperature 0.5, top-5 tokens, minimum length 6, no repeated word pairs (varied)
For 20 start prefixes it prints the outputs, and for the 162 held-out jokes it prints how many outputs are complete sentences
(end with . ! or ?), how many contain a repeated word pair (loops such as 'denim denim denim') and the mean length.
All examples are printed uncurated (fixed seeds) and saved to result/stage_4_result/generation_demo_<cell>.txt.
Environment switches: DEVICE=cpu, EPOCHS=10.
'''
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from local_code.stage_4_code.Dataset_Loader_Jokes import Dataset_Loader_Jokes, detokenize
from local_code.stage_4_code.Method_RNN_Generator import Method_RNN_Generator

RESULT_DIR = '../../result/stage_4_result/'
os.makedirs(RESULT_DIR, exist_ok=True)
rnn_type = sys.argv[1].lower() if len(sys.argv) > 1 else 'gru'
assert rnn_type in ('rnn', 'lstm', 'gru')
use_glove = len(sys.argv) > 2 and sys.argv[2].lower() == 'glove'
suffix = rnn_type + ('_glove' if use_glove else '')
if use_glove:
    assert os.environ.get('GLOVE'), 'set GLOVE=<path to the 50d GloVe text file> first'

data_obj = Dataset_Loader_Jokes('jokes', '')
folder = '../../data/stage_4_data/text_generation/'
data_obj.dataset_source_folder_path = folder if os.path.exists(folder + 'data') else '../../data/stage_4_data/'
data_obj.dataset_source_file_name = 'data'
m = Method_RNN_Generator('recurrent language model', '')
m.rnn_type = rnn_type
if use_glove:
    m.glove_path = os.environ['GLOVE']
m.use_gpu = os.environ.get('DEVICE') != 'cpu'
m.verbose = False
if os.environ.get('EPOCHS'):
    m.max_epoch = int(os.environ['EPOCHS'])
print('************ Start (%s%s) ************' % (rnn_type.upper(), ' + GloVe' if use_glove else ''))
m.data = data_obj.load()
m.run()
print('trained: best epoch %d, training time %.1fs on %s' % (m.best_epoch, m.method_training_time, m.device))

DEFAULT = dict()
IMPROVED = dict(min_len=8, no_repeat=True)
SAMPLED = dict(temperature=0.5, top_k=5, min_len=6, no_repeat=True)

def complete(text):
    return text.rstrip().endswith(('.', '!', '?'))

prefixes = [['what', 'did', 'the'], ['why', 'did', 'the'], ['how', 'many', 'people'], ['my', 'girlfriend', 'told'],
            ['what', 'do', 'you'], ['i', 'told', 'my'], ['a', 'man', 'walks'], ['why', 'don', "'t"], ['what', 'is', 'the'],
            ['knock', 'knock', '.'], ['did', 'you', 'hear'], ['my', 'wife', 'said'], ['what', 'happens', 'when'],
            ['two', 'guys', 'walk'], ['how', 'do', 'you'], ['why', 'was', 'the'], ['my', 'dog', 'is'], ['i', 'went', 'to'],
            ['what', 'kind', 'of'], ['if', 'you', 'ever']]
prefixes = [[t for t in p] for p in prefixes if all(t in m.data['vocab'].stoi for t in p)]

lines = []
def out(s=''):
    print(s); lines.append(s)

for title, kw in (('default (greedy)', DEFAULT), ('improved (greedy, min length 8, no repeated word pairs)', IMPROVED),
                  ('sampled (temperature 0.5, top-5, min length 6, no repeated word pairs)', SAMPLED)):
    out('--- %s: %d prefixes, uncurated ---' % (title, len(prefixes)))
    for i, p in enumerate(prefixes):
        out('  ' + detokenize(m.generate(p, seed=i, **kw)))
    out()

def has_loop(text):
    w = text.split()
    return len(set(zip(w, w[1:]))) != len(w) - 1

out('--- complete sentences, loops and length on the 162 held-out jokes (prefix = first 3 words of each joke) ---')
val = m.data['val_tokens']
for name, kw in (('default', DEFAULT), ('improved', IMPROVED), ('sampled', SAMPLED)):
    texts = [detokenize(m.generate(r[:3], seed=i, **kw)) for i, r in enumerate(val)]
    out('%-9s ends with . ! or ?: %5.1f%% | repeated word pair: %4.1f%% | mean length %.1f words (real jokes %.1f)' %
        (name, 100 * np.mean([complete(t) for t in texts]), 100 * np.mean([has_loop(t) for t in texts]),
         np.mean([len(t.split()) for t in texts]), np.mean([len(r) - 1 for r in val])))
open(RESULT_DIR + 'generation_demo_%s.txt' % suffix, 'w').write('\n'.join(lines) + '\n')
print('\nsaved', RESULT_DIR + 'generation_demo_%s.txt' % rnn_type)
