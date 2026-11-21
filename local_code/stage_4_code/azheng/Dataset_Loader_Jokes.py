'''
Dataset loader for the stage 4 text generation data: a CSV of short jokes ("ID","Joke").

Each joke becomes a list of lower-case word and punctuation tokens ending with <eos>. Jokes are cut at max_len tokens and
filled with 0 (<pad>). A fixed share of the jokes is held out as a validation set.
'''
import csv
import re
import numpy as np
from local_code.base_class.dataset import dataset
from local_code.stage_4_code.azheng.text_utils import Vocabulary

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?]")


def tokenize_joke(text):
    return _TOKEN.findall(text.lower().replace('’', "'").replace('‘', "'"))


def detokenize(tokens):
    text = ' '.join(t for t in tokens if t not in ('<eos>', '<pad>'))
    return re.sub(r'\s+([.,!?])', r'\1', text)


class Dataset_Loader_Jokes(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    max_len = 40                 # tokens per joke including <eos>
    min_freq = 2                 # words seen fewer times in the training jokes become <unk>
    val_fraction = 0.1
    seed = 0

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading jokes from ' + self.dataset_source_folder_path + self.dataset_source_file_name + '...')
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, encoding='utf-8', errors='ignore') as f:
            jokes = [row['Joke'] for row in csv.DictReader(f) if row.get('Joke')]
        docs = [tokenize_joke(j)[:self.max_len - 1] + ['<eos>'] for j in jokes]
        docs = [d for d in docs if len(d) > 4]                     # need at least the 3 start words and something to generate
        order = np.random.RandomState(self.seed).permutation(len(docs))
        n_val = int(round(self.val_fraction * len(docs)))
        val_docs = [docs[i] for i in order[:n_val]]
        train_docs = [docs[i] for i in order[n_val:]]
        vocab = Vocabulary(specials=('<pad>', '<unk>', '<eos>')).build(train_docs, max_size=100000, min_freq=self.min_freq)
        enc = lambda ds: np.array([vocab.encode(d, self.max_len) for d in ds], dtype=np.int64)
        print('  %d jokes: %d train / %d validation | vocabulary %d tokens | mean length %.1f tokens'
              % (len(docs), len(train_docs), len(val_docs), len(vocab), np.mean([len(d) for d in docs])))
        return {'train': {'X': enc(train_docs)}, 'val': {'X': enc(val_docs)}, 'vocab': vocab,
                'train_tokens': train_docs, 'val_tokens': val_docs}
