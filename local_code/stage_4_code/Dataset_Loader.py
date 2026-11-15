'''
Dataset loader for the stage 4 text classification data (IMDB-style movie reviews).

Folder layout: <folder>/train/{pos,neg}/*.txt and <folder>/test/{pos,neg}/*.txt.
Steps: read -> clean (lower-case, strip HTML, split words, drop punctuation / stop words) -> vocabulary from the
TRAIN reviews -> every review becomes max_len word ids (cut if longer, filled with 0 if shorter).
'''
import os
import glob
import pickle
import numpy as np
from local_code.base_class.dataset import dataset
from local_code.stage_4_code.text_utils import clean_review, Vocabulary


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None      # folder that contains train/ and test/
    dataset_source_file_name = None
    max_len = 200                          # n: words kept per review
    max_vocab = 20000
    min_freq = 2                           # words seen fewer times in train become <unk>
    remove_stopwords = True
    cache_folder = None                    # cleaned tokens are cached here so cleaning (about a minute) runs once

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _read_split(self, split):
        docs, labels = [], []
        for label, sub in ((0, 'neg'), (1, 'pos')):
            files = sorted(glob.glob(os.path.join(self.dataset_source_folder_path, split, sub, '*.txt')))
            for fp in files:
                with open(fp, encoding='utf-8', errors='ignore') as f:
                    docs.append(clean_review(f.read(), self.remove_stopwords))
                labels.append(label)
        return docs, np.array(labels, dtype=np.int64)

    def _tokens(self):
        cache = None
        if self.cache_folder:
            os.makedirs(self.cache_folder, exist_ok=True)
            cache = os.path.join(self.cache_folder, 'review_tokens_stop%d.pkl' % int(self.remove_stopwords))
            if os.path.exists(cache):
                print('  using cleaned tokens cached in', cache)
                with open(cache, 'rb') as f:
                    return pickle.load(f)
        out = {'train': self._read_split('train'), 'test': self._read_split('test')}
        if cache:
            with open(cache, 'wb') as f:
                pickle.dump(out, f)
        return out

    def load(self):
        print('loading and cleaning reviews from ' + str(self.dataset_source_folder_path) + '...')
        tok = self._tokens()
        (tr_docs, y_tr), (te_docs, y_te) = tok['train'], tok['test']
        vocab = Vocabulary().build(tr_docs, self.max_vocab, self.min_freq)
        X_tr = np.array([vocab.encode(d, self.max_len) for d in tr_docs], dtype=np.int64)
        X_te = np.array([vocab.encode(d, self.max_len) for d in te_docs], dtype=np.int64)
        lens = np.array([len(d) for d in tr_docs])
        print('  train %s test %s | vocabulary %d words | median length %d words, %.0f%% of reviews cut at %d'
              % (X_tr.shape, X_te.shape, len(vocab), np.median(lens), 100 * (lens > self.max_len).mean(), self.max_len))
        return {'train': {'X': X_tr, 'y': y_tr}, 'test': {'X': X_te, 'y': y_te}, 'vocab': vocab}
