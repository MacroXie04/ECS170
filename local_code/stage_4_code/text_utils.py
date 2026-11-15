'''
Text helpers for stage 4: cleaning / tokenizing, vocabulary, GloVe loading.
'''
import re
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Stop words come from scikit-learn's list, but negations are kept: "not good" must not become "good".
NEGATIONS = {'no', 'not', 'nor', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'cannot', 'without'}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - NEGATIONS

_HTML = re.compile(r'<[^>]+>')
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def clean_review(text, remove_stopwords=True):
    '''review text -> list of lower-case word tokens (no HTML, punctuation, digits, stop words, 1-letter tokens)'''
    text = _HTML.sub(' ', text.lower().replace('’', "'"))
    tokens = _WORD.findall(text)
    return [t for t in tokens if len(t) > 1 and not (remove_stopwords and t in STOP_WORDS)]


class Vocabulary:
    PAD, UNK = '<pad>', '<unk>'

    def __init__(self, specials=('<pad>', '<unk>')):
        self.itos = list(specials)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def build(self, token_lists, max_size=20000, min_freq=2):
        counts = {}
        for toks in token_lists:
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
        words = sorted((w for w, c in counts.items() if c >= min_freq and w not in self.stoi),
                       key=lambda w: (-counts[w], w))
        for w in words[:max_size - len(self.itos)]:
            self.stoi[w] = len(self.itos)
            self.itos.append(w)
        return self

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, max_len=None, pad=True):
        ids = [self.stoi.get(t, 1) for t in tokens]          # 1 = <unk>
        if max_len is not None:
            ids = ids[:max_len]
            if pad:
                ids = ids + [0] * (max_len - len(ids))        # 0 = <pad>: short documents are filled with zeros
        return ids

    def decode(self, ids):
        return [self.itos[i] for i in ids]


def load_glove_matrix(path, vocab, seed=0):
    '''Read a GloVe text file line by line (never the whole file) and keep only the words in `vocab`.
    Words that are not in GloVe get small random vectors; <pad> is all zeros. Returns (matrix, found, dim).'''
    rng = np.random.RandomState(seed)
    with open(path, encoding='utf-8') as f:
        dim = len(f.readline().rstrip().split(' ')) - 1
    matrix = rng.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    matrix[0] = 0.0
    found = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            word = line[:line.find(' ')]
            idx = vocab.stoi.get(word)
            if idx is not None and idx > 1:
                matrix[idx] = np.asarray(line.rstrip().split(' ')[1:], dtype=np.float32)
                found += 1
    return matrix, found, dim
