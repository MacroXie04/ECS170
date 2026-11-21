"""A train-only question/punchline compatibility model with frozen word vectors.

This estimates compatibility against shuffled negatives, not humor or logical
correctness. It never produces text, retrieves a joke, or reads test examples.
"""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path
import torch
from torch import nn
from local_code.stage_4_code.Dataset_Loader import tokenize, Vocabulary

STOP = set('a an the is are was were do does did to of in on at for and or but it its this that i you he she they we what why how where when which who'.split())


def question_pair(text):
    tokens = tokenize(text)
    if '?' not in tokens: return None
    cut = tokens.index('?') + 1
    left, right = tokens[:cut], tokens[cut:]
    if sum(any(c.isalnum() for c in t) for t in left) < 3 or sum(any(c.isalnum() for c in t) for t in right) < 2:
        return None
    return left, right


def make_pairs(records, seed=1704):
    """One positive, one random negative, and one same-opening negative per joke."""
    pairs = [(row['id'], question_pair(row['text'])) for row in records]
    pairs = [(identifier,pair) for identifier,pair in pairs if pair is not None]
    rng = random.Random(seed); result = []
    positives = {(tuple(pair[0]),tuple(pair[1])) for _,pair in pairs}
    for index,(identifier,(left,right)) in enumerate(pairs):
        result.append(dict(left=left,right=right,label=1,setup_id=identifier,punchline_id=identifier,kind='positive'))
        choices = [j for j,(_,other) in enumerate(pairs) if j != index and tuple(other[1]) != tuple(right)
                   and (tuple(left),tuple(other[1])) not in positives]
        if not choices: continue
        same = [j for j in choices if pairs[j][1][0][:2] == left[:2]
                and .5*len(right) <= len(pairs[j][1][1]) <= 2*len(right)]
        first = rng.choice(choices)
        hard = [j for j in (same or choices) if j != first] or [first]
        for kind,j in [('random_negative',first),('same_opening_negative',rng.choice(hard))]:
            other_id,other = pairs[j]
            result.append(dict(left=left,right=other[1],label=0,setup_id=identifier,punchline_id=other_id,kind=kind))
    return result


class PairRanker(nn.Module):
    def __init__(self, matrix, vocab_tokens):
        super().__init__()
        if matrix.ndim != 2 or matrix.shape[0] != len(vocab_tokens): raise ValueError('Word vector shape mismatch')
        self.register_buffer('vectors', matrix)
        self.vocab_tokens = list(vocab_tokens)
        self.stoi = {t:i for i,t in enumerate(vocab_tokens)}
        self.network = nn.Sequential(nn.Linear(matrix.shape[1]*4+4,64),nn.Tanh(),nn.Dropout(.2),nn.Linear(64,1))

    def _words(self,tokens):
        words = [t for t in tokens if any(c.isalnum() for c in t)]
        content = [t for t in words if t not in STOP] or words
        return [self.stoi[t] for t in content if self.stoi.get(t,0) >= 4]

    def features(self, pairs):
        result = []
        for pair in pairs:
            left,right = self._words(pair['left']),self._words(pair['right'])
            dim = self.vectors.shape[1]
            a = self.vectors[left] if left else self.vectors.new_zeros((1,dim))
            b = self.vectors[right] if right else self.vectors.new_zeros((1,dim))
            u,v = nn.functional.normalize(a.mean(0),dim=0),nn.functional.normalize(b.mean(0),dim=0)
            similarities = a @ b.T
            overlap = len(set(left)&set(right))/max(1,len(set(left)|set(right)))
            extras = torch.stack([(u*v).sum(),similarities.max(1).values.mean(),similarities.max(0).values.mean(),u.new_tensor(overlap)])
            result.append(torch.cat([u,v,(u-v).abs(),u*v,extras]))
        return torch.stack(result) if result else self.vectors.new_empty((0,self.network[0].in_features))

    def forward(self, features): return self.network(features).squeeze(-1)

    @torch.no_grad()
    def score_texts(self,texts):
        pairs = [question_pair(text) for text in texts]
        valid = [i for i,pair in enumerate(pairs) if pair is not None]
        scores = [None]*len(texts)
        if not valid: return scores
        was_training = self.training; self.eval()
        try:
            features = self.features([dict(left=pairs[i][0],right=pairs[i][1]) for i in valid])
            for i,score in zip(valid,self(features).sigmoid().cpu().tolist()): scores[i] = score
        finally: self.train(was_training)
        return scores


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def load_ranker(checkpoint,device='cpu'):
    path = Path(checkpoint).resolve(); root = Path(__file__).resolve().parents[2]
    config = json.loads((path.parent.parent/'config.json').read_text())
    saved = torch.load(path,map_location='cpu',weights_only=True)
    if saved['config_hash'] != canonical(config): raise ValueError('Ranker configuration changed')
    for name,expected in config['source_sha256'].items():
        source = (root/name).resolve()
        if not source.is_relative_to(root) or digest(source) != expected: raise ValueError('Ranker source changed: '+name)
    vocab = Vocabulary.from_dict(saved['vocab'])
    if vocab.to_dict() != saved['vocab'] or hashlib.sha256(b''.join(t.encode()+b'\0' for t in vocab.tokens)).hexdigest() != config['data_metadata']['vocabulary_sha256']:
        raise ValueError('Ranker vocabulary changed')
    with torch.random.fork_rng(devices=[]): model = PairRanker(saved['model']['vectors'],vocab.tokens)
    model.load_state_dict(saved['model'],strict=True); model.to(device).eval()
    return model, dict(config=config,epoch=saved['epoch'],checkpoint_sha256=digest(path),validation_auc=saved['validation_auc'])
