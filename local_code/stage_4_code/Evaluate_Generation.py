'''
Compare generated jokes with the original training jokes.

For every reference joke the model is given its first three words and asked for the whole joke. Scores (token level):
  unigram / bigram precision : share of the generated words / word pairs that also occur in the reference (clipped counts)
  exact match                : generated text identical to the reference
  copied from training       : generated text identical to ANY training joke (memorising rather than writing)
'''
from collections import Counter
from local_code.base_class.evaluate import evaluate


def _precision(gen, ref, n):
    g = Counter(tuple(gen[i:i + n]) for i in range(len(gen) - n + 1))
    r = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
    total = sum(g.values())
    return sum(min(c, r[k]) for k, c in g.items()) / total if total else 0.0


class Evaluate_Generation(evaluate):
    data = None            # {'generated': [token lists], 'reference': [token lists], 'training': [token lists]}

    def evaluate(self):
        gen, ref = self.data['generated'], self.data['reference']
        train_set = set(tuple(t) for t in self.data['training'])
        n = len(gen)
        return {'unigram_precision': sum(_precision(g, r, 1) for g, r in zip(gen, ref)) / n,
                'bigram_precision': sum(_precision(g, r, 2) for g, r in zip(gen, ref)) / n,
                'exact_match': sum(g == r for g, r in zip(gen, ref)) / n,
                'copied_from_training': sum(tuple(g) in train_set for g in gen) / n,
                'mean_generated_length': sum(len(g) for g in gen) / n,
                'mean_reference_length': sum(len(r) for r in ref) / n}
