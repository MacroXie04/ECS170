"""Soft train-span penalties during beam search, without retrieval or rewriting.

The penalty changes which recurrent-model branches are explored. It does not
certify semantic novelty or coherence, and does not forbid every training copy.
Original-model probability, repetition-adjusted LM score, and copy cost remain
separate so the search objective can be reconstructed from saved evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math

import torch

from local_code.stage_4_code.Generation_Decoder import (
    _clone_hidden, _length_norm, _scores,
)
from local_code.stage_4_code.Generation_Optimization import format_completion, validate_prefix
from local_code.stage_4_code.Novel_Generation import TrainingNoveltyIndex, lexical_tokens


class CopySpanIndex:
    """Document-frequency-weighted lexical n-grams from TRAIN records only.

    N-grams occurring in at least max(3, ceil(0.02*N)) training rows are treated
    as common boilerplate and incur no cost. Other matching extensions cost
    1/sqrt(document frequency). Only generated words enter the suffix history;
    the three prompt words never themselves incur a copy penalty. Punctuation
    neither breaks a matching lexical span nor creates an additional cost.
    """

    def __init__(self, training_records, min_match_words=5):
        if isinstance(min_match_words, bool) or not isinstance(min_match_words, int) or min_match_words < 3:
            raise ValueError('min_match_words must be an integer of at least three')
        self.novelty = training_records if isinstance(training_records, TrainingNoveltyIndex) else TrainingNoveltyIndex(training_records)
        self.min_match_words = min_match_words
        self.boilerplate_document_frequency = max(3, math.ceil(.02*len(self.novelty.rows)))
        documents = defaultdict(list)
        for identifier, words, _ in self.novelty.rows:
            for gram in set(tuple(words[i:i+min_match_words]) for i in range(len(words)-min_match_words+1)):
                documents[gram].append(identifier)
        self.extensions = defaultdict(dict)
        self.ignored_boilerplate_count = 0
        for gram, identifiers in documents.items():
            frequency = len(identifiers)
            if frequency >= self.boilerplate_document_frequency:
                self.ignored_boilerplate_count += 1
                continue
            self.extensions[gram[:-1]][gram[-1]] = dict(
                lexical_ngram=list(gram), document_frequency=frequency,
                training_ids=sorted(identifiers), raw_cost=1/math.sqrt(frequency))

    def next_words(self, generated_words):
        width = self.min_match_words-1
        if len(generated_words) < width:
            return {}
        return self.extensions.get(tuple(generated_words[-width:]), {})


@dataclass
class _Hypothesis:
    ids: list[int]
    model_logp: float = 0.
    search_lm_logp: float = 0.
    raw_copy_cost: float = 0.
    words: tuple[str, ...] = ()
    events: list[dict] = field(default_factory=list)
    hidden: object = None
    logits: torch.Tensor | None = None
    finish_reason: str = 'max_new_tokens'


def _score(hypothesis, weight):
    return hypothesis.search_lm_logp-weight*hypothesis.raw_copy_cost


def _rank(hypothesis, prefix_length, weight):
    return _score(hypothesis,weight)/_length_norm(len(hypothesis.ids)-prefix_length,.7)


def _repetitions(words):
    counts = Counter(zip(words,words[1:],words[2:]))
    return sum(n-1 for n in counts.values())


def _check_logits(logits, size):
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < 1 or logits.shape[-1] != size:
        raise ValueError('model.step must return logits shaped [1,positive_length,vocab_size]')


@torch.no_grad()
def search_candidates(model, vocab, prompt, training_records, *, beam_width=12,
                      max_new_tokens=48, copy_penalty=2., min_match_words=5,
                      num_return_sequences=8, seed=1704, top_k=40,
                      min_relative_probability=.01):
    """Return several beam completions under a soft copied-span search cost.

    A copied n-word span starts a cost at its nth generated lexical word, then
    another cost for every matching overlapping extension. Frequent n-grams
    are exempt. Costs are subtracted AFTER LM log-softmax without renormalizing:
    this is a path-ranking objective, not a modified likelihood estimate.

    Beam constraints match the earlier decoder: no PAD/UNK/BOS output, minimum
    four emitted tokens before EOS (limited by the token budget), blocked token
    trigrams, repetition penalty 1.1, and GNMT length penalty .7. Copy penalties
    are soft: high-probability copies may remain in the candidate pool, and no
    post-hoc substitution or hidden filtering occurs. Final hypotheses are
    deduplicated by lexical text, retaining the highest search score; punctuation
    variants are recorded in ``equivalent_token_sequence_count``.

    Before adding copy costs, expansions must be within the original model's
    top-k permitted token IDs AND have at least ``min_relative_probability``
    times its best permitted token's probability. PAD/UNK/BOS and premature EOS
    are excluded when defining that reference maximum. Repetition constraints
    can exhaust these plausible options; they are never relaxed to reach an
    implausible token merely because it would evade a copy penalty.

    Completed hypotheses are preferred when any exist. If none finish, return
    explicitly incomplete paths, never fabricated EOS. The deterministic beam
    uses no RNG; ``seed`` is validated and recorded only for interface consistency.
    All module train/eval flags are restored even if inference fails.
    """
    for name,value in (('beam_width',beam_width),('max_new_tokens',max_new_tokens),
                       ('num_return_sequences',num_return_sequences),('seed',seed),('top_k',top_k)):
        if isinstance(value,bool) or not isinstance(value,int):
            raise ValueError(name+' must be an integer')
    if beam_width < 1 or max_new_tokens < 0 or not 1 <= num_return_sequences <= beam_width:
        raise ValueError('Require beam_width >= num_return_sequences >= 1 and max_new_tokens >= 0')
    if seed < 0 or seed >= 2**63:
        raise ValueError('seed must be a nonnegative signed 64-bit integer')
    if isinstance(min_match_words,bool) or not isinstance(min_match_words,int) or min_match_words < 3:
        raise ValueError('min_match_words must be an integer of at least three')
    if isinstance(copy_penalty,bool) or not isinstance(copy_penalty,(int,float)) or not math.isfinite(copy_penalty) or copy_penalty < 0:
        raise ValueError('copy_penalty must be finite and nonnegative')
    if top_k < 1 or isinstance(min_relative_probability,bool) or not isinstance(min_relative_probability,(int,float)) or not math.isfinite(min_relative_probability) or not 0 < min_relative_probability <= 1:
        raise ValueError('Require top_k >= 1 and finite min_relative_probability in (0,1]')
    if getattr(model,'task',None) != 'generation' or int(model.vocab_size) != len(vocab):
        raise ValueError('Expected a generation model with the supplied vocabulary size')
    prefix = validate_prefix(prompt,vocab)
    if isinstance(training_records,CopySpanIndex):
        index = training_records
        if index.min_match_words != min_match_words:
            raise ValueError('Prebuilt copy index has a different minimum span length')
    else:
        index = CopySpanIndex(training_records,min_match_words)
    words_by_id = [lexical_tokens(token) if i >= 4 else () for i,token in enumerate(vocab.tokens)]
    if any(len(words) > 1 for words in words_by_id):
        raise ValueError('Copy-aware search requires a word-tokenized vocabulary')
    ids_by_word = defaultdict(list)
    for token,words in enumerate(words_by_id):
        if words: ids_by_word[words[0]].append(token)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    minimum = min(4,max_new_tokens)
    settings = dict(strategy='copy_aware_beam',beam_width=beam_width,max_new_tokens=max_new_tokens,
        copy_penalty=copy_penalty,min_match_words=min_match_words,num_return_sequences=num_return_sequences,
        seed=seed,seed_applies_to='none; deterministic beam',min_new_tokens=minimum,
        top_k=top_k,min_relative_probability=min_relative_probability,
        plausibility_reference='Original model distribution, excluding special tokens and premature EOS; copy penalties and repetition constraints do not lower the reference maximum.',
        no_repeat_ngram_size=3,repetition_penalty=1.1,length_penalty=.7,
        boilerplate_document_frequency=index.boilerplate_document_frequency,
        boilerplate_ngram_count=index.ignored_boilerplate_count,
        penalty_definition='Unnormalized path cost: copy_penalty/sqrt(training document frequency) per matching rare lexical n-gram extension.')
    initial = _Hypothesis(list(prefix))
    active, finished, stalled = [initial], [], []
    modes = [(module,module.training) for module in model.modules()]
    model.eval()
    try:
        if max_new_tokens:
            logits,hidden = model.step(torch.tensor([prefix],dtype=torch.long,device=device))
            _check_logits(logits,len(vocab))
            initial.logits,initial.hidden = logits[0,-1],hidden
        for emitted in range(max_new_tokens):
            expansions = []
            for hypothesis in active:
                original,scores = _scores(hypothesis.logits,hypothesis.ids,emitted,minimum,3,1.1,1.)
                plausible = original.clone()
                plausible[:3] = -torch.inf
                if emitted < minimum: plausible[3] = -torch.inf
                plausible_ids = torch.argsort(plausible,descending=True,stable=True)[:min(top_k,len(vocab))]
                allowed = torch.zeros_like(plausible,dtype=torch.bool)
                allowed[plausible_ids] = torch.isfinite(plausible[plausible_ids])
                if torch.isfinite(plausible).any():
                    allowed &= plausible >= plausible.max()+math.log(min_relative_probability)
                scores[~allowed] = -torch.inf
                matches = {}
                for word,event in index.next_words(hypothesis.words).items():
                    for token in ids_by_word.get(word,()): matches[token] = event
                adjusted = scores.clone()
                for token,event in matches.items():
                    adjusted[token] -= copy_penalty*event['raw_cost']

                def extend(token,reason='max_new_tokens'):
                    event = matches.get(token)
                    events = hypothesis.events
                    cost = hypothesis.raw_copy_cost
                    if event is not None:
                        events = events+[dict(event,generated_token_index=emitted,
                                             generated_word_index=len(hypothesis.words),
                                             weighted_cost=copy_penalty*event['raw_cost'])]
                        cost += event['raw_cost']
                    return _Hypothesis(ids=hypothesis.ids+[token],
                        model_logp=hypothesis.model_logp+float(original[token]),
                        search_lm_logp=hypothesis.search_lm_logp+float(scores[token]),
                        raw_copy_cost=cost,words=hypothesis.words+words_by_id[token],
                        events=events,hidden=hypothesis.hidden,finish_reason=reason)

                if torch.isfinite(adjusted[3]): finished.append(extend(3,'eos'))
                adjusted[3] = -torch.inf
                finite = int(torch.isfinite(adjusted).sum())
                if not finite:
                    hypothesis.finish_reason = 'constraints_exhausted'
                    stalled.append(hypothesis)
                    continue
                # Stable sort makes equal-score choices deterministic by token ID.
                indices = torch.argsort(adjusted,descending=True,stable=True)[:min(beam_width,finite)]
                expansions.extend(extend(int(token)) for token in indices)
            expansions.sort(key=lambda h:(-_score(h,copy_penalty),tuple(h.ids)))
            active = expansions[:beam_width]
            if not active: break
            if emitted+1 < max_new_tokens:
                for hypothesis in active:
                    logits,hidden = model.step(torch.tensor([[hypothesis.ids[-1]]],dtype=torch.long,device=device),
                                               _clone_hidden(hypothesis.hidden))
                    _check_logits(logits,len(vocab))
                    hypothesis.logits,hypothesis.hidden = logits[0,-1],hidden
        pool = finished if finished else active+stalled
        pool.sort(key=lambda h:(-_rank(h,len(prefix),copy_penalty),tuple(h.ids)))
        distinct = {}
        for hypothesis in pool:
            key = hypothesis.words
            if key in distinct:
                distinct[key][1] += 1
            else:
                distinct[key] = [hypothesis,1]
        results = []
        for hypothesis,equivalent_count in list(distinct.values())[:num_return_sequences]:
            continuation = vocab.decode(hypothesis.ids[len(prefix):])
            text = format_completion(prompt,hypothesis.ids,len(prefix),vocab)
            new_count = len(hypothesis.ids)-len(prefix)
            full_words,new_words = lexical_tokens(text),lexical_tokens(continuation)
            results.append(dict(candidate_id=len(results),text=text,continuation=continuation,
                ids=hypothesis.ids,prompt=prompt,sources=[dict(strategy='copy_aware_beam',seed=seed,decoding=settings)],
                ended_with_eos=bool(hypothesis.ids[-1]==3),finish_reason=hypothesis.finish_reason,
                generated_token_count=new_count,new_word_count=len(new_words),
                repeated_trigrams=_repetitions(full_words),continuation_repeated_trigrams=_repetitions(new_words),
                log_probability=hypothesis.model_logp,
                average_log_probability=hypothesis.model_logp/new_count if new_count else None,
                search_lm_log_probability=hypothesis.search_lm_logp,
                raw_copy_cost=hypothesis.raw_copy_cost,copy_penalty_cost=copy_penalty*hypothesis.raw_copy_cost,
                copy_events=hypothesis.events,search_score=_score(hypothesis,copy_penalty),
                normalized_search_score=_rank(hypothesis,len(prefix),copy_penalty),
                equivalent_token_sequence_count=equivalent_count,
                novelty=index.novelty.assess(text,continuation)))
        return results
    finally:
        for module,training in modes: module.training = training
