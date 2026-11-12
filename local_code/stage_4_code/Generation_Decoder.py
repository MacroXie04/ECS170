"""Constrained recurrent decoding, without templates or retrieval fallbacks.

Beam search retains separate completed hypotheses and length-normalizes their
scores. The returned original-model log probability is kept separate from the
search score affected by repetition penalties. Neither value measures humor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


PAD, UNK, BOS, EOS = 0, 1, 2, 3


@dataclass
class _Beam:
    ids: list[int]
    model_logp: float
    search_logp: float
    hidden: object = None
    logits: torch.Tensor | None = None


def _clone_hidden(hidden):
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return tuple(_clone_hidden(value) for value in hidden)
    if isinstance(hidden, list):
        return [_clone_hidden(value) for value in hidden]
    if isinstance(hidden, torch.Tensor):
        return hidden.clone()
    raise TypeError('Recurrent hidden state must be tensors, a tuple/list, or None')


def _banned_ngram_tokens(ids, size):
    if size == 0 or len(ids) + 1 < size:
        return set()
    if size == 1:
        return set(ids)
    tail = tuple(ids[-(size - 1):])
    return {ids[index + size - 1] for index in range(len(ids) - size + 1)
            if tuple(ids[index:index + size - 1]) == tail}


def _scores(logits, ids, generated, minimum, ngram_size, penalty, temperature):
    # Compute probabilities in float64 on CPU. This also isolates sampling from
    # the GPU/global RNG; hidden-state/model inference still uses its own device.
    raw = logits.detach().reshape(-1).double().cpu()
    if torch.isnan(raw).any() or torch.isposinf(raw).any() or not torch.isfinite(raw).any():
        raise FloatingPointError('Model returned no finite logits or invalid positive infinity/NaN')
    original = raw.log_softmax(-1)
    adjusted = raw.clone()
    if penalty != 1:
        seen = sorted({token for token in ids if token >= EOS + 1})
        if seen:
            values = adjusted[seen]
            adjusted[seen] = torch.where(values < 0, values * penalty, values / penalty)
    search = (adjusted / temperature).log_softmax(-1)
    search[:3] = -torch.inf
    if generated < minimum:
        search[EOS] = -torch.inf
    blocked = _banned_ngram_tokens(ids, ngram_size)
    if blocked:
        search[list(blocked)] = -torch.inf
    return original, search


def _length_norm(length, alpha):
    return ((5 + max(1, length)) / 6) ** alpha


def _rank(beam, prefix_length, alpha):
    return beam.search_logp / _length_norm(len(beam.ids) - prefix_length, alpha)


def _nucleus_distribution(scores, top_k, top_p):
    values, indices = torch.sort(scores, descending=True, stable=True)
    eligible = torch.isfinite(values)
    values, indices = values[eligible], indices[eligible]
    if top_k:
        values, indices = values[:top_k], indices[:top_k]
    probabilities = values.softmax(-1)
    # Retain the token that crosses the nucleus threshold, always at least one.
    keep = probabilities.cumsum(-1) - probabilities < top_p
    probabilities, indices = probabilities[keep], indices[keep]
    probabilities = probabilities / probabilities.sum()
    return probabilities, indices


def _result(beam, prefix_length, strategy, alpha, reason, parameters):
    return {
        'ids': list(beam.ids),
        'ended_with_eos': beam.ids[-1] == EOS,
        'generated_token_count': len(beam.ids) - prefix_length,
        'log_probability': float(beam.model_logp),
        'search_log_probability': float(beam.search_logp),
        'normalized_score': float(_rank(beam, prefix_length, alpha)),
        'strategy': strategy,
        'finish_reason': reason,
        'decoding': parameters,
    }


@torch.no_grad()
def decode(model, prefix_ids, strategy='beam', max_new_tokens=48, min_new_tokens=4,
           beam_width=5, length_penalty=.7, temperature=.7, top_k=40, top_p=.9,
           no_repeat_ngram_size=3, repetition_penalty=1.1, seed=1704):
    """Continue token IDs with beam search, nucleus sampling, or greedy decoding.

    ``min_new_tokens`` counts continuation tokens before EOS is permitted.
    ``max_new_tokens`` counts every emitted token including EOS. EOS is never
    fabricated at the limit: an unfinished continuation is labeled explicitly.
    PAD/UNK/BOS cannot be generated, and repeated n-grams are blocked across the
    supplied prefix and continuation. Impossible constraints yield an explicitly
    incomplete result rather than silently relaxing the constraints.

    Temperature/top-k/top-p affect sampling only. Beam/greedy use temperature 1;
    beam ranks accumulated adjusted log probabilities using the GNMT length
    penalty ((5 + emitted_tokens) / 6) ** length_penalty. Repetition penalties
    alter logits before log-softmax, and constraints mask tokens afterward.
    Returned ``log_probability`` sums the original, unmodified model's next-token
    log probabilities for emitted tokens only, excluding the prefix.

    Sampling uses its own CPU generator. The model's train/eval mode is restored.
    ``model.step`` must accept token matrices [1,T] and return logits [1,T,V]
    plus a tensor or tuple/list of tensors for recurrent hidden state.
    """
    if getattr(model, 'task', 'generation') != 'generation':
        raise ValueError('decode requires a generation model')
    if strategy not in ('beam', 'sample', 'greedy'):
        raise ValueError('strategy must be beam, sample, or greedy')
    integers = {'max_new_tokens': max_new_tokens, 'min_new_tokens': min_new_tokens,
                'beam_width': beam_width, 'top_k': top_k,
                'no_repeat_ngram_size': no_repeat_ngram_size}
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers.values()):
        raise ValueError('Token limits, beam width, top-k and n-gram size must be integers')
    if max_new_tokens < 0 or not 0 <= min_new_tokens <= max_new_tokens:
        raise ValueError('Require 0 <= min_new_tokens <= max_new_tokens')
    if beam_width < 1 or top_k < 0 or no_repeat_ngram_size < 0:
        raise ValueError('Require beam_width >= 1, top_k >= 0, n-gram size >= 0')
    if not math.isfinite(length_penalty) or length_penalty < 0:
        raise ValueError('length_penalty must be finite and nonnegative')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive; use greedy for argmax')
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('top_p must be in (0,1]')
    if not math.isfinite(repetition_penalty) or repetition_penalty < 1:
        raise ValueError('repetition_penalty must be finite and at least 1')
    prefix = list(prefix_ids)
    if not prefix or any(isinstance(token, bool) or not isinstance(token, int) for token in prefix):
        raise ValueError('prefix_ids must be a nonempty sequence of integer token IDs')
    vocabulary_size = int(model.vocab_size)
    if vocabulary_size < 5 or any(token < 0 or token >= vocabulary_size for token in prefix):
        raise ValueError('Prefix token outside a vocabulary containing all four special tokens')
    if EOS in prefix[:-1] or PAD in prefix or UNK in prefix or BOS in prefix[1:]:
        raise ValueError('Prefix may start with BOS and end with EOS, but cannot contain PAD/UNK/internal boundaries')
    parameters = dict(max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
        beam_width=beam_width, length_penalty=length_penalty, temperature=temperature,
        top_k=top_k, top_p=top_p, no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty, seed=seed,
        temperature_applies_to='sample only')
    prefix_length = len(prefix)
    empty = _Beam(prefix, 0., 0.)
    if prefix[-1] == EOS or max_new_tokens == 0:
        return _result(empty, prefix_length, strategy, length_penalty,
                       'prefix_eos' if prefix[-1] == EOS else 'max_new_tokens', parameters)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        logits, hidden = model.step(torch.tensor([prefix], dtype=torch.long, device=device))
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != vocabulary_size:
            raise ValueError('model.step must return logits shaped [1,T,vocab_size]')
        initial = _Beam(prefix, 0., 0., hidden, logits[0, -1])
        if strategy != 'beam':
            current = initial
            reason = 'max_new_tokens'
            for emitted in range(max_new_tokens):
                original, scores = _scores(current.logits, current.ids, emitted,
                    min_new_tokens, no_repeat_ngram_size, repetition_penalty,
                    temperature if strategy == 'sample' else 1.)
                if not torch.isfinite(scores).any():
                    reason = 'constraints_exhausted'
                    break
                if strategy == 'greedy':
                    chosen = int(scores.argmax())
                else:
                    probabilities, indices = _nucleus_distribution(scores, top_k, top_p)
                    chosen = int(indices[torch.multinomial(probabilities, 1, generator=generator).item()])
                current = _Beam(current.ids + [chosen], current.model_logp + float(original[chosen]),
                                current.search_logp + float(scores[chosen]), current.hidden)
                if chosen == EOS:
                    reason = 'eos'
                    break
                if emitted + 1 < max_new_tokens:
                    output, state = model.step(torch.tensor([[chosen]], dtype=torch.long, device=device),
                                               _clone_hidden(current.hidden))
                    current.logits, current.hidden = output[0, -1], state
            return _result(current, prefix_length, strategy, length_penalty, reason, parameters)

        active, finished, stalled = [initial], [], []
        for emitted in range(max_new_tokens):
            candidates = []
            for beam in active:
                original, scores = _scores(beam.logits, beam.ids, emitted, min_new_tokens,
                    no_repeat_ngram_size, repetition_penalty, 1.)
                if torch.isfinite(scores[EOS]):
                    finished.append(_Beam(beam.ids + [EOS], beam.model_logp + float(original[EOS]),
                                          beam.search_logp + float(scores[EOS])))
                scores[EOS] = -torch.inf
                finite = int(torch.isfinite(scores).sum())
                if finite == 0:
                    stalled.append(beam)
                    continue
                values, indices = torch.topk(scores, min(beam_width, finite))
                for value, index in zip(values.tolist(), indices.tolist()):
                    candidates.append(_Beam(beam.ids + [index], beam.model_logp + float(original[index]),
                                            beam.search_logp + value, beam.hidden))
            finished.sort(key=lambda beam: _rank(beam, prefix_length, length_penalty), reverse=True)
            finished = finished[:beam_width]
            candidates.sort(key=lambda beam: (beam.search_logp, tuple(-x for x in beam.ids)), reverse=True)
            active = candidates[:beam_width]
            if not active:
                break
            if emitted + 1 < max_new_tokens:
                for beam in active:
                    output, state = model.step(torch.tensor([[beam.ids[-1]]], dtype=torch.long, device=device),
                                               _clone_hidden(beam.hidden))
                    beam.logits, beam.hidden = output[0, -1], state
        # Prefer a completed hypothesis to an unfinished fragment. The metadata
        # distinguishes the latter when the budget/constraints forbid completion.
        pool = finished or active or stalled
        chosen = max(pool, key=lambda beam: _rank(beam, prefix_length, length_penalty))
        reason = 'eos' if finished else ('max_new_tokens' if active else 'constraints_exhausted')
        return _result(chosen, prefix_length, strategy, length_penalty, reason, parameters)
    finally:
        model.train(was_training)
