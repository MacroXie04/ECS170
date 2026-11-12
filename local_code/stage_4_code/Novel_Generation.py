"""Generate and screen original LM continuations against the training split.

This module never retrieves a replacement joke. Lexical novelty is an auditable
screen for memorization, not evidence of semantic coherence or generalization.
Callers must supply the exact TRAIN partition used to fit the checkpoint.
"""
from __future__ import annotations

import math
from collections import Counter
from difflib import SequenceMatcher

from local_code.stage_4_code.Dataset_Loader import tokenize
from local_code.stage_4_code.Generation_Decoder import decode
from local_code.stage_4_code.Generation_Optimization import format_completion, validate_prefix


def lexical_tokens(text):
    """Case-folded word tokens, discarding punctuation (including apostrophes)."""
    return tuple(word for token in tokenize(text)
                 if (word := ''.join(c for c in token.casefold() if c.isalnum())))


def _repeated_trigrams(tokens):
    counts = Counter(zip(tokens, tokens[1:], tokens[2:]))
    return sum(count - 1 for count in counts.values())


class TrainingNoveltyIndex:
    """Immutable lexical snapshots of caller-supplied training records only.

    Continuation similarity compares with each training joke after its first
    three lexical words. Shared-span checks additionally compare against the
    entire training joke, so changing a prompt cannot conceal a copied ending.
    No validation/test examples are added to this index.
    """

    def __init__(self, training_records):
        rows = []
        for record in training_records:
            if record.get('split', 'train') != 'train':
                raise ValueError('The novelty index accepts TRAIN records only')
            if 'id' not in record or not isinstance(record.get('text'), str):
                raise ValueError('Every training record needs an id and text string')
            words = lexical_tokens(record['text'])
            if words:
                rows.append((str(record['id']), words, words[3:]))
        if not rows:
            raise ValueError('Training novelty checks require nonempty TRAIN text')
        self.rows = tuple(rows)

    def _compare(self, words, continuation=False):
        result = dict(exact_copy=False, exact_match_ids=[], nearest_similarity=0.,
                      nearest_match_ids=[], max_shared_token_count=0,
                      shared_fraction=0., shared_match_ids=[])
        if continuation:
            result.update(exact_training_span=False, exact_span_match_ids=[],
                          exact_training_suffix=False, exact_suffix_match_ids=[])
        if not words:
            return result
        for record_id, full, tail in self.rows:
            comparison = tail if continuation else full
            similarity = SequenceMatcher(None, words, comparison, autojunk=False).ratio()
            if words == comparison:
                result['exact_match_ids'].append(record_id)
            if similarity > result['nearest_similarity']:
                result['nearest_similarity'] = similarity
                result['nearest_match_ids'] = [record_id]
            elif similarity == result['nearest_similarity'] and similarity > 0:
                result['nearest_match_ids'].append(record_id)
            shared = SequenceMatcher(None, words, full, autojunk=False).find_longest_match().size
            if shared > result['max_shared_token_count']:
                result['max_shared_token_count'] = shared
                result['shared_match_ids'] = [record_id]
            elif shared == result['max_shared_token_count'] and shared > 0:
                result['shared_match_ids'].append(record_id)
            if continuation:
                if shared == len(words):
                    result['exact_span_match_ids'].append(record_id)
                if len(words) <= len(full) and words == full[-len(words):]:
                    result['exact_suffix_match_ids'].append(record_id)
        result['exact_copy'] = bool(result['exact_match_ids'])
        result['shared_fraction'] = result['max_shared_token_count'] / len(words)
        if continuation:
            result['exact_training_span'] = bool(result['exact_span_match_ids'])
            result['exact_training_suffix'] = bool(result['exact_suffix_match_ids'])
        return result

    def assess(self, text, continuation):
        """Return JSON-safe diagnostics without modifying any supplied records."""
        return dict(training_record_count=len(self.rows),
                    full_text=self._compare(lexical_tokens(text)),
                    continuation=self._compare(lexical_tokens(continuation), True))


def generate_candidates(model, vocab, prompt, training_records, *, num_samples=16,
                        seed=1704, temperature=.8, top_p=.9, top_k=40,
                        max_new_tokens=48):
    """Return one beam result plus ``num_samples`` private-RNG nucleus draws.

    Identical token sequences are merged, retaining every strategy/seed in
    ``sources``. Prefix likelihood is excluded; mean log probability divides
    the original LM continuation log probability by all generated tokens,
    including punctuation and EOS. No candidate is silently dropped for failing
    a novelty/length/completion check. ``new_word_count`` counts lexical words,
    not a linguistic estimate of content words or semantic quality.
    """
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 0:
        raise ValueError('num_samples must be a nonnegative integer')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed + num_samples >= 2**63:
        raise ValueError('seed and derived seeds must be nonnegative signed 64-bit integers')
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError('max_new_tokens must be a nonnegative integer')
    prefix = validate_prefix(prompt, vocab)
    index = training_records if isinstance(training_records, TrainingNoveltyIndex) else TrainingNoveltyIndex(training_records)
    candidates = {}
    runs = [('beam', seed)] + [('sample', seed + i + 1) for i in range(num_samples)]
    for strategy, run_seed in runs:
        decoded = decode(model, prefix, strategy=strategy, max_new_tokens=max_new_tokens,
                         min_new_tokens=min(4, max_new_tokens), beam_width=5,
                         temperature=temperature, top_p=top_p, top_k=top_k, seed=run_seed)
        key = tuple(decoded['ids'])
        source = dict(strategy=strategy, seed=run_seed, decoding=decoded['decoding'])
        if key in candidates:
            candidates[key]['sources'].append(source)
            continue
        continuation = vocab.decode(decoded['ids'][len(prefix):])
        text = format_completion(prompt, decoded['ids'], len(prefix), vocab)
        generated_count = len(decoded['ids']) - len(prefix)
        full_words, new_words = lexical_tokens(text), lexical_tokens(continuation)
        candidates[key] = dict(candidate_id=len(candidates), text=text, continuation=continuation,
                               ids=list(key), prompt=prompt, sources=[source],
                               ended_with_eos=decoded['ended_with_eos'],
                               finish_reason=decoded['finish_reason'],
                               generated_token_count=generated_count,
                               new_word_count=len(new_words),
                               repeated_trigrams=_repeated_trigrams(full_words),
                               continuation_repeated_trigrams=_repeated_trigrams(new_words),
                               log_probability=decoded['log_probability'],
                               average_log_probability=(decoded['log_probability'] / generated_count
                                                        if generated_count else None),
                               novelty=index.assess(text, continuation))
    return list(candidates.values())


def select_candidate(candidates, max_similarity=.80, min_content_words=5):
    """Select the highest LM mean-log-probability eligible continuation.

    Eligibility requires EOS, enough new lexical words, no repeated lexical
    trigram, no exact full/continuation/span copy, and continuation similarity
    AND longest shared-span fraction <= ``max_similarity``. Full-text similarity
    is reported but does not inflate selection scores from a common prefix.
    ``min_content_words`` is a legacy-friendly name for lexical word count;
    function words are counted. This is not a semantic coherence classifier.
    Failure returns ``selected=None``; callers must not treat a rejected output
    as a success. Neither input candidates nor their ordering are changed.
    """
    if not math.isfinite(max_similarity) or not 0 <= max_similarity <= 1:
        raise ValueError('max_similarity must be finite and in [0,1]')
    if isinstance(min_content_words, bool) or not isinstance(min_content_words, int) or min_content_words < 1:
        raise ValueError('min_content_words must be a positive integer')
    eligible, rejected = [], []
    for candidate in candidates:
        reasons = []
        full, continuation = candidate['novelty']['full_text'], candidate['novelty']['continuation']
        if not candidate['ended_with_eos']:
            reasons.append('incomplete')
        if candidate['new_word_count'] < min_content_words:
            reasons.append('too_few_new_words')
        if candidate['repeated_trigrams']:
            reasons.append('repeated_trigrams')
        if full['exact_copy']:
            reasons.append('exact_training_copy')
        if continuation['exact_copy'] or continuation['exact_training_span']:
            reasons.append('copied_training_continuation')
        if continuation['nearest_similarity'] > max_similarity:
            reasons.append('continuation_similarity')
        if continuation['shared_fraction'] > max_similarity:
            reasons.append('shared_training_span')
        score = candidate['average_log_probability']
        if score is None or not math.isfinite(score):
            reasons.append('invalid_model_probability')
        if reasons:
            rejected.append(dict(candidate_id=candidate['candidate_id'], reasons=reasons))
        else:
            eligible.append(candidate)
    # Stable sorting makes the generation order the tie breaker, with no extra RNG.
    ranked = sorted(eligible, key=lambda candidate: candidate['average_log_probability'], reverse=True)
    return dict(selected=ranked[0] if ranked else None,
                status='selected' if ranked else 'no_eligible_candidate',
                eligible_count=len(ranked), candidate_count=len(eligible) + len(rejected),
                eligible_candidate_ids=[candidate['candidate_id'] for candidate in ranked],
                rejections=rejected,
                constraints=dict(max_similarity=max_similarity, min_content_words=min_content_words))
