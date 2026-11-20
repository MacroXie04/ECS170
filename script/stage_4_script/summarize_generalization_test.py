#!/usr/bin/env python3
"""Summarize locked test outputs and two independent assistant reviews."""
from __future__ import annotations
import json
import math
from pathlib import Path
from finalize_generalization import ROOT, BASE, read, sha, summarize
from local_code.stage_4_code.Experiment_Runner import write_json
from local_code.stage_4_code.Novel_Generation import select_candidate


def main():
    directory = BASE/'final_test'
    if (directory/'summary.json').exists(): raise FileExistsError('Final summary already exists')
    decision = read(BASE/'decision.json')
    for name, expected in decision['profile_sha256'].items():
        if sha(BASE/name) != expected: raise ValueError('Profile changed after selection')
    screening = read(BASE/'initial_spec.json')['screening']
    summaries, bindings, _ = summarize(directory,screening)
    ratings = read(directory/'confirmatory_ratings.json')
    inputs = read(directory/'blind_review.json')
    if len(ratings) != len(inputs) or {r['id'] for r in ratings} != {r['id'] for r in inputs}:
        raise ValueError('Confirmatory ratings must cover each item exactly once')
    if any(type(r['score']) is not int or r['score'] not in (0,1,2) for r in ratings):
        raise ValueError('Invalid confirmatory score')
    scores = {r['id']:r['score'] for r in ratings}
    same_text = {}
    for row in inputs:
        text = row['text'].strip().casefold()
        if text in same_text and same_text[text] != scores[row['id']]:
            raise ValueError('Identical text scored inconsistently in confirmatory review')
        same_text[text] = scores[row['id']]
    rows = read(directory/'outputs.json')
    first = {r['id']:r['score'] for r in read(directory/'coherence_ratings.json')}
    for summary in summaries:
        summary['test_loss'] = summary.pop('validation_loss')
        summary['test_perplexity'] = math.exp(summary['test_loss'])
        subset = [row for row in rows if row['candidate'] == summary['candidate']]
        summary['confirmatory_coherent_count'] = sum(scores[r['blind_id']]==2 for r in subset)
        summary['confirmatory_novel_coherent_count'] = sum(scores[r['blind_id']]==2 and r['result'] is not None and
            select_candidate([r['result']],**screening)['selected'] is not None for r in subset)
        summary['reviewer_score_disagreements'] = sum(scores[r['blind_id']] != first[r['blind_id']] for r in subset)
    bindings['confirmatory_ratings.json'] = sha(directory/'confirmatory_ratings.json')
    record = dict(summaries=summaries,review_bindings=bindings,decision_sha256=sha(BASE/'decision.json'),
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
                       'script/stage_4_script/finalize_generalization.py':sha(ROOT/'script/stage_4_script/finalize_generalization.py')},
        profile_selection_changed=False,limitations=['Two blinded assistant reviews, not a human humor study.',
            '16 prefixes with in-vocabulary words; small descriptive panel.',
            'No-copy behavior from screening is a policy outcome, not learned semantic generalization.'])
    write_json(directory/'summary.json',record)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__': main()
