#!/usr/bin/env python3
"""Bind blinded development ratings and freeze profiles before test evaluation."""
from __future__ import annotations
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Experiment_Runner import write_json
from local_code.stage_4_code.Novel_Generation import select_candidate
from predict_novel_generation import REQUIRED_SOURCES, training_digest

BASE = ROOT/'result/stage_4_result/generation_generalization'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(directory, screening):
    rows, blind, mapping, ratings = [read(directory/name) for name in
        ('outputs.json','blind_review.json','blind_mapping.json','coherence_ratings.json')]
    ids = {r['id'] for r in blind}
    if len(ids) != len(blind) or len(ratings) != len(ids) or {r['id'] for r in ratings} != ids:
        raise ValueError('Incomplete or duplicate blinded ratings')
    if any(type(r['score']) is not int or r['score'] not in (0,1,2) for r in ratings):
        raise ValueError('Invalid score')
    score = {r['id']:r['score'] for r in ratings}
    by_id = {r['id']:r for r in mapping}
    if set(by_id) != ids or {r['blind_id'] for r in rows} != ids:
        raise ValueError('Blinded mapping coverage mismatch')
    blind_by_id = {r['id']:r for r in blind}
    totals = defaultdict(list)
    consistency = {}
    for row in rows:
        identifier = row['blind_id']
        if any(row[k] != by_id[identifier][k] for k in ('candidate','prompt_index')) or row['text'] != blind_by_id[identifier]['text']:
            raise ValueError('Blinded mapping text/identity mismatch')
        text = row['text'].strip().casefold()
        if text in consistency and consistency[text] != score[identifier]:
            raise ValueError('Identical generated text has inconsistent ratings')
        consistency[text] = score[identifier]
        row['coherence_score'] = score[identifier]
        totals[row['candidate']].append(row)
    losses = {r['model']:r['loss']['loss'] for r in read(directory/'metrics.json')}
    summaries = []
    for name, subset in totals.items():
        result = dict(candidate=name, panel_size=len(subset), coherence_score_counts={str(i):0 for i in (0,1,2)},
            coherent_count=0, novel_coherent_count=0, exact_training_copies=0, copied_continuations=0,
            high_similarity_outputs=0, novelty_eligible_count=0, output_count=0, seen_coherent=0, unseen_coherent=0,
            validation_loss=losses[subset[0]['model_spec']['name']], model_spec=subset[0]['model_spec'], mode=name.rsplit('_',1)[1])
        for row in subset:
            coherent = row['coherence_score'] == 2
            result['coherence_score_counts'][str(row['coherence_score'])] += 1
            result['coherent_count'] += coherent
            result['seen_coherent' if row['seen_prefix'] else 'unseen_coherent'] += coherent
            candidate = row['result']
            if candidate is None: continue
            result['output_count'] += 1
            full, tail = candidate['novelty']['full_text'], candidate['novelty']['continuation']
            result['exact_training_copies'] += full['exact_copy']
            result['copied_continuations'] += tail['exact_copy'] or tail['exact_training_span']
            result['high_similarity_outputs'] += (full['exact_copy'] or tail['exact_copy'] or tail['exact_training_span'] or
                tail['nearest_similarity'] > screening['max_similarity'] or tail['shared_fraction'] > screening['max_similarity'])
            eligible = select_candidate([candidate], **screening)['selected'] is not None
            result['novelty_eligible_count'] += eligible
            result['novel_coherent_count'] += coherent and eligible
        summaries.append(result)
    binding = {name:sha(directory/name) for name in ('outputs.json','blind_review.json','blind_mapping.json','coherence_ratings.json','metrics.json','provenance.json')}
    return summaries, binding, consistency


def main():
    if (BASE/'decision.json').exists(): raise FileExistsError('Selection already locked')
    spec = read(BASE/'initial_spec.json')
    summaries, bindings, texts = [], {}, {}
    for name in ('initial_comparison','regularized_comparison','tied_mixture_comparison'):
        values, binding, consistency = summarize(BASE/name,spec['screening'])
        for text, score in consistency.items():
            if text in texts and texts[text] != score: raise ValueError('Cross-comparison identical text rated inconsistently')
            texts[text] = score
        summaries.extend(values); bindings[name] = binding
    previous = next(r for r in summaries if r['candidate'] == 'previous_late_beam')
    best = min(summaries,key=lambda r:(-r['novel_coherent_count'],-r['coherent_count'],r['validation_loss'],r['candidate']))
    promote = best['novel_coherent_count'] > previous['novel_coherent_count'] and best['coherent_count'] >= previous['coherent_count']-2
    selected = best if promote else previous
    regularized = min((r for r in summaries if r['candidate'].startswith('tied') and r['mode']=='beam'),key=lambda r:r['validation_loss'])
    strict = min((r for r in summaries if r['mode']=='screened'),key=lambda r:(-r['novel_coherent_count'],-r['coherent_count'],r['validation_loss']))
    data = Dataset_Loader(ROOT/'data/stage_4_data',seed=1704).load_generation()
    data_binding = {k:data['metadata'][k] for k in ('source_sha256','vocabulary_sha256','split_id_sha256')}
    data_binding['training_rows_sha256'] = training_digest(data['train'])
    profile_files = {}
    for filename, row, status in [('recommendation.json',selected,'promoted' if promote else 'previous_default_retained'),
        ('recommendation_predictive.json',regularized,'experimental_best_validation_loss'),
        ('recommendation_strict.json',strict,'experimental_lexical_screen_not_semantic_guarantee')]:
        model_spec = json.loads(json.dumps(row['model_spec']))
        for component in model_spec['components']:
            component['checkpoint_sha256'] = sha(ROOT/component['checkpoint'])
        sources = set(REQUIRED_SOURCES)
        if len(model_spec['components']) == 2: sources.add('local_code/stage_4_code/Generation_Ensemble.py')
        if any(c.get('loader') == 'regularized' for c in model_spec['components']): sources.add('local_code/stage_4_code/Regularized_Generation.py')
        profile = dict(schema_version=1,candidate=row['candidate'],status=status,mode=row['mode'],model_spec=model_spec,
            data_binding=data_binding,source_sha256={name:sha(ROOT/name) for name in sorted(sources)},
            sampling=spec['sampling'],screening=spec['screening'],selection_rule=spec['selection_rule'],
            dev_evidence=row,limitations=['Assistant coherence ratings, not a human humor study.',
                'Lexical novelty screening is not proof of semantic originality.',
                'Small manually stratified development panel; final test does not change this selection.'])
        write_json(BASE/filename,profile); profile_files[filename] = sha(BASE/filename)
    decision = dict(locked_at_utc=datetime.now(timezone.utc).isoformat(),promoted=promote,
        selected=selected['candidate'],best_joint_candidate=best['candidate'],previous=previous['candidate'],
        selection_rule=spec['selection_rule'],review_bindings=bindings,profile_sha256=profile_files,
        finalizer_sha256=sha(__file__),test_evaluated_at_selection=False,
        note='A failed joint coherence/novelty threshold retains the previous default; experimental profiles remain explicit opt-ins.')
    write_json(BASE/'dev_summary.json',summaries)
    write_json(BASE/'decision.json',decision)
    print(json.dumps(decision,indent=2))


if __name__ == '__main__': main()
