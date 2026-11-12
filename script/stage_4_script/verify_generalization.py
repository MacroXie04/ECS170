#!/usr/bin/env python3
"""Audit frozen generalization comparisons without displaying generated text.

Uses current TRAIN records only for novelty checks. Replays plain split NLL
and candidate teacher-forced likelihoods on CPU; it does not resample outputs
or assess their semantic quality. Missing/incomplete jobs are reported as skips.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader, tokenize
from local_code.stage_4_code.Experiment_Runner import collate, run_epoch
from local_code.stage_4_code.Generation_Ensemble import MixtureModel
from local_code.stage_4_code.Generation_Optimization import (
    load_generation_checkpoint, format_completion, validate_prefix,
)
from local_code.stage_4_code.Novel_Generation import (
    TrainingNoveltyIndex, lexical_tokens, select_candidate,
)

BASE = ROOT / 'result/stage_4_result/generation_generalization'
COMPARISON_SOURCES = {
    'script/stage_4_script/compare_generalization.py',
    'script/stage_4_script/compare_generation.py',
    'local_code/stage_4_code/Novel_Generation.py',
    'local_code/stage_4_code/Generation_Decoder.py',
    'local_code/stage_4_code/Generation_Optimization.py',
    'local_code/stage_4_code/Generation_Ensemble.py',
    'local_code/stage_4_code/Dataset_Loader.py',
}
COMPLETION_FILES = ('provenance.json', 'outputs.json', 'candidate_pools.json',
                    'metrics.json', 'blind_review.json', 'blind_mapping.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def rows_digest(rows):
    return canonical_digest([dict(id=row['id'], text=row['text']) for row in rows])


def read_json(path):
    return json.loads(Path(path).read_text())


def repo_path(name):
    path = (ROOT / name).resolve()
    require(path.is_relative_to(ROOT), 'Bound path escapes repository')
    return path


def verify_sources(bindings, required=()):
    require(isinstance(bindings, dict) and set(required) <= set(bindings),
            'Missing required source hash bindings')
    for name, expected in bindings.items():
        require(digest(repo_path(name)) == expected, 'Source hash mismatch: ' + name)
    return len(bindings)


def first_three(text):
    result, words = [], 0
    for token in tokenize(text):
        result.append(token)
        words += any(character.isalnum() for character in token)
        if words == 3:
            return result
    return result


def verify_panel(manifest, data, split):
    prompts = manifest['prompts']
    prompt_hash = hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
    require(prompt_hash == manifest['sha256'], 'Panel prompt hash mismatch')
    require(manifest['vocabulary_sha256'] == data['metadata']['vocabulary_sha256'],
            'Panel vocabulary mismatch')
    require(manifest['train_rows_sha256'] == rows_digest(data['train']), 'Panel TRAIN rows mismatch')
    require(manifest[split + '_rows_sha256'] == rows_digest(data[split]), 'Panel evaluation rows mismatch')
    verify_sources(manifest['source_sha256'], {'local_code/stage_4_code/Dataset_Loader.py'})
    for name in ('train', 'validation', 'test'):
        if name + '_rows_sha256' in manifest:
            require(manifest[name + '_rows_sha256'] == rows_digest(data[name]), 'Panel row hash mismatch: ' + name)
    by_id = {row['id']: row for row in data[split]}
    require(len({row['id'] for row in prompts}) == len(prompts), 'Duplicate panel IDs')
    require(len({row['prompt'] for row in prompts}) == len(prompts), 'Duplicate panel prompts')
    for item in prompts:
        require(item['source_split'] == split, 'Panel source split mismatch')
        require(item['id'] in by_id, 'Panel ID absent from evaluation split')
        validate_prefix(item['prompt'], data['vocab'])
        require(first_three(by_id[item['id']]['text']) == tokenize(item['prompt']), 'Panel prefix mismatch')
        count = sum(first_three(row['text']) == tokenize(item['prompt']) for row in data['train'])
        require(item['train_prefix_count'] == count, 'Panel train-prefix count mismatch')
        require(item['prefix_seen_in_train'] == bool(count), 'Panel seen-prefix flag mismatch')
        if 'stratum' in item:
            require(item['stratum'] == ('seen' if count else 'unseen'), 'Panel stratum mismatch')
    return prompts


def load_bound_model(spec, metrics, data):
    components, bindings = [], []
    require(len(spec['components']) in (1, 2), 'Expected one or two components')
    require(len(spec['components']) == len(metrics['components']), 'Component count mismatch')
    for part, recorded in zip(spec['components'], metrics['components']):
        require(part['checkpoint'] == recorded['checkpoint'], 'Component checkpoint path mismatch')
        checkpoint = repo_path(part['checkpoint'])
        actual_hash = digest(checkpoint)
        require(actual_hash == recorded['checkpoint_sha256'], 'Recorded checkpoint hash mismatch')
        if 'checkpoint_sha256' in part:
            require(actual_hash == part['checkpoint_sha256'], 'Spec checkpoint hash mismatch')
        loader_name = part.get('loader', 'original')
        require(loader_name in ('original', 'regularized'), 'Unknown checkpoint loader')
        if loader_name == 'regularized':
            from local_code.stage_4_code.Regularized_Generation import load_regularized_checkpoint
            loader = load_regularized_checkpoint
        else:
            loader = load_generation_checkpoint
        model, vocab, meta = loader(checkpoint, 'cpu')
        require(meta['epoch'] == recorded['epoch'], 'Recorded checkpoint epoch mismatch')
        require(vocab.to_dict() == data['vocab'].to_dict(), 'Checkpoint vocabulary mismatch')
        require(meta['config']['data_metadata'] == data['metadata'], 'Checkpoint training dataset/split mismatch')
        verify_sources(meta['config']['source_sha256'], {
            'local_code/stage_4_code/Dataset_Loader.py',
            'local_code/stage_4_code/Method_RNN.py',
            'local_code/stage_4_code/Experiment_Runner.py',
        })
        bindings.append(dict(checkpoint=part['checkpoint'], checkpoint_sha256=actual_hash,
                             config_sha256=meta['config_hash'], epoch=meta['epoch']))
        components.append(model)
    model = components[0] if len(components) == 1 else MixtureModel(components, late_weight=spec['late_weight'])
    return model.eval(), bindings


def repeated_trigrams(words):
    return sum(count - 1 for count in Counter(zip(words, words[1:], words[2:])).values())


def check_number(actual, expected, label, tolerance=2e-5):
    require(math.isfinite(actual) and math.isfinite(expected) and
            math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), label)


@torch.no_grad()
def verify_likelihoods(model, candidates, prefix_size):
    maximum_error = 0.
    for start in range(0, len(candidates), 32):
        batch = candidates[start:start + 32]
        sequences = [torch.tensor(row['ids'], dtype=torch.long) for row in batch]
        tokens = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        logits = model(tokens[:, :-1]).double().log_softmax(-1)
        chosen = logits.gather(-1, tokens[:, 1:, None]).squeeze(-1)
        for i, candidate in enumerate(batch):
            replay = float(chosen[i, prefix_size - 1:len(candidate['ids']) - 1].sum())
            recorded = candidate['log_probability']
            # CUDA incremental and CPU batched GRU kernels differ slightly.
            # Check their mean token log probabilities with an explicit bound;
            # retain the observed total discrepancy in the audit report.
            generated = len(candidate['ids']) - prefix_size
            require(math.isfinite(replay) and math.isfinite(recorded) and
                    abs(replay - recorded) <= 2e-4 * max(1, generated),
                    'Candidate original-model likelihood mismatch')
            maximum_error = max(maximum_error, abs(replay - recorded))
    return maximum_error


def verify_pool(pool, prompt, vocab, index, sampling):
    candidates = pool['candidates']
    require(bool(candidates), 'Empty candidate pool')
    prefix = validate_prefix(prompt, vocab)
    require(len({tuple(row['ids']) for row in candidates}) == len(candidates), 'Duplicate candidate token sequences')
    observed_sources = []
    base_seed = sampling.get('seed', 1704)
    num_samples = sampling.get('num_samples', 16)
    maximum = sampling.get('max_new_tokens', 48)
    for number, candidate in enumerate(candidates):
        ids = candidate['ids']
        require(candidate['candidate_id'] == number, 'Candidate ID/order mismatch')
        require(candidate['prompt'] == prompt and ids[:len(prefix)] == prefix, 'Candidate prefix mismatch')
        new = ids[len(prefix):]
        require(len(new) <= maximum, 'Candidate exceeds token budget')
        require(not any(token in (0, 1, 2) for token in new), 'Candidate generated blocked special token')
        require(3 not in new[:-1], 'Candidate contains internal EOS')
        ended = bool(new and new[-1] == 3)
        require(candidate['ended_with_eos'] == ended, 'Candidate EOS metadata mismatch')
        require((candidate['finish_reason'] == 'eos') == ended, 'Candidate finish reason mismatch')
        require(candidate['generated_token_count'] == len(new), 'Candidate generated-token count mismatch')
        continuation = vocab.decode(new)
        text = format_completion(prompt, ids, len(prefix), vocab)
        require(candidate['continuation'] == continuation and candidate['text'] == text, 'Candidate token/text mismatch')
        words = lexical_tokens(continuation)
        require(candidate['new_word_count'] == len(words), 'Candidate word count mismatch')
        require(candidate['repeated_trigrams'] == repeated_trigrams(lexical_tokens(text)), 'Candidate repetition mismatch')
        require(candidate['continuation_repeated_trigrams'] == repeated_trigrams(words), 'Continuation repetition mismatch')
        average = candidate['log_probability'] / len(new) if new else None
        require(candidate['average_log_probability'] == average, 'Candidate average probability mismatch')
        require(candidate['novelty'] == index.assess(text, continuation), 'Candidate TRAIN novelty mismatch')
        for source in candidate['sources']:
            observed_sources.append((source['strategy'], source['seed']))
            expected = dict(max_new_tokens=maximum, min_new_tokens=min(4, maximum), beam_width=5,
                length_penalty=.7, temperature=sampling.get('temperature', .8),
                top_k=sampling.get('top_k', 40), top_p=sampling.get('top_p', .9),
                no_repeat_ngram_size=3, repetition_penalty=1.1, seed=source['seed'],
                temperature_applies_to='sample only')
            require(source['decoding'] == expected, 'Recorded decoding settings mismatch')
    expected_sources = [('beam', base_seed)] + [('sample', base_seed + i + 1) for i in range(num_samples)]
    require(Counter(observed_sources) == Counter(expected_sources), 'Candidate source draws missing/duplicated')
    require(candidates[0]['sources'][0]['strategy'] == 'beam', 'First candidate is not beam')
    return len(prefix)


def verify_blind_files(output, rows):
    blind = read_json(output/'blind_review.json')
    mapping = read_json(output/'blind_mapping.json')
    order = list(range(len(rows)))
    random.Random(170431).shuffle(order)
    require(len(blind) == len(mapping) == len(rows), 'Blind-review row count mismatch')
    for number, index in enumerate(order, 1):
        row, identifier = rows[index], f'N{number:03d}'
        require(row['blind_id'] == identifier, 'Blind ID shuffle mismatch')
        require(blind[number - 1] == dict(id=identifier, prompt=row['prompt'], text=row['text'],
                                         no_output=not bool(row['text'])), 'Blind-review content mismatch')
        require(mapping[number - 1] == dict(id=identifier, candidate=row['candidate'],
                                           prompt_index=row['prompt_index']), 'Blind mapping mismatch')
    ratings_path = output/'coherence_ratings.json'
    if not ratings_path.exists():
        return dict(status='not_available')
    ratings = read_json(ratings_path)
    require(len(ratings) == len(rows), 'Rating count mismatch')
    require(len({item['id'] for item in ratings}) == len(ratings), 'Duplicate rating IDs')
    require({item['id'] for item in ratings} == {item['blind_id'] for item in rows}, 'Rating ID coverage mismatch')
    scores = {item['id']: item['score'] for item in ratings}
    require(all(type(score) is int and score in (0, 1, 2) for score in scores.values()), 'Invalid coherence score')
    require(all(scores[row['blind_id']] == 0 for row in rows if not row['text']), 'No-output received nonzero rating')
    return dict(status='coverage_verified', sha256=digest(ratings_path), count=len(ratings),
                note='Coverage/types only; semantic ratings are not independently verified here.')


def audit_comparison(output, spec_path, panel_path, data, replay_metrics=True):
    missing = [name for name in COMPLETION_FILES if not (output/name).exists()]
    if missing:
        return dict(comparison=str(output.relative_to(ROOT)), status='skipped_incomplete', missing=missing)
    provenance, spec, panel = read_json(output/'provenance.json'), read_json(spec_path), read_json(panel_path)
    require(digest(spec_path) == provenance['spec_sha256'], 'Comparison spec hash mismatch')
    require(digest(panel_path) == provenance['panel_sha256'], 'Comparison panel file hash mismatch')
    require(provenance['selection_rule'] == spec['selection_rule'], 'Selection rule mismatch')
    required_sources = set(COMPARISON_SOURCES)
    if any(part.get('loader') == 'regularized' for model in spec['models'] for part in model['components']):
        required_sources.add('local_code/stage_4_code/Regularized_Generation.py')
    verify_sources(provenance['source_sha256'], required_sources)
    require(provenance['data_source_sha256'] == data['metadata']['source_sha256'], 'Raw dataset hash mismatch')
    require(provenance['vocabulary_sha256'] == data['metadata']['vocabulary_sha256'], 'Comparison vocabulary mismatch')
    split = provenance['split']
    require(split in ('validation', 'test'), 'Invalid evaluation split')
    prompts = verify_panel(panel, data, split)
    index = TrainingNoveltyIndex(data['train'])
    pools, rows, metrics = (read_json(output/name) for name in ('candidate_pools.json', 'outputs.json', 'metrics.json'))
    require(len({model['name'] for model in spec['models']}) == len(spec['models']), 'Duplicate model names')
    require([m['model'] for m in metrics] == [s['name'] for s in spec['models']], 'Metric model order mismatch')
    expected_pools = [(s['name'], j) for s in spec['models'] for j in range(len(prompts))]
    require([(p['model'], p['prompt_index']) for p in pools] == expected_pools, 'Pool membership/order mismatch')
    expected_rows, checks, candidate_count, maximum_error = [], [], 0, 0.
    pool_by_key = {(p['model'], p['prompt_index']): p for p in pools}
    for model_spec, recorded_metrics in zip(spec['models'], metrics):
        model, bindings = load_bound_model(model_spec, recorded_metrics, data)
        metric_check = dict(status='not_replayed')
        if replay_metrics:
            replay = run_epoch(model, DataLoader(data[split], batch_size=128, shuffle=False,
                collate_fn=collate), 'generation', torch.device('cpu'))
            require(replay['count'] == recorded_metrics['loss']['count'], 'Evaluation token count mismatch')
            check_number(replay['loss'], recorded_metrics['loss']['loss'], 'Evaluation NLL mismatch')
            check_number(replay['accuracy'], recorded_metrics['loss']['accuracy'], 'Evaluation accuracy mismatch')
            metric_check = dict(status='cpu_replay_verified', recorded=recorded_metrics['loss'], replayed=replay)
        for j, prompt in enumerate(prompts):
            pool = pool_by_key[(model_spec['name'], j)]
            prefix_size = verify_pool(pool, prompt['prompt'], data['vocab'], index, spec['sampling'])
            candidate_count += len(pool['candidates'])
            maximum_error = max(maximum_error, verify_likelihoods(model, pool['candidates'], prefix_size))
            for mode in model_spec.get('modes', ['beam', 'screened']):
                require(mode in ('beam', 'screened'), 'Unknown comparison mode')
                selection = (select_candidate(pool['candidates'], **spec['screening']) if mode == 'screened'
                             else dict(selected=pool['candidates'][0], status='selected'))
                selected = selection.pop('selected')
                expected_rows.append(dict(candidate=model_spec['name']+'_'+mode, prompt_index=j,
                    prompt=prompt['prompt'], seen_prefix=prompt['prefix_seen_in_train'], selection=selection,
                    result=selected, text=selected['text'] if selected else '', model_spec=model_spec))
        checks.append(dict(model=model_spec['name'], components=bindings, metrics=metric_check))
        del model
    require(len(rows) == len(expected_rows), 'Selected-output row count mismatch')
    for row, expected in zip(rows, expected_rows):
        require({key: value for key, value in row.items() if key != 'blind_id'} == expected,
                'Selected output differs from fixed mode/screening rule')
    ratings = verify_blind_files(output, rows)
    return dict(comparison=str(output.relative_to(ROOT)), status='verified', split=split,
                prompt_count=len(prompts), model_count=len(spec['models']), candidate_count=candidate_count,
                output_count=len(rows), maximum_candidate_log_probability_error=maximum_error,
                candidate_log_probability_tolerance_per_generated_token=2e-4,
                artifacts_sha256={name: digest(output/name) for name in COMPLETION_FILES},
                panel_sha256=digest(panel_path), spec_sha256=digest(spec_path),
                train_rows_sha256=rows_digest(data['train']), evaluation_rows_sha256=rows_digest(data[split]),
                models=checks, ratings=ratings,
                limitations=['Seed/settings coverage checked; stochastic decoding was not regenerated.',
                             'Lexical novelty does not establish semantic originality or generalization.',
                             'Recorded artifact bindings verify consistency, not external authenticity.'])


def recompute_summary(directory, screening, consistency):
    rows = read_json(directory/'outputs.json')
    scores = {item['id']: item['score'] for item in read_json(directory/'coherence_ratings.json')}
    losses = {item['model']: item['loss']['loss'] for item in read_json(directory/'metrics.json')}
    groups = defaultdict(list)
    for row in rows:
        score = scores[row['blind_id']]
        text_key = row['text'].strip().casefold()
        require(text_key not in consistency or consistency[text_key] == score,
                'Identical text received inconsistent development ratings')
        consistency[text_key] = score
        groups[row['candidate']].append((row, score))
    summaries = []
    for name, entries in groups.items():
        row = entries[0][0]
        summary = dict(candidate=name, panel_size=len(entries),
            coherence_score_counts={str(i): 0 for i in (0, 1, 2)}, coherent_count=0,
            novel_coherent_count=0, exact_training_copies=0, copied_continuations=0,
            high_similarity_outputs=0, novelty_eligible_count=0, output_count=0,
            seen_coherent=0, unseen_coherent=0,
            validation_loss=losses[row['model_spec']['name']], model_spec=row['model_spec'],
            mode=name.rsplit('_', 1)[1])
        for row, score in entries:
            coherent = score == 2
            summary['coherence_score_counts'][str(score)] += 1
            summary['coherent_count'] += coherent
            summary['seen_coherent' if row['seen_prefix'] else 'unseen_coherent'] += coherent
            candidate = row['result']
            if candidate is None:
                continue
            summary['output_count'] += 1
            full, tail = candidate['novelty']['full_text'], candidate['novelty']['continuation']
            copied = tail['exact_copy'] or tail['exact_training_span']
            summary['exact_training_copies'] += full['exact_copy']
            summary['copied_continuations'] += copied
            summary['high_similarity_outputs'] += (full['exact_copy'] or copied or
                tail['nearest_similarity'] > screening['max_similarity'] or
                tail['shared_fraction'] > screening['max_similarity'])
            eligible = select_candidate([candidate], **screening)['selected'] is not None
            summary['novelty_eligible_count'] += eligible
            summary['novel_coherent_count'] += coherent and eligible
        summaries.append(summary)
    return summaries


def audit_decision(data):
    path = BASE/'decision.json'
    if not path.exists():
        return dict(status='not_available')
    decision, initial = read_json(path), read_json(BASE/'initial_spec.json')
    require(decision['selection_rule'] == initial['selection_rule'], 'Locked selection rule changed')
    require(decision['test_evaluated_at_selection'] is False, 'Decision does not assert test-free selection')
    require(decision['finalizer_sha256'] == digest(ROOT/'script/stage_4_script/finalize_generalization.py'),
            'Finalizer changed after decision lock')
    jobs = ('initial_comparison', 'regularized_comparison', 'tied_mixture_comparison')
    require(set(decision['review_bindings']) == set(jobs), 'Decision review set mismatch')
    summaries, consistency = [], {}
    bound_artifacts = {'outputs.json', 'blind_review.json', 'blind_mapping.json',
                       'coherence_ratings.json', 'metrics.json', 'provenance.json'}
    for job in jobs:
        bindings = decision['review_bindings'][job]
        require(set(bindings) == bound_artifacts, 'Missing decision evidence bindings')
        for name, expected in bindings.items():
            require(digest(BASE/job/name) == expected, 'Development evidence changed after lock: ' + job + '/' + name)
        summaries.extend(recompute_summary(BASE/job, initial['screening'], consistency))
    require(summaries == read_json(BASE/'dev_summary.json'), 'Development summary does not match blinded evidence')
    previous = next(row for row in summaries if row['candidate'] == 'previous_late_beam')
    best = min(summaries, key=lambda row: (-row['novel_coherent_count'], -row['coherent_count'],
                                          row['validation_loss'], row['candidate']))
    promoted = (best['novel_coherent_count'] > previous['novel_coherent_count'] and
                best['coherent_count'] >= previous['coherent_count'] - 2)
    selected = best if promoted else previous
    predictive = min((row for row in summaries if row['candidate'].startswith('tied') and row['mode'] == 'beam'),
                     key=lambda row: row['validation_loss'])
    strict = min((row for row in summaries if row['mode'] == 'screened'),
                 key=lambda row: (-row['novel_coherent_count'], -row['coherent_count'], row['validation_loss']))
    require(decision['promoted'] == promoted and decision['selected'] == selected['candidate'] and
            decision['best_joint_candidate'] == best['candidate'] and decision['previous'] == previous['candidate'],
            'Locked decision violates the fixed development selection criterion')
    expected_profiles = {
        'recommendation.json': (selected, 'promoted' if promoted else 'previous_default_retained'),
        'recommendation_predictive.json': (predictive, 'experimental_best_validation_loss'),
        'recommendation_strict.json': (strict, 'experimental_lexical_screen_not_semantic_guarantee'),
    }
    require(set(decision['profile_sha256']) == set(expected_profiles), 'Locked profile set mismatch')
    profiles = []
    for filename, (summary, status) in expected_profiles.items():
        profile = read_json(BASE/filename)
        require(digest(BASE/filename) == decision['profile_sha256'][filename], 'Profile changed after decision: ' + filename)
        require(profile['schema_version'] == 1 and profile['candidate'] == summary['candidate'] and
                profile['mode'] == summary['mode'] and profile['status'] == status and
                profile['dev_evidence'] == summary, 'Profile selection/evidence mismatch')
        require(profile['sampling'] == initial['sampling'] and profile['screening'] == initial['screening'] and
                profile['selection_rule'] == initial['selection_rule'], 'Profile decoding/selection rule mismatch')
        require(profile['data_binding'] == dict(
            source_sha256=data['metadata']['source_sha256'],
            vocabulary_sha256=data['metadata']['vocabulary_sha256'],
            split_id_sha256=data['metadata']['split_id_sha256'], training_rows_sha256=rows_digest(data['train'])),
            'Profile data/TRAIN reference binding mismatch')
        required = (COMPARISON_SOURCES - {'local_code/stage_4_code/Generation_Ensemble.py'}) | {
            'script/stage_4_script/predict_novel_generation.py', 'local_code/stage_4_code/Method_RNN.py'}
        components = profile['model_spec']['components']
        if len(components) == 2:
            required.add('local_code/stage_4_code/Generation_Ensemble.py')
        if any(part.get('loader') == 'regularized' for part in components):
            required.add('local_code/stage_4_code/Regularized_Generation.py')
        verify_sources(profile['source_sha256'], required)
        bare_spec = json.loads(json.dumps(profile['model_spec']))
        for part in bare_spec['components']:
            require(part.pop('checkpoint_sha256') == digest(repo_path(part['checkpoint'])), 'Profile checkpoint hash mismatch')
        require(bare_spec == summary['model_spec'], 'Profile model differs from selected development system')
        profiles.append(profile)
    if (BASE/'test_panel.json').exists():
        panel = read_json(BASE/'test_panel.json')
        final = read_json(BASE/'final_spec.json')
        require(panel['decision_sha256'] == final['decision_sha256'] == digest(path), 'Test decision lock mismatch')
        require(datetime.fromisoformat(decision['locked_at_utc']) < datetime.fromisoformat(panel['frozen_at_utc']),
                'Test panel was not frozen after decision lock')
        require(final['profile_sha256'] == decision['profile_sha256'], 'Test profile lock mismatch')
        expected_models = {}
        for profile in profiles:
            model = json.loads(json.dumps(profile['model_spec']))
            if model['name'] not in expected_models:
                model['modes'] = []
                expected_models[model['name']] = model
            if profile['mode'] not in expected_models[model['name']]['modes']:
                expected_models[model['name']]['modes'].append(profile['mode'])
        require(final['models'] == list(expected_models.values()), 'Test systems differ from locked profiles')
        require(final['sampling'] == initial['sampling'] and final['screening'] == initial['screening'],
                'Test settings differ from development lock')
        excluded = {tuple(first_three(item['prompt'])) for manifest in (
            read_json(BASE/'dev_panel.json'), read_json(ROOT/'result/stage_4_result/generation_optimization/comparison/panel.json'))
            for item in manifest['prompts']}
        require(all(tuple(first_three(item['prompt'])) not in excluded for item in panel['prompts']),
                'Test prefix overlaps a development panel')
    return dict(status='verified', decision_sha256=digest(path), summary_sha256=digest(BASE/'dev_summary.json'),
                profile_sha256=decision['profile_sha256'], ratings_sha256={job:
                    digest(BASE/job/'coherence_ratings.json') for job in jobs},
                selected=selected['candidate'], promoted=promoted, evaluated_candidates=len(summaries),
                note='Recomputed criterion and hash bindings; ratings remain assistant assessments.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', nargs=3, action='append', metavar=('COMPARISON', 'SPEC', 'PANEL'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-metric-replay', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new audit output path; previous audits are immutable')
    jobs = args.job or [
        (BASE/'initial_comparison', BASE/'initial_spec.json', BASE/'dev_panel.json'),
        (BASE/'regularized_comparison', BASE/'regularized_spec.json', BASE/'dev_panel.json'),
        (BASE/'tied_mixture_comparison', BASE/'tied_mixture_spec.json', BASE/'dev_panel.json'),
        (BASE/'final_test', BASE/'final_spec.json', BASE/'test_panel.json'),
    ]
    torch.set_num_threads(4)
    data = Dataset_Loader(ROOT/'data/stage_4_data', seed=1704).load_generation()
    audits = []
    for output, spec, panel in jobs:
        paths = [repo_path(path) for path in (output, spec, panel)]
        try:
            audit = audit_comparison(*paths, data, replay_metrics=not args.skip_metric_replay)
        except (ValueError, KeyError, TypeError, IndexError, OSError, RuntimeError) as error:
            audit = dict(comparison=str(paths[0].relative_to(ROOT)), status='failed', error=str(error))
        audits.append(audit)
        print(json.dumps({key: audit[key] for key in ('comparison', 'status', 'error') if key in audit}), flush=True)
    try:
        decision = audit_decision(data)
    except (ValueError, KeyError, TypeError, IndexError, OSError, RuntimeError) as error:
        decision = dict(status='failed', error=str(error))
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  auditor_sha256=digest(__file__), torch=str(torch.__version__), device='cpu',
                  data_source_sha256=data['metadata']['source_sha256'],
                  vocabulary_sha256=data['metadata']['vocabulary_sha256'], audits=audits, decision=decision,
                  status='failed' if decision['status'] == 'failed' or any(a['status'] == 'failed' for a in audits)
                  else 'verified_available_comparisons')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return int(result['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
