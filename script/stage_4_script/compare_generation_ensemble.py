#!/usr/bin/env python3
"""Three predeclared early/late checkpoint mixtures, evaluated on validation only."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader, tokenize
from local_code.stage_4_code.Experiment_Runner import collate, run_epoch, write_json
from local_code.stage_4_code.Generation_Decoder import decode
from local_code.stage_4_code.Generation_Ensemble import MixtureModel
from local_code.stage_4_code.Generation_Optimization import load_generation_checkpoint, validate_prefix
from compare_generation import first_three_words, normalized, repetition


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'result/stage_4_result/generation_optimization/ensemble_comparison')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    outputs = ['summary.json', 'outputs.json', 'blind_review.json', 'blind_mapping.json', 'provenance.json']
    if any((args.output/name).exists() for name in outputs):
        raise FileExistsError('Refusing to replace existing ensemble comparison or ratings; use a new --output')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    base = ROOT/'result/stage_4_result/generation_optimization'
    panel_file = base/'comparison/panel.json'
    manifest = json.loads(panel_file.read_text())
    panel = manifest['prompts']
    panel_hash = hashlib.sha256(json.dumps(panel, sort_keys=True).encode()).hexdigest()
    if panel_hash != manifest['sha256'] or len(panel) != 12:
        raise ValueError('Expected the unchanged twelve-prompt comparison panel')
    data = Dataset_Loader(ROOT/'data/stage_4_data', seed=1704).load_generation()
    validation_by_id = {row['id']: row for row in data['validation']}
    for item in panel:
        if item['source_split'] != 'validation':
            raise ValueError('Every prompt must be from validation')
        assert first_three_words(validation_by_id[item['id']]['text']) == tokenize(item['prompt'])
        assert sum(first_three_words(row['text']) == tokenize(item['prompt']) for row in data['train']) == item['train_prefix_count']
    directory = base/'runs/generation_gru_compact256/checkpoints'
    early, vocab, early_provenance = load_generation_checkpoint(directory/'best.pt', args.device)
    late, late_vocab, late_provenance = load_generation_checkpoint(directory/'last.pt', args.device)
    assert vocab.to_dict() == late_vocab.to_dict() == data['vocab'].to_dict()
    assert early_provenance['config_hash'] == late_provenance['config_hash']
    val_loader = DataLoader(data['validation'], batch_size=128, shuffle=False, collate_fn=collate)
    train_tokens = [normalized(row['text']) for row in data['train']]
    exact = set(train_tokens)
    settings = dict(strategy='beam', max_new_tokens=48, min_new_tokens=4, beam_width=5,
                    length_penalty=.7, no_repeat_ngram_size=3, repetition_penalty=1.1, seed=1704)
    summaries, rows = [], []
    for weight in (.25, .5, .75):
        candidate = f'compact_gru_probability_mix_late_{int(weight * 100):02d}_beam'
        model = MixtureModel([early, late], late_weight=weight).to(args.device).eval()
        validation = run_epoch(model, val_loader, 'generation', torch.device(args.device))
        candidate_rows = []
        for index, item in enumerate(panel):
            prefix = validate_prefix(item['prompt'], vocab)
            generated = decode(model, prefix, **settings)
            ids = generated['ids']
            text = vocab.decode(ids)
            tokens = normalized(text)
            similarities = [difflib.SequenceMatcher(None, tokens, reference, autojunk=False).ratio() for reference in train_tokens]
            nearest = max(range(len(similarities)), key=similarities.__getitem__)
            row = dict(candidate=candidate, prompt_index=index, prompt=item['prompt'], text=text,
                seen_prefix=item['prefix_seen_in_train'], ended_with_eos=generated['ended_with_eos'],
                new_token_count=len(ids)-len(prefix), repeated_trigrams=repetition(tokens),
                exact_training_copy=tokens in exact, nearest_training_similarity=similarities[nearest],
                nearest_training_id=data['train'][nearest]['id'], settings=settings,
                late_weight=weight, finish_reason=generated['finish_reason'])
            candidate_rows.append(row)
            rows.append(row)
        summary = dict(candidate=candidate, late_weight=weight,
            early_checkpoint=str((directory/'best.pt').relative_to(ROOT)), early_epoch=early_provenance['epoch'],
            early_checkpoint_sha256=early_provenance['checkpoint_sha256'],
            late_checkpoint=str((directory/'last.pt').relative_to(ROOT)), late_epoch=late_provenance['epoch'],
            late_checkpoint_sha256=late_provenance['checkpoint_sha256'],
            validation_loss=validation['loss'], validation_token_accuracy=validation['accuracy'],
            validation_target_count=validation['count'], panel_size=len(panel),
            eos_rate=sum(row['ended_with_eos'] for row in candidate_rows)/len(panel),
            repeated_trigram_outputs=sum(row['repeated_trigrams']>0 for row in candidate_rows),
            exact_training_copies=sum(row['exact_training_copy'] for row in candidate_rows),
            near_training_copies=sum(row['nearest_training_similarity']>=.9 for row in candidate_rows))
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    indices = list(range(len(rows)))
    random.Random(170413).shuffle(indices)
    blind, mapping = [], []
    for order, index in enumerate(indices, 1):
        identifier = f'E{order:03d}'
        rows[index]['blind_id'] = identifier
        blind.append(dict(id=identifier, text=rows[index]['text']))
        mapping.append(dict(id=identifier, candidate=rows[index]['candidate'], prompt_index=rows[index]['prompt_index']))
    source_paths = [Path(__file__).resolve(), ROOT/'local_code/stage_4_code/Generation_Ensemble.py',
        ROOT/'local_code/stage_4_code/Generation_Decoder.py', ROOT/'local_code/stage_4_code/Generation_Optimization.py',
        ROOT/'local_code/stage_4_code/Method_RNN.py', ROOT/'local_code/stage_4_code/Dataset_Loader.py',
        ROOT/'local_code/stage_4_code/Experiment_Runner.py', ROOT/'script/stage_4_script/compare_generation.py']
    provenance = dict(panel_sha256=panel_hash, panel_file_sha256=digest(panel_file),
        dataset_source_sha256=data['metadata']['source_sha256'], vocabulary_sha256=data['metadata']['vocabulary_sha256'],
        source_sha256={str(path.relative_to(ROOT)):digest(path) for path in source_paths},
        config_hash=early_provenance['config_hash'],
        component_checkpoints={str((directory/'best.pt').relative_to(ROOT)):early_provenance['checkpoint_sha256'],
                               str((directory/'last.pt').relative_to(ROOT)):late_provenance['checkpoint_sha256']},
        predeclared_late_weights=[.25,.5,.75], aggregation='Fixed mixture of per-component softmax probabilities, not checkpoint parameter averaging',
        frozen_replacement_rule='Replace late-GRU + beam only if the mixture retains at least 9/12 score-2 coherence on the same panel and lowers validation NLL; otherwise retain the current default.',
        experiment_scope='Additional exploratory validation-only comparison after the initial model review; no new training and no test evaluation.',
        purpose='Blindly assess 36 generations before any default selection; preserve the original comparison and ratings.',
        copy_metric='SequenceMatcher token ratio >=0.90 is a near copy; exact normalized token equality separately',
        environment=dict(torch=str(torch.__version__), device=args.device, cudnn_tf32=torch.backends.cudnn.allow_tf32))
    for name, value in [('summary.json', summaries), ('outputs.json', rows), ('blind_review.json', blind),
                        ('blind_mapping.json', mapping), ('provenance.json', provenance)]:
        write_json(args.output/name, value)
    print(f'Completed {len(rows)} ensemble outputs in {args.output}', flush=True)


if __name__ == '__main__':
    main()
