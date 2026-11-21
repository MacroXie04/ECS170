#!/usr/bin/env python3
"""Compare predeclared generation profiles on a frozen prefix panel."""
from __future__ import annotations
import argparse
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
from local_code.stage_4_code.Generation_Optimization import load_generation_checkpoint
from local_code.stage_4_code.Generation_Ensemble import MixtureModel
from local_code.stage_4_code.Novel_Generation import TrainingNoveltyIndex, generate_candidates, select_candidate
from compare_generation import first_three_words


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(spec, device):
    models, vocab, provenance = [], None, []
    for component in spec['components']:
        if component.get('loader', 'original') == 'regularized':
            from local_code.stage_4_code.Regularized_Generation import load_regularized_checkpoint
            model, current, meta = load_regularized_checkpoint(ROOT/component['checkpoint'], device)
        else:
            model, current, meta = load_generation_checkpoint(ROOT/component['checkpoint'], device)
        if vocab is not None and vocab.to_dict() != current.to_dict():
            raise ValueError('Mixture vocabulary mismatch')
        models.append(model); vocab = current
        provenance.append(dict(checkpoint=component['checkpoint'], checkpoint_sha256=digest(ROOT/component['checkpoint']), epoch=meta['epoch']))
    if len(models) == 1:
        return models[0], vocab, provenance
    if len(models) != 2:
        raise ValueError('Expected one or two model components')
    return MixtureModel(models, late_weight=spec['late_weight']).to(device).eval(), vocab, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory; evaluations are immutable')
    specs = json.loads(args.spec.read_text())
    manifest = json.loads(args.panel.read_text()); panel = manifest['prompts']
    if hashlib.sha256(json.dumps(panel, sort_keys=True).encode()).hexdigest() != manifest['sha256']:
        raise ValueError('Frozen panel hash mismatch')
    data = Dataset_Loader(ROOT/'data/stage_4_data', seed=1704).load_generation()
    by_id = {row['id']:row for row in data[args.split]}
    for item in panel:
        assert item['source_split'] == args.split
        assert first_three_words(by_id[item['id']]['text']) == tokenize(item['prompt'])
        assert sum(first_three_words(row['text']) == tokenize(item['prompt']) for row in data['train']) == item['train_prefix_count']
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output.mkdir(parents=True)
    source_names = ['script/stage_4_script/compare_generalization.py', 'script/stage_4_script/compare_generation.py',
        'local_code/stage_4_code/Novel_Generation.py', 'local_code/stage_4_code/Generation_Decoder.py',
        'local_code/stage_4_code/Generation_Optimization.py', 'local_code/stage_4_code/Generation_Ensemble.py',
        'local_code/stage_4_code/Dataset_Loader.py']
    if any(c.get('loader') == 'regularized' for s in specs['models'] for c in s['components']):
        source_names.append('local_code/stage_4_code/Regularized_Generation.py')
    provenance = dict(panel_sha256=digest(args.panel), spec_sha256=digest(args.spec),
        source_sha256={n:digest(ROOT/n) for n in source_names},
        data_source_sha256=data['metadata']['source_sha256'], vocabulary_sha256=data['metadata']['vocabulary_sha256'],
        split=args.split, selection_rule=specs['selection_rule'], torch=str(torch.__version__), device=args.device)
    write_json(args.output/'provenance.json', provenance)
    index = TrainingNoveltyIndex(data['train'])
    rows, pools, summaries = [], [], []
    for spec in specs['models']:
        model, vocab, meta = load_model(spec, args.device)
        assert vocab.to_dict() == data['vocab'].to_dict()
        # On final test evaluation, losses are evaluated for the locked systems only.
        loss = run_epoch(model, DataLoader(data[args.split], batch_size=128, shuffle=False, collate_fn=collate), 'generation', torch.device(args.device))
        for j, item in enumerate(panel):
            pool = generate_candidates(model, vocab, item['prompt'], index, **specs['sampling'])
            pools.append(dict(model=spec['name'], prompt_index=j, candidates=pool))
            for mode in spec.get('modes', ['beam', 'screened']):
                selection = select_candidate(pool, **specs['screening']) if mode == 'screened' else dict(selected=pool[0], status='selected')
                selected = selection.pop('selected')
                row = dict(candidate=spec['name']+'_'+mode, prompt_index=j, prompt=item['prompt'],
                    seen_prefix=item['prefix_seen_in_train'], selection=selection, result=selected,
                    text=selected['text'] if selected else '', model_spec=spec)
                rows.append(row)
            print(json.dumps(dict(model=spec['name'], prompt_index=j, candidate_count=len(pool))), flush=True)
        summaries.append(dict(model=spec['name'], components=meta, loss=loss))
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    order = list(range(len(rows))); random.Random(170431).shuffle(order)
    blind, mapping = [], []
    for k, j in enumerate(order, 1):
        ident = f'N{k:03d}'
        rows[j]['blind_id'] = ident
        blind.append(dict(id=ident, prompt=rows[j]['prompt'], text=rows[j]['text'], no_output=not bool(rows[j]['text'])))
        mapping.append(dict(id=ident, candidate=rows[j]['candidate'], prompt_index=rows[j]['prompt_index']))
    for filename, payload in [('outputs.json',rows),('candidate_pools.json',pools),('metrics.json',summaries),('blind_review.json',blind),('blind_mapping.json',mapping)]:
        write_json(args.output/filename,payload)
    print('Completed '+str(len(rows))+' outputs', flush=True)


if __name__ == '__main__': main()
