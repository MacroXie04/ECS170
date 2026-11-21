#!/usr/bin/env python3
"""Generate three-word continuations with auditable training-copy controls."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Novel_Generation import TrainingNoveltyIndex, generate_candidates, select_candidate
from compare_generalization import load_model

DEFAULT_PROFILE = ROOT/'result/stage_4_result/generation_generalization/recommendation.json'
REQUIRED_SOURCES = {
    'script/stage_4_script/predict_novel_generation.py',
    'script/stage_4_script/compare_generalization.py',
    'script/stage_4_script/compare_generation.py',
    'local_code/stage_4_code/Novel_Generation.py',
    'local_code/stage_4_code/Generation_Decoder.py',
    'local_code/stage_4_code/Generation_Optimization.py',
    'local_code/stage_4_code/Dataset_Loader.py',
    'local_code/stage_4_code/Method_RNN.py',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_digest(records):
    rows = [dict(id=row['id'], text=row['text']) for row in records]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def bound_path(name):
    path = (ROOT/name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Bound artifact must be inside this repository: '+str(name))
    return path


def verify_profile(profile, data):
    """Fail closed when a model, source, or training-copy reference has drifted."""
    if profile.get('schema_version') != 1 or profile.get('mode') not in ('beam', 'screened'):
        raise ValueError('Unsupported generation profile')
    expected = profile['data_binding']
    if expected['training_rows_sha256'] != training_digest(data['train']):
        raise ValueError('Training novelty reference differs from the selected profile')
    components = profile['model_spec']['components']
    if len(components) not in (1, 2) or any(c.get('loader','original') not in ('original','regularized') for c in components):
        raise ValueError('Expected one or two components with supported loaders')
    for key in ('source_sha256', 'vocabulary_sha256', 'split_id_sha256'):
        if expected[key] != data['metadata'][key]:
            raise ValueError('Dataset binding changed: '+key)
    required = set(REQUIRED_SOURCES)
    if len(components) == 2: required.add('local_code/stage_4_code/Generation_Ensemble.py')
    if any(c.get('loader') == 'regularized' for c in components):
        required.add('local_code/stage_4_code/Regularized_Generation.py')
    if not required.issubset(profile['source_sha256']):
        raise ValueError('Profile is missing required source bindings')
    for name, value in profile['source_sha256'].items():
        if digest(bound_path(name)) != value:
            raise ValueError('Generation source changed: '+name)
    for component in components:
        checkpoint = bound_path(component['checkpoint'])
        if digest(checkpoint) != component['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed: '+component['checkpoint'])
        config = json.loads((checkpoint.parent.parent/'config.json').read_text())
        for key in ('source_sha256', 'vocabulary_sha256', 'split_id_sha256'):
            if config['data_metadata'][key] != data['metadata'][key]:
                raise ValueError('Checkpoint data binding changed: '+key)


def predict(prompt, profile=DEFAULT_PROFILE, device='auto', seed=None, strict_novelty=False):
    profile_path = Path(profile)
    settings = json.loads(profile_path.read_text())
    data = Dataset_Loader(ROOT/'data/stage_4_data', seed=1704).load_generation()
    verify_profile(settings, data)
    if device == 'auto': device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, vocab, provenance = load_model(settings['model_spec'], device)
    if vocab.to_dict() != data['vocab'].to_dict():
        raise ValueError('Model vocabulary differs from the training-copy reference')
    sampling = dict(settings['sampling'])
    if seed is not None: sampling['seed'] = seed
    pool = generate_candidates(model, vocab, prompt, TrainingNoveltyIndex(data['train']), **sampling)
    if not pool:
        raise ValueError('Generator produced no auditable candidates')
    mode = 'screened' if strict_novelty else settings['mode']
    selection = select_candidate(pool, **settings['screening']) if mode == 'screened' else dict(selected=pool[0],status='selected')
    selected = selection.pop('selected')
    return dict(profile=settings['candidate'], profile_sha256=digest(profile_path), mode=mode,
                status=selection['status'], text=selected['text'] if selected else None,
                result=selected, selection=selection, sampling=sampling, checkpoints=provenance,
                notice=('Lexical novelty is a training-copy check, not a guarantee of coherence or originality.' if selected else
                        'No complete continuation met the fixed training-copy checks. No copied fallback was returned.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text'); group.add_argument('--text-file',type=Path)
    parser.add_argument('--profile',type=Path,default=DEFAULT_PROFILE)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--strict-novelty',action='store_true',help='Require fixed copy-screen eligibility even for a beam profile')
    args = parser.parse_args()
    torch.set_num_threads(4)
    prompt = args.text if args.text is not None else args.text_file.read_text(encoding='utf-8').strip()
    try:
        result = predict(prompt,args.profile,args.device,args.seed,args.strict_novelty)
    except (ValueError,FileNotFoundError,KeyError) as error:
        parser.exit(2, str(error)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
