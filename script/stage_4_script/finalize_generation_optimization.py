#!/usr/bin/env python3
"""Bind qualitative reviews, select a coherence profile, then evaluate its held-out loss."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Experiment_Runner import collate,evaluation,write_json
from local_code.stage_4_code.Generation_Optimization import load_generation_checkpoint

OUTPUT=ROOT/'result/stage_4_result/generation_optimization'
COMPARE=OUTPUT/'comparison'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_review(review_path,input_path,mapping_path):
    review=read(review_path);inputs=read(input_path);mapping=read(mapping_path)
    ratings=review['ratings']
    expected={r['id'] for r in inputs}
    if len(ratings)!=len(expected) or {r['id'] for r in ratings}!=expected:
        raise ValueError('Ratings do not cover every blinded input exactly once')
    if any(type(r['score']) is not int or r['score'] not in (0,1,2) for r in ratings):
        raise ValueError('Invalid coherence score')
    hashes={'input_sha256':sha(input_path),'mapping_sha256':sha(mapping_path)}
    for key,value in hashes.items():
        if key in review and review[key]!=value:
            raise ValueError('Stale review binding: '+str(review_path))
    review.update(hashes)
    write_json(review_path,review)
    by_id={r['id']:r['score'] for r in ratings}
    return [(r,by_id[r['id']]) for r in mapping]


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    rows=read(COMPARE/'outputs.json')
    summaries=read(COMPARE/'summary.json')
    first=bind_review(COMPARE/'coherence_ratings.json',COMPARE/'blind_review.json',COMPARE/'blind_mapping.json')
    second=bind_review(COMPARE/'confirmatory_ratings.json',COMPARE/'confirmatory_blind.json',COMPARE/'confirmatory_mapping.json')
    first_scores={r['id']:score for r,score in first}
    second_by_candidate={}
    for item,score in second:second_by_candidate.setdefault(item['candidate'],[]).append(score)
    for summary in summaries:
        subset=[r for r in rows if r['candidate']==summary['candidate']]
        scores=[first_scores[r['blind_id']] for r in subset]
        summary.update(coherence_score_counts={str(i):scores.count(i) for i in (0,1,2)},
                       mean_coherence=sum(scores)/len(scores),coherent_count=scores.count(2),
                       familiar_coherent=sum(first_scores[r['blind_id']]==2 for r in subset if r['seen_prefix']),
                       unseen_coherent=sum(first_scores[r['blind_id']]==2 for r in subset if not r['seen_prefix']))
        if summary['candidate'] in second_by_candidate:
            scores2=second_by_candidate[summary['candidate']]
            summary['independent_review']={'score_counts':{str(i):scores2.count(i) for i in (0,1,2)},
                'mean_coherence':sum(scores2)/len(scores2),'coherent_count':scores2.count(2)}
    # Greedy candidates are ablation controls. The deployment profile uses the
    # constrained beam family to also suppress cycles and prefer completed text.
    eligible=[s for s in summaries if s['candidate'].endswith('_beam')]
    selected=min(eligible,key=lambda s:(-s['coherent_count'],-s['mean_coherence'],s['near_training_copies'],s['validation_loss']))
    selected_rows=[r for r in rows if r['candidate']==selected['candidate']]
    rule='Among beam candidates: most score-2 coherent outputs, then highest mean score, fewer near training copies, then lower validation NLL.'
    decoder_sources=['local_code/stage_4_code/Generation_Decoder.py','local_code/stage_4_code/Generation_Optimization.py',
                     'script/stage_4_script/predict_generation.py']
    source_hashes={name:sha(ROOT/name) for name in decoder_sources}
    profile=dict(format_version=1,candidate=selected['candidate'],checkpoint=selected['checkpoint'],
        checkpoint_sha256=selected['checkpoint_sha256'],epoch=selected['epoch'],
        decoding=selected_rows[0]['settings'],source_sha256=source_hashes,selection_rule=rule,
        selection_basis='Assistant qualitative scores on 12 fixed validation-prefix prompts; not human or held-out semantic evaluation',
        validation_assessment=selected,
        limitations=['Familiar-prefix reconstruction dominates; training jokes can be reproduced verbatim.',
                     'Later checkpoint has higher held-out next-token loss than validation-loss-selected weights.',
                     'No retrieval, hand-written continuation or pretrained language model is used.'])
    if (OUTPUT/'recommendation.json').exists():
        old=read(OUTPUT/'recommendation.json')
        if old['candidate']!=profile['candidate'] or old['checkpoint_sha256']!=profile['checkpoint_sha256']:
            raise ValueError('A selected profile already exists; do not silently replace it after test evaluation')
    # Record the decision before opening the selected model's test predictions.
    write_json(OUTPUT/'recommendation.json',profile)
    alternative=next(s for s in summaries if s['candidate']=='compact_gru_best_beam')
    alternative_rows=[r for r in rows if r['candidate']==alternative['candidate']]
    alternative_profile={**profile,'candidate':alternative['candidate'],'checkpoint':alternative['checkpoint'],
        'checkpoint_sha256':alternative['checkpoint_sha256'],'epoch':alternative['epoch'],
        'validation_assessment':alternative,'decoding':alternative_rows[0]['settings'],
        'selection_rule':'Lowest validation next-token loss among the six newly trained model configurations.'}
    alternative_profile['limitations']=['Lower next-token loss does not ensure a coherent free-running punchline.',
        'Small corpus and unseen-prefix generalization remain limiting.']
    write_json(OUTPUT/'recommendation_validation_loss.json',alternative_profile)
    write_json(OUTPUT/'comparison_summary.json',summaries)
    data=Dataset_Loader(ROOT/'data/stage_4_data',seed=1704).load_generation()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,vocab,provenance=load_generation_checkpoint(ROOT/selected['checkpoint'],device)
    assert vocab.to_dict()==data['vocab'].to_dict()
    test_dir=OUTPUT/'selected_test';test_dir.mkdir(exist_ok=True)
    test_metrics=evaluation(model,DataLoader(data['test'],batch_size=128,collate_fn=collate,shuffle=False),
                            'generation',torch.device(device),test_dir)
    write_json(test_dir/'metrics.json',dict(candidate=selected['candidate'],test=test_metrics,
        selection_file_sha256=sha(OUTPUT/'recommendation.json'),evaluated_after_validation_profile_selection=True,
        test_id_sha256=data['metadata']['split_id_sha256']['test']))
    baseline=next(s for s in summaries if s['candidate']=='baseline_gru_greedy')
    baseline_test=read(ROOT/'result/stage_4_result/runs/generation_gru_baseline/metrics.json')['test']
    best_test=read(OUTPUT/'runs/generation_gru_compact256/metrics.json')['test']
    lines=['# Stage 4 generation optimization','',
        'Six additional generation models were trained using the original instructor corpus, vocabulary, group split and seed. Baseline source files and all classification results remain unchanged. No external model, retrieval or hand-written joke completion is used. Work is local on the GPU server; no push or pull request was made.','',
        '## Recommended inference','',
        'From `/home/ubuntu/PycharmProjects/ECS170`:','',
        '```bash',
        '.venv-cuda/bin/python script/stage_4_script/predict_generation.py --text "What did the"',
        '.venv-cuda/bin/python script/stage_4_script/predict_generation.py --text "Why did the"',
        '```','',
        f'Default: **{selected["candidate"]}**, checkpoint epoch **{selected["epoch"]}**. This is the later checkpoint selected for visible coherence on the validation prompt panel, not the minimum-NLL epoch. It uses 256-dimensional embeddings and GRU hidden state, one recurrent layer, dropout 0.10, batch 32, Adam 0.001, and seed 1704.','',
        'Beam width 5 retains multiple continuations; length penalty 0.7, repetition penalty 1.1 and repeated-trigram blocking reduce degenerate continuations. EOS is allowed after four new tokens, generation is capped at 48, and no EOS or sentence is fabricated. A completed sequence still need not make semantic sense.','',
        'For the alternative chosen by lower validation next-token loss:','',
        '```bash',
        '.venv-cuda/bin/python script/stage_4_script/predict_generation.py --profile result/stage_4_result/generation_optimization/recommendation_validation_loss.json --text "What did the"',
        '```','',
        'Optional nucleus sampling: `--strategy sample --temperature 0.7 --top-p 0.9 --seed 1704`. Sampling has not been shown to improve coherence in the blinded beam comparison. Exactly three known words are required; punctuation does not count as a word. Original prompt capitalization is preserved.','',
        '## Validation comparison and honest limitations','',
        'The fixed panel has 12 three-word starts from validation jokes: eight prefixes also occur in training and four do not. The panel was chosen before optimized outputs were inspected. All 108 outputs from nine configurations were reviewed with model names hidden. Scores: 0 broken/incomplete, 1 readable but disconnected, 2 complete and semantically responsive. A second independent assistant reviewed the baseline and selected profile, also blinded. These are assistant judgments on a small development panel, not human ratings or an independent test estimate.','',
        '| Configuration | Score 2 / 12 | Mean (0-2) | Exact / near training copies | Validation NLL |',
        '|---|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f'| {s["candidate"]} | {s["coherent_count"]} | {s["mean_coherence"]:.3f} | {s["exact_training_copies"]} / {s["near_training_copies"]} | {s["validation_loss"]:.4f} |')
    lines += ['',f'The selected profile has {selected["familiar_coherent"]}/8 familiar-prefix and {selected["unseen_coherent"]}/4 unseen-prefix outputs rated 2. Exact copies are normalized-token matches to training text; near copies have token SequenceMatcher ratio at least 0.90 (including exact copies). Of 12 outputs, {selected["exact_training_copies"]} are exact training copies and {selected["near_training_copies"]} are near copies. Better-looking continuations therefore do not demonstrate new-joke creativity.',
        '',f'Second review: baseline {baseline["independent_review"]["coherent_count"]}/12 and selected {selected["independent_review"]["coherent_count"]}/12 rated 2. Both reviews remain inspectable with reasons and input hashes. EOS completion and zero repeated trigrams are policy diagnostics, not independent evidence of learned coherence.',
        '',f'**Tradeoff:** held-out token perplexity is {baseline_test["perplexity"]:.2f} for baseline, {best_test["perplexity"]:.2f} for the minimum-validation-loss alternative, and {test_metrics["perplexity"]:.2f} for the coherence profile. The later profile is more overfit. Its test score was computed after the qualitative selection was saved; it did not choose the profile.',
        '', '## Complete before/after panel','', '| Three-word prompt | Baseline GRU greedy | Selected profile | Exact training copy? |','|---|---|---|---|']
    base_rows={r['prompt_index']:r for r in rows if r['candidate']=='baseline_gru_greedy'}
    for r in selected_rows:
        b=base_rows[r['prompt_index']]
        lines.append('| '+' | '.join(str(v).replace('|','\\|').replace('\n',' ') for v in
            [r['prompt'],b['text'],r['text'],'yes' if r['exact_training_copy'] else 'no'])+' |')
    lines += ['', '## Reproduction and evidence','',
        'Both new training grids reuse `run_stage4.py --task generation --cells gru,lstm,rnn --output result/stage_4_result/generation_optimization/runs --device cuda` with these extra arguments:',
        '', '- `--tag wide256 --embedding-dim 128 --hidden-dim 256 --layers 2 --dropout 0.15 --batch-size 64 --max-epochs 160 --patience 20`',
        '- `--tag compact256 --embedding-dim 256 --hidden-dim 256 --layers 1 --dropout 0.10 --batch-size 32 --max-epochs 180 --patience 20`',
        '', 'Compare with `script/stage_4_script/compare_generation.py`. Returned ratings are bound to the blinded input and mapping SHA256 values by `finalize_generation_optimization.py`; changed inputs invalidate old ratings. The nine decoding comparisons include best-loss and later checkpoints plus greedy controls. Selection is: '+rule,
        '', 'Review files, all generations, diagnostics, copy matches and source hashes are in `result/stage_4_result/generation_optimization/comparison/`. The selected profile and alternative are in the parent directory; selected held-out metrics are in `selected_test/`. Weights stay in ignored checkpoint directories on the server. Frozen baselines remain available and verifiable with the original verification command.',
        '', 'Run tests: `.venv-cuda/bin/python -m unittest discover -s tests/stage4 -v`. The initial five-page PDF describes the original six baseline runs; this document records the subsequent optimization, and should be incorporated during final report revision.','']
    (ROOT/'script/stage_4_script/GENERATION_OPTIMIZATION.md').write_text('\n'.join(lines))
    print(json.dumps({'selected':selected['candidate'],'coherent_count':selected['coherent_count'],
                      'baseline_coherent_count':baseline['coherent_count'],'test':test_metrics},indent=2))


if __name__=='__main__':main()
