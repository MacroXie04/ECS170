#!/usr/bin/env python3
"""Apply the predeclared mixture acceptance rule; never relax it after viewing results."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from script.stage_4_script.finalize_generation_optimization import bind_review,read,sha
from local_code.stage_4_code.Experiment_Runner import write_json


def main():
    directory=ROOT/'result/stage_4_result/generation_optimization'
    compared=directory/'ensemble_comparison'
    old_profile=read(directory/'recommendation.json')
    before=sha(directory/'recommendation.json')
    ratings=bind_review(compared/'coherence_ratings.json',compared/'blind_review.json',compared/'blind_mapping.json')
    scores={r['id']:score for r,score in ratings}
    outputs=read(compared/'outputs.json')
    summaries=read(compared/'summary.json')
    for row in summaries:
        cases=[r for r in outputs if r['candidate']==row['candidate']]
        values=[scores[r['blind_id']] for r in cases]
        row.update(coherent_count=values.count(2),mean_coherence=sum(values)/len(values),
                   score_counts={str(i):values.count(i) for i in (0,1,2)})
        row['meets_frozen_replacement_rule']=(row['coherent_count']>=9 and row['validation_loss']<old_profile['validation_assessment']['validation_loss'])
    passing=[r for r in summaries if r['meets_frozen_replacement_rule']]
    # A passing model would require a separately reviewed deployment-profile
    # implementation. Current experiments yielded no passing candidate.
    if passing:
        raise RuntimeError('A mixture passes the frozen rule; review deployment integration before changing the profile')
    write_json(compared/'reviewed_summary.json',summaries)
    decision=dict(rule=read(compared/'provenance.json')['frozen_replacement_rule'],
        passing_candidates=[],decision='Keep the existing late-GRU plus beam profile; no mixture met the coherence threshold.',
        retained_candidate=old_profile['candidate'],recommendation_sha256=before,
        input_sha256=sha(compared/'blind_review.json'),ratings_sha256=sha(compared/'coherence_ratings.json'),
        test_evaluation_performed=False,reviewer='Independent assistant qualitative review, not a human study')
    write_json(compared/'decision.json',decision)
    assert before==sha(directory/'recommendation.json')
    document=ROOT/'script/stage_4_script/GENERATION_OPTIMIZATION.md'
    text=document.read_text().split('\n## Additional early/late probability mixtures\n')[0].rstrip()
    lines=['','## Additional early/late probability mixtures','',
        'A final validation-only experiment mixed the early and late compact GRU next-token probability distributions. It did not average parameter tensors or retrain either network. Each component retained its own recurrent state. The tested late weights (0.25, 0.50, 0.75) and replacement rule were fixed before generated outputs were scored: retain at least 9/12 score-2 outputs while lowering validation NLL.','',
        '| Late probability weight | Validation NLL | Coherent / 12 | Exact / near training copies | Replace default? |',
        '|---:|---:|---:|---:|---|']
    for row in summaries:
        lines.append(f'| {row["late_weight"]:.2f} | {row["validation_loss"]:.4f} | {row["coherent_count"]} | {row["exact_training_copies"]} / {row["near_training_copies"]} | No |')
    lines += ['',
        'Mixtures improved next-token validation loss but did not meet the frozen 9/12 coherence threshold, so the default profile was retained. All 36 texts and independently blinded ratings are in `ensemble_comparison/`; its decision file records the unchanged recommended-profile hash. No mixture test-set evaluation was performed. These exploratory comparisons do not turn development-panel judgments into independent generalization estimates.',
        '', 'Reproduce with `compare_generation_ensemble.py`, review the blinded file using the same rubric, then run `finalize_ensemble_comparison.py`. New tests verify probability arithmetic, independent recurrent state, causal forward/step agreement and decoder compatibility.','']
    document.write_text(text+'\n'+'\n'.join(lines))
    print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
