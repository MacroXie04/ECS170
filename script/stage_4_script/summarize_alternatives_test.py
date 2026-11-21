#!/usr/bin/env python3
"""Bind two blind reviews of locked test and regression outputs."""
from __future__ import annotations
import json
from pathlib import Path
from finalize_alternatives import ROOT,BASE,read,sha,summarize
from local_code.stage_4_code.Experiment_Runner import write_json


def main():
    output=BASE/'final_summary.json'
    if output.exists():raise FileExistsError('Final report evidence already exists')
    decision=read(BASE/'decision.json')
    for name,expected in decision['profile_sha256'].items():
        if sha(BASE/name)!=expected:raise ValueError('Locked profile changed')
    result=dict(decision_sha256=sha(BASE/'decision.json'),profile_sha256=decision['profile_sha256'],
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
                       'script/stage_4_script/finalize_alternatives.py':sha(ROOT/'script/stage_4_script/finalize_alternatives.py')},
        panels={},profile_selection_changed=False,
        caveats=['Two blinded assistant reviews, not human ratings or proof of original humor.',
                 'Novelty checks measure overlap with this training partition, not the wider internet.',
                 'Regression reuses the original12 prompts; only the16testprefixes are newly evaluated in this round.',
                 'GloVe supplies external word-level knowledge; the recurrent generators are our own models.'])
    for directory,split,panel_file in [('final_test','test','test_panel.json'),('regression','validation','regression_panel.json')]:
        panel=read(BASE/panel_file);count=len(panel['prompts'])
        if panel['decision_sha256']!=result['decision_sha256']:raise ValueError('Panel was not bound to current pretest decision')
        primary,first_bindings=summarize(BASE/directory,split,'coherence_ratings.json',count)
        secondary,second_bindings=summarize(BASE/directory,split,'confirmatory_ratings.json',count)
        secondary={r['candidate']:r for r in secondary}
        first_scores={r['id']:r['score'] for r in read(BASE/directory/'coherence_ratings.json')}
        second_scores={r['id']:r['score'] for r in read(BASE/directory/'confirmatory_ratings.json')}
        rows=read(BASE/directory/'outputs.json')
        for row in primary:
            second=secondary[row['candidate']]
            row['confirmatory']={k:second[k] for k in ['coherent_count','direct_copy_free_coherent_count','strict_novel_coherent_count','score_counts','seen_coherent','unseen_coherent']}
            row['reviewer_score_disagreements']=sum(first_scores[r['blind_id']]!=second_scores[r['blind_id']] for r in rows if r['candidate']==row['candidate'])
        result['panels'][directory]=dict(summaries=primary,primary_bindings=first_bindings,confirmatory_bindings=second_bindings,
            panel_sha256=sha(BASE/panel_file),coverage=panel['coverage'])
    write_json(output,result)
    for name,panel in result['panels'].items():
        for row in panel['summaries']:
            print(name,row['candidate'],'coherent',row['coherent_count'],row['confirmatory']['coherent_count'],
                  'strict-novel',row['strict_novel_coherent_count'],row['confirmatory']['strict_novel_coherent_count'],
                  'copies',row['exact_full_copies'],row['copied_continuations'],'count',row['count'])


if __name__=='__main__':main()
