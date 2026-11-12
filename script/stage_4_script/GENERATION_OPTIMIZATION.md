# Stage 4 generation optimization

Six additional generation models were trained using the original instructor corpus, vocabulary, group split and seed. Baseline source files and all classification results remain unchanged. No external model, retrieval or hand-written joke completion is used. Work is local on the GPU server; no push or pull request was made.

## Recommended inference

From `/home/ubuntu/PycharmProjects/ECS170`:

```bash
.venv-cuda/bin/python script/stage_4_script/predict_generation.py --text "What did the"
.venv-cuda/bin/python script/stage_4_script/predict_generation.py --text "Why did the"
```

Default: **compact_gru_late_beam**, checkpoint epoch **27**. This is the later checkpoint selected for visible coherence on the validation prompt panel, not the minimum-NLL epoch. It uses 256-dimensional embeddings and GRU hidden state, one recurrent layer, dropout 0.10, batch 32, Adam 0.001, and seed 1704.

Beam width 5 retains multiple continuations; length penalty 0.7, repetition penalty 1.1 and repeated-trigram blocking reduce degenerate continuations. EOS is allowed after four new tokens, generation is capped at 48, and no EOS or sentence is fabricated. A completed sequence still need not make semantic sense.

For the alternative chosen by lower validation next-token loss:

```bash
.venv-cuda/bin/python script/stage_4_script/predict_generation.py --profile result/stage_4_result/generation_optimization/recommendation_validation_loss.json --text "What did the"
```

Optional nucleus sampling: `--strategy sample --temperature 0.7 --top-p 0.9 --seed 1704`. Sampling has not been shown to improve coherence in the blinded beam comparison. Exactly three known words are required; punctuation does not count as a word. Original prompt capitalization is preserved.

## Validation comparison and honest limitations

The fixed panel has 12 three-word starts from validation jokes: eight prefixes also occur in training and four do not. The panel was chosen before optimized outputs were inspected. All 108 outputs from nine configurations were reviewed with model names hidden. Scores: 0 broken/incomplete, 1 readable but disconnected, 2 complete and semantically responsive. A second independent assistant reviewed the baseline and selected profile, also blinded. These are assistant judgments on a small development panel, not human ratings or an independent test estimate.

| Configuration | Score 2 / 12 | Mean (0-2) | Exact / near training copies | Validation NLL |
|---|---:|---:|---:|---:|
| baseline_gru_greedy | 0 | 0.417 | 0 / 0 | 5.1395 |
| baseline_gru_beam | 2 | 0.500 | 1 / 2 | 5.1395 |
| compact_gru_best_greedy | 0 | 0.417 | 0 / 0 | 4.9887 |
| compact_gru_best_beam | 2 | 0.500 | 1 / 2 | 4.9887 |
| compact_gru_late_greedy | 7 | 1.417 | 6 / 7 | 5.5826 |
| compact_gru_late_beam | 9 | 1.667 | 6 / 8 | 5.5826 |
| wide_gru_best_beam | 1 | 0.417 | 1 / 4 | 5.1009 |
| wide_gru_late_beam | 7 | 1.250 | 4 / 8 | 5.3769 |
| compact_lstm_late_beam | 7 | 1.250 | 6 / 8 | 5.5167 |

The selected profile has 7/8 familiar-prefix and 2/4 unseen-prefix outputs rated 2. Exact copies are normalized-token matches to training text; near copies have token SequenceMatcher ratio at least 0.90 (including exact copies). Of 12 outputs, 6 are exact training copies and 8 are near copies. Better-looking continuations therefore do not demonstrate new-joke creativity.

Second review: baseline 0/12 and selected 9/12 rated 2. Both reviews remain inspectable with reasons and input hashes. EOS completion and zero repeated trigrams are policy diagnostics, not independent evidence of learned coherence.

**Tradeoff:** held-out token perplexity is 207.06 for baseline, 175.81 for the minimum-validation-loss alternative, and 325.58 for the coherence profile. The later profile is more overfit. Its test score was computed after the qualitative selection was saved; it did not choose the profile.

## Complete before/after panel

| Three-word prompt | Baseline GRU greedy | Selected profile | Exact training copy? |
|---|---|---|---|
| what do you | what do you call a cow with a cow? a buccaneer | what do you call a cow with no legs? ground beef. | yes |
| why did the | why did the chicken cross the road? because he was outstanding! | why did the chicken cross the playground? to get to the other slide | yes |
| what did the | what did the 0 say to the other? he was outstanding! | what did the 0 say to the 8? nice belt. | yes |
| how do you | how do you call a cow with a lightbulb? a buccaneer. | how do you make a squid laugh? ten tickles. | yes |
| what kind of | what kind of a cow with a wombat? a stick | what kind of birds stick together? vel - crows | yes |
| what does a | what does a cow with a lightbulb? a stick | what does a baker wear on his feet? loafers. | yes |
| where did the | where did the chicken cross the road? because he was outstanding! | where did the general keep his armies? in his sleevies. | no |
| i'm reading a | i'm reading a bar............................................... | i'm reading a book about anti - gravity... | no |
| what side of | what side of a cow with a wombat? a stick | what side of radiation bakes you cookies? a gramma ray | no |
| time flies like | time flies like to the other day? because it was a little. | time flies like clowns? they taste funny! | no |
| a guy walks | a guy walks into a bar............................................. | a guy walks into a bar... the third one ducks. | no |
| i was wondering | i was wondering a bar.............................................. | i was wondering in the desert and saw a redwood tree. i guess it must have been an obstacle illusion. | no |

## Reproduction and evidence

Both new training grids reuse `run_stage4.py --task generation --cells gru,lstm,rnn --output result/stage_4_result/generation_optimization/runs --device cuda` with these extra arguments:

- `--tag wide256 --embedding-dim 128 --hidden-dim 256 --layers 2 --dropout 0.15 --batch-size 64 --max-epochs 160 --patience 20`
- `--tag compact256 --embedding-dim 256 --hidden-dim 256 --layers 1 --dropout 0.10 --batch-size 32 --max-epochs 180 --patience 20`

Compare with `script/stage_4_script/compare_generation.py`. Returned ratings are bound to the blinded input and mapping SHA256 values by `finalize_generation_optimization.py`; changed inputs invalidate old ratings. The nine decoding comparisons include best-loss and later checkpoints plus greedy controls. Selection is: Among beam candidates: most score-2 coherent outputs, then highest mean score, fewer near training copies, then lower validation NLL.

Review files, all generations, diagnostics, copy matches and source hashes are in `result/stage_4_result/generation_optimization/comparison/`. The selected profile and alternative are in the parent directory; selected held-out metrics are in `selected_test/`. Weights stay in ignored checkpoint directories on the server. Frozen baselines remain available and verifiable with the original verification command.

Run tests: `.venv-cuda/bin/python -m unittest discover -s tests/stage4 -v`. The initial five-page PDF describes the original six baseline runs; this document records the subsequent optimization, and should be incorporated during final report revision.

## Additional early/late probability mixtures

A final validation-only experiment mixed the early and late compact GRU next-token probability distributions. It did not average parameter tensors or retrain either network. Each component retained its own recurrent state. The tested late weights (0.25, 0.50, 0.75) and replacement rule were fixed before generated outputs were scored: retain at least 9/12 score-2 outputs while lowering validation NLL.

| Late probability weight | Validation NLL | Coherent / 12 | Exact / near training copies | Replace default? |
|---:|---:|---:|---:|---|
| 0.25 | 4.8997 | 6 | 4 / 6 | No |
| 0.50 | 4.9368 | 8 | 6 / 8 | No |
| 0.75 | 5.0657 | 8 | 6 / 8 | No |

Mixtures improved next-token validation loss but did not meet the frozen 9/12 coherence threshold, so the default profile was retained. All 36 texts and independently blinded ratings are in `ensemble_comparison/`; its decision file records the unchanged recommended-profile hash. No mixture test-set evaluation was performed. These exploratory comparisons do not turn development-panel judgments into independent generalization estimates.

Reproduce with `compare_generation_ensemble.py`, review the blinded file using the same rubric, then run `finalize_ensemble_comparison.py`. New tests verify probability arithmetic, independent recurrent state, causal forward/step agreement and decoder compatibility.
