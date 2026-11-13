# Stage 4: alternative methods for less training-text reuse

本轮已完成补丁上传、117 项完整测试、CPU/GPU 实际推理、固定测试前缀与旧样例回归。新方案减少了直接复现，但在测试中明显损失了连贯性；尚未证实能够同时改善两个目标，因此继续保留原默认。开发集出现的少量严格去重且连贯的结果，不能直接当作测试泛化成功。所有实验及两份独立助手盲评均保留，未根据最终测试结果调整模型或参数。

## What changed

This round adds copy-aware beam search, four own-GRU experiments using publicly pretrained GloVe word vectors, and a question/punchline compatibility experiment. These are isolated additions: the original RNN/LSTM/GRU baselines, earlier experiments, inference interfaces, and report PDF remain unchanged. No external text-generation model, retrieved replacement joke, handwritten completion, or semantic rewriting is used.

The GloVe experiments explicitly use external word-level knowledge. The 100-dimensional vectors come from the official [Stanford GloVe release](https://github.com/stanfordnlp/GloVe), trained on Wikipedia 2014 and Gigaword 5; the pretrained vectors are distributed under PDDL. Archive/member hashes and coverage are recorded. This is an optional disclosed extension, not a claim of training entirely from scratch.

The fixed corpus/split/vocabulary is unchanged: 1,297 training, 164 validation, 161 test jokes and 4,201 vocabulary entries. GloVe covers 92.49% of ordinary training vocabulary types and 96.80% of ordinary training tokens, but does not add vocabulary entries or resolve unknown prompt words. Rare-word semantics and meaningful punchlines remain difficult with about 24,030 ordinary training word/punctuation tokens.

## Bounded model and search experiments

All four GRUs use 100-dimensional GloVe input vectors, 256 hidden units, dropout 0.2, AdamW weight decay 0.01, and the same training partition. Two use a projected, tied output embedding; two keep an independent output projection. Each architecture compares free fine-tuning with a 0.1 anchoring penalty that keeps matched word embeddings near their original vectors. Ordinary validation NLL selects the best checkpoint; no generated output or test loss chooses the semantic model.

| Semantic run | Best / completed epochs | Validation NLL |
|---|---:|---:|
| `glove100_finetune` | 12 / 27 | 5.582459 |
| `glove100_anchor01` | 12 / 27 | 5.585229 |
| `glove_input_finetune` | 21 / 36 | 5.374822 |
| `glove_input_anchor01` | 21 / 36 | 5.361097 |

The best semantic variant is `glove_input_anchor01` (best epoch 21, final epoch 36). Its best and final checkpoints and a fixed 50/50 probability mixture of its best checkpoint with the previous late compact GRU are compared. All four semantic models have worse validation NLL than the prior best predictive system; pretrained vectors are not claimed to improve prediction here.

Copy-aware search applies a soft cost when a generated word extends a rare five-word training span. The cost is weighted by inverse square root of training document frequency; spans present in at least 26 training rows are exempt as common wording. The three prompt words themselves incur no copy cost. Before costs, expansions must be among the original model’s top 40 eligible tokens and at least 1% as probable as its best eligible token, so the search cannot escape copying by choosing arbitrarily implausible words.

Each prompt uses beam width 12, a 48-token cap, three fixed copy weights (0, 0.75, 1.5), and up to eight distinct completed hypotheses per search. Original model likelihood, adjusted search likelihood, matching-span events and copy costs are saved separately. Incomplete paths remain explicitly incomplete; no EOS is fabricated.

Four alternatives per model are evaluated: the zero-penalty beam control, the best soft-0.75 beam result, a direct-copy-filtered union of all three searches, and the previous strict-0.80-filtered union. The original width-5 beam is retained as the baseline, giving 17 configurations. The direct union ranks eligible candidates by mean original model continuation log probability minus a fixed soft similarity cost. It rejects exact full-joke copies, whole continuations found as training spans, unfinished outputs, short/repeated text and question-only outputs, but is not a semantic coherence detector. No eligible candidate means an explicit null result, with no copied fallback.

The separate frozen-GloVe compatibility MLP used training question/punchline positives and random or same-opening/length-matched shuffled negatives. Its held-out synthetic-pair AUC was 0.6094, below the predeclared 0.65 gate, so it was disabled and never used to choose generated text. Not every hard-negative slot had a matching alternative: actual same-opening/length matches were 706/853 training and 69/109 validation slots. Synthetic-pair discrimination would not by itself validate joke quality even if the gate had passed.

## Evaluation and decision

A fresh 16-prefix validation panel was frozen before generation, with eight training-seen and eight unseen prefixes. It excludes the old 12-prefix panel and both panels from the previous generalization round. It was manually stratified by prefix alone, not a random population estimate. All 272 outputs were retained; the 130 distinct texts were each blind-rated once, so identical output receives the same rating.

Coherence scores are 0 for broken/incomplete/repetitive or missing output, 1 for readable but disconnected/generic/unresponsive text, and 2 for a complete meaningful setup/punchline or internally coherent wordplay. These are assistant judgments, not human humor ratings.

The primary ranking favors coherent outputs that pass the unchanged strict continuation similarity/shared-span threshold of 0.80, then direct-copy-free coherent output, total coherence, fewer direct copies and lower validation NLL. Promotion also requires more direct-copy-free coherent output, fewer direct copies, no decline in strict novel/coherent output, and a decline of at most 2/16 in total coherence versus the old width-5 beam. All prompts, including missing outputs, remain in the denominator.

**Direct-copy-free is weaker than strict novelty.** Full normalized joke equality and a whole continuation found in training are separate diagnostics. Partial near copies may pass the direct filter. Strict novelty additionally bounds continuation sequence similarity and longest shared-span fraction. Neither measure proves semantic originality or novelty on the wider internet.

| Development configuration | Coherent /16 | Direct-free + coherent | Strict + coherent | Exact full copies | Copied continuations | Outputs /16 |
|---|---:|---:|---:|---:|---:|---:|
| `previous_late__beam12` | 7 | 1 | 0 | 6 | 10 | 16 |
| `previous_late__soft075` | 4 | 0 | 0 | 4 | 9 | 16 |
| `previous_late__direct_union` | 3 | 3 | 1 | 0 | 0 | 16 |
| `previous_late__strict_union` | 1 | 1 | 1 | 0 | 0 | 16 |
| `previous_late__old_beam5` | 7 | 1 | 0 | 4 | 7 | 16 |
| `semantic_best__beam12` | 0 | 0 | 0 | 0 | 7 | 16 |
| `semantic_best__soft075` | 0 | 0 | 0 | 0 | 6 | 16 |
| `semantic_best__direct_union` | 0 | 0 | 0 | 0 | 0 | 16 |
| `semantic_best__strict_union` | 0 | 0 | 0 | 0 | 0 | 16 |
| `semantic_late__beam12` | 0 | 0 | 0 | 0 | 7 | 16 |
| `semantic_late__soft075` | 0 | 0 | 0 | 0 | 5 | 16 |
| `semantic_late__direct_union` | 0 | 0 | 0 | 0 | 0 | 16 |
| `semantic_late__strict_union` | 0 | 0 | 0 | 0 | 0 | 16 |
| `semantic_late_mix50__beam12` | 4 | 1 | 0 | 3 | 12 | 16 |
| `semantic_late_mix50__soft075` | 3 | 0 | 0 | 2 | 11 | 16 |
| `semantic_late_mix50__direct_union` | 4 | 4 | 3 | 0 | 0 | 16 |
| `semantic_late_mix50__strict_union` | 3 | 3 | 3 | 0 | 0 | 16 |

The experimental choice is `semantic_late_mix50__direct_union`: 4/16 coherent development outputs, including 3/16 passing the strict novelty check, versus 7/16 coherent and 0/16 strict-novel/coherent for the old default. Its total coherence drop is 3/16, beyond the permitted 2/16. Thus `promoted: false`; the original default stays selected. This is evidence of a limited tradeoff, not success on all objectives.

The interrupted run’s decision and finalizer were archived before adding completeness and source-freshness guards. Independent verification confirmed unchanged selection logic, outcomes, ratings, and profile bytes. Final panels were created only after the guarded decision was locked.

## Locked final tests and regression

The new test panel contains 16 unique in-vocabulary prefixes (8 training-seen, 8 unseen), sampled with seed 170455 after locking. It excludes all four previously exposed panels. Test loss on the underlying dataset had already been reported in earlier coursework, so this is a new qualitative prefix panel, not an entirely new dataset. The original 12-prefix development panel is also rerun explicitly as regression evidence. No tuning or profile replacement follows these outputs. Each final panel has two independent blind assistant reviews.

### Fresh test prefixes

| Profile | Coherent: review 1 / review 2 | Strict + coherent: review 1 / review 2 | Exact full copies | Copied continuations | High similarity | Outputs |
|---|---:|---:|---:|---:|---:|---:|
| `previous_late__old_beam5` | 8/16 · 8/16 | 0/16 · 0/16 | 5 | 13 | 13 | 16/16 |
| `semantic_late_mix50__direct_union` | 2/16 · 2/16 | 0/16 · 0/16 | 0 | 0 | 7 | 16/16 |

### Original-panel regression

| Profile | Coherent: review 1 / review 2 | Strict + coherent: review 1 / review 2 | Exact full copies | Copied continuations | High similarity | Outputs |
|---|---:|---:|---:|---:|---:|---:|
| `previous_late__old_beam5` | 8/12 · 9/12 | 0/12 · 1/12 | 7 | 11 | 11 | 12/12 |
| `semantic_late_mix50__direct_union` | 1/12 · 0/12 | 0/12 · 0/12 | 0 | 0 | 4 | 12/12 |

The two final model distributions have these ordinary test losses on all 2,961 prediction targets. Decoding and filtering do not change a model’s NLL or perplexity.

| Model | Test NLL | Test perplexity |
|---|---:|---:|
| `previous_late` | 5.785620 | 325.5839 |
| `semantic_late_mix50` | 5.250534 | 190.6680 |

The previous round’s best predictive mixture remains better at test perplexity (128.0448). This round targets generation tradeoffs rather than claiming a new best predictive score. A reduction in exact copies, near-copy checks, or perplexity is never substituted for the separate coherence counts.

## Run the saved profiles

Use the existing server environment. Prompts must contain exactly three supported words. JSON output includes the selected text or null, decision status, overlap diagnostics, checkpoint provenance and whether external word vectors are used.

```bash
cd /home/ubuntu/PycharmProjects/ECS170

# Retained original default.
.venv-cuda/bin/python script/stage_4_script/predict_alternative_generation.py --text "what did the"

# Explicit experimental semantic-mixture/copy-aware option.
.venv-cuda/bin/python script/stage_4_script/predict_alternative_generation.py --text "what did the" --profile result/stage_4_result/generation_alternatives/recommendation_experimental.json
```

`--device cpu` / `cuda` and `--text-file` are supported. Runtime profiles bind required source files, raw data, split IDs, ordered training-reference rows, vocabulary and checkpoints. Semantic loaders also verify pretrained-vector provenance. The original `predict_generation.py` and `predict_novel_generation.py` remain unchanged.

## Verification and local artifacts

- Full Stage 4 suite: **117 tests passed**; the log hash is bound in `integration_verification.json`.
- Eight real inference runs: both profiles twice on CPU and twice on CUDA, using an old prompt. Repeated IDs/selection/status, RNG preservation and unchanged profiles passed; CPU/CUDA IDs matched on that prompt.
- Four semantic runs: all eight best/final checkpoints independently replayed on validation data.
- Development path audit: 1,552 saved paths, 272 outputs and 130 distinct blind texts. Final path audit: 784 paths/32 outputs for fresh test and 588 paths/24 outputs for regression.
- Independent audits reconstruct probabilities, copy events/costs, plausibility constraints, unions/selections, complete panel coverage and source/data/model bindings. They do not claim exhaustive replay of every discarded beam path. Cross-device likelihood tolerance is 0.0002 per generated token.
- `verification_selection_guard.json`, `verification_final_paths.json` and `verification_final_summary.json` bind the final decision, paths, both reviews and all aggregate figures. Twelve in-memory integrity-tamper checks passed under `python -O`.

All evidence is under `result/stage_4_result/generation_alternatives/`. Checkpoints and external vector binaries remain ignored under their existing data/checkpoint paths. The ranker’s `pairs.json` reconstructs supplied training/validation text and is also ignored; only its hash and training protocol are retained in public-ready evidence. Generated candidate pools are evidence, not a replacement training corpus.

```bash
.venv-cuda/bin/python -m unittest discover -s tests/stage4 -p "test_*.py" -v
.venv-cuda/bin/python script/stage_4_script/verify_alternatives.py --output /tmp/ecs170-alternatives-development-audit.json
.venv-cuda/bin/python script/stage_4_script/verify_alternatives.py --job result/stage_4_result/generation_alternatives/final_test result/stage_4_result/generation_alternatives/final_spec.json result/stage_4_result/generation_alternatives/test_panel.json --job result/stage_4_result/generation_alternatives/regression result/stage_4_result/generation_alternatives/final_spec.json result/stage_4_result/generation_alternatives/regression_panel.json --output /tmp/ecs170-alternatives-final-audit.json
```

The independent summary/decision audit can be rerun with `.venv-cuda/bin/python script/stage_4_script/verify_alternative_summary.py --output /tmp/alternative-summary-audit.json`.

The generator weights and all evaluation/selection artifacts are frozen. New experiments should use new output directories and must disclose that these panels have already been examined. No push, Pull Request or course submission was made in this round.
