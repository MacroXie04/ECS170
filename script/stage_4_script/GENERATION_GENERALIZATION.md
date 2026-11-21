# Stage 4 generation: generalization experiment

本轮增加了共享词嵌入权重的 GRU、较强正则化、概率混合和训练文本复现检查。新混合模型的测试困惑度降至 128.0448，但在新前缀上，连贯性从原模型的 8/16（第二位评审 9/16）降至 3/16；严格复现检查后的输出为 0/16 连贯。所有方案都没有得到同时满足连贯与固定新颖性条件的输出，因此保留原来的默认模型。新方案作为明确标注的实验选项保留，不能把词面去重或较低困惑度表述为已解决笑话泛化问题。

## Result and scope

This follow-up improved next-token prediction and added auditable controls against training-text reuse. It did **not** demonstrate improved coherent, novel joke generation. The frozen development decision retains `previous_late_beam`; the new predictive and screened profiles are explicit experimental options. No model or decoding choice was changed after the final test-prefix evaluation.

All new models were trained from scratch on the existing training split. No external data, pretrained embeddings, handwritten completions, retrieval replacements, or semantic rewriting were used. The novelty index compares generated text with training records; it never substitutes a retrieved joke. The original Stage 4 report/PDF and its baseline results remain unchanged.

## Data and evaluation design

The existing normalized-content-group split is unchanged: 1,297 training rows, 164 validation rows, and 161 test rows. The vocabulary is fitted only on training data and contains 4,201 entries including four special tokens. The training corpus has 24,030 ordinary word/punctuation tokens and 4,197 ordinary token types: 2,564 types occur once (61.1%), and 3,215 occur at most twice (76.6%). The data loader records 26,624 training tokens when BOS and EOS are included. This limited evidence for rare words constrains grammatical and semantic learning.

`dev_panel.json` freezes 16 validation prefixes before this experiment's generated outputs were inspected: eight prefixes seen in training and eight unseen prefixes. Every word is in the frozen vocabulary, and none reuses the older 12-prompt panel. Familiar-prefix frequencies range from 1 to 30 training examples. This is a manually stratified development panel selected from validation texts; it is not a random population estimate. Reference completions were not supplied to the model or blind reviewer.

After `decision.json` was locked, `test_panel.json` selected 16 fresh, unique test prefixes with seed 170499: eight seen and eight unseen in training. It excludes every prefix in the old 12-prompt panel and the new 16-prompt development panel. The candidate set, profile hashes, and selection decision were already fixed. The dataset's test loss had been used in earlier coursework experiments, so this is a fresh **qualitative prefix panel**, not a completely untouched new dataset.

The rubric is 0 for incomplete, ungrammatical, repetitive, or missing output; 1 for readable but disconnected, generic, contradictory, or unresponsive text; and 2 for a complete, understandable setup/punchline or internally responsive wordplay. Identical text receives the same score within each review. Coherence is assessed separately from copiedness and personal humor preference. The development panel has one blinded assistant review; the final panel has two separately blinded assistant reviews. These small descriptive counts are not a human humor study or an estimate with established statistical significance.

## Two bounded training runs

`Regularized_Generation.py` implements a single-layer causal GRU whose input embedding and output projection share weights. Shared weights are initialized with normal standard deviation 0.1; output bias is zero. The PAD row is initialized to zero and its shared input/output gradient is zeroed. Both runs use dropout 0.35, AdamW at learning rate 0.001 with weight decay 0.01, batch size 64, gradient clipping at 1.0, and seed 1704. Training has a 100-epoch cap and 12-epoch patience. Checkpoints are selected by the lowest **ordinary, unsmoothed validation NLL**, with the earliest epoch winning ties.

| Training run | Embedding / hidden size | Parameters | Label smoothing | Best epoch | Epochs completed | Validation NLL |
|---|---:|---:|---:|---:|---:|---:|
| `tied128_dropout35` | 128 | 641,001 | 0 | 34 | 46 | 5.065001 |
| `tied256_dropout35_smooth05` | 256 | 1,474,409 | 0.05 | 22 | 34 | 4.824824 |

At width 256, tying reduces the parameter count from the previous untied model's 2,549,865 to 1,474,409, a 42.18% reduction. Training loss with label smoothing is not used as a substitute for comparable evaluation NLL. These two configurations change several features together, so this experiment does not identify the individual causal benefit of tying, dropout, weight decay, or smoothing.

The original `mix25` combines probabilities from the earlier compact-GRU best checkpoint and its late checkpoint, assigning 25% to the late model. The new `tied_mix25` instead assigns 75% to the best tied 256-dimensional model and 25% to the previous late GRU. `tied_mix50` uses equal weights. Each component retains its own recurrent state; probabilities, not checkpoint weights or raw logits, are combined.

## Candidate generation and copy controls

For each prompt, the saved pool contains one beam result plus 16 nucleus samples, with temperature 0.8, top-p 0.9, top-k 40, and a 48-token continuation cap. Sampling uses private deterministic seeds. Identical token sequences are merged while retaining their source strategies and seeds. Failed candidates remain in the saved evidence. Beam profiles return the beam result; screened profiles choose the eligible candidate with the highest mean original-model continuation log probability, including punctuation and EOS in the denominator.

Screening requires EOS, at least five new lexical words, no repeated lexical trigram, no exact full-text copy, and no continuation that exactly matches either a training continuation or any contiguous span of a training joke. It also requires both continuation sequence similarity and longest shared training-span fraction to be at most 0.80. The legacy setting name `min_content_words` counts lexical words, including function words; it is not a semantic content-word detector. Case and punctuation are normalized for copy checks.

The summary columns have distinct meanings:

- **Exact copies:** the complete output equals a training joke after lexical normalization.
- **Copied continuations:** the generated continuation exactly matches a training continuation or a contiguous span anywhere in a training joke. Changing only the three-word setup does not conceal a copied ending.
- **High similarity:** an exact full/continuation/span copy, continuation similarity above 0.80, or a shared-span fraction above 0.80.
- **Eligible:** all screening conditions pass, including completion, length, and repetition requirements.
- **Coherent + eligible:** score 2 and eligibility on the same output; all 16 prompts remain in the denominator, including abstentions.

These are lexical screening rules, not semantic originality or coherence classifiers. A grammatical-looking sequence or EOS does not establish a meaningful punchline. When no candidate qualifies, the guarded mode returns `text: null` and `status: no_eligible_candidate`, with rejection diagnostics. It does not silently fall back to a copied or rejected candidate.

## Development comparison

All 12 model/decoding combinations are reported below from `dev_summary.json`. Every combination returned 16 outputs in this panel. The primary selection rule required a strict improvement in coherent-and-eligible count over the previous default, with total coherent count declining by no more than 2/16. Ties were resolved by total coherence and then lower validation NLL.

| Candidate | Validation NLL | Coherent /16 | Coherent + eligible /16 | Exact copies | Copied continuations | High similarity | Eligible /16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `previous_late_beam` | 5.582584 | 7 | 0 | 6 | 13 | 15 | 1 |
| `previous_late_screened` | 5.582584 | 0 | 0 | 0 | 0 | 0 | 16 |
| `mix25_beam` | 4.899708 | 1 | 0 | 0 | 13 | 14 | 2 |
| `mix25_screened` | 4.899708 | 0 | 0 | 0 | 0 | 0 | 16 |
| `tied128_dropout35_beam` | 5.065001 | 0 | 0 | 0 | 9 | 9 | 6 |
| `tied128_dropout35_screened` | 5.065001 | 0 | 0 | 0 | 0 | 0 | 16 |
| `tied256_dropout35_smooth05_beam` | 4.824824 | 1 | 0 | 0 | 10 | 10 | 5 |
| `tied256_dropout35_smooth05_screened` | 4.824824 | 0 | 0 | 0 | 0 | 0 | 16 |
| `tied_mix25_beam` | 4.725207 | 2 | 0 | 0 | 12 | 13 | 2 |
| `tied_mix25_screened` | 4.725207 | 0 | 0 | 0 | 0 | 0 | 16 |
| `tied_mix50_beam` | 4.763324 | 8 | 0 | 6 | 15 | 16 | 0 |
| `tied_mix50_screened` | 4.763324 | 0 | 0 | 0 | 0 | 0 | 16 |

No candidate produced a coherent-and-eligible output. `tied_mix50_beam` had the largest raw coherence count, but all 16 outputs met the high-similarity condition. It did not pass the promotion rule. `decision.json` therefore records `promoted: false`, and `recommendation.json` retains the previous late-GRU beam profile. The lower-NLL `tied_mix25` and its screened variant are retained only as explicit experimental choices.

## Locked final test-prefix comparison

The following counts come from `final_test/summary.json`. All three profiles returned 16 outputs, so the screened model's low copy count was not produced by hiding abstentions in this particular panel.

| Locked profile | Coherent: review 1 /16 | Coherent: review 2 /16 | Coherent + eligible: both reviews | Exact copies | Copied continuations | High similarity | Eligible /16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `previous_late_beam` | 8 | 9 | 0 | 6 | 12 | 13 | 2 |
| `tied_mix25_beam` | 3 | 3 | 0 | 1 | 12 | 13 | 3 |
| `tied_mix25_screened` | 0 | 0 | 0 | 0 | 0 | 0 | 16 |

The two reviewers disagreed on the 0/1/2 score for 2, 1, and 1 of the 16 outputs, respectively. Their raw coherence counts are reported separately rather than averaged or resolved to improve the result. Both reviews found zero coherent-and-eligible outputs for every profile. On reviewer 1's seen/unseen strata, the default had 7/8 and 1/8 coherent outputs; the new mixture had 2/8 and 1/8; screening had 0/8 in both strata.

| Predictive model | Test NLL | Test perplexity |
|---|---:|---:|
| Previous late compact GRU | 5.785620 | 325.5839 |
| New tied/late probability mixture, late weight 0.25 | 4.852380 | 128.0448 |

The earlier compact model selected for its lowest validation loss had previously reported test perplexity about 175.81. The new mixture's 128.0448 is a predictive improvement over that value as well as the late model. All use the same tokenizer, vocabulary, split, and ordinary NLL definition; the current test calculation has 2,961 prediction targets. Screening does not change the model's NLL or perplexity.

The exact-copy count fell from 6 to 1 for the new beam mixture, but copied continuations remained 12/16 and high-similarity outputs remained 13/16. This demonstrates why exact-string count alone is insufficient. The screened profile reduced the measured copies to zero while producing zero coherent outputs. Lower predictive loss and lower observed lexical copying therefore do not establish success on the joint user-facing objective.

## Running the three saved profiles

Run from the server checkout with the existing environment. Inputs must contain exactly three words supported by the saved vocabulary. The command returns JSON with the selected text, status, candidate diagnostics, profile hash, and checkpoint provenance.

```bash
cd /home/ubuntu/PycharmProjects/ECS170

# Retained default: previous late GRU with beam decoding.
.venv-cuda/bin/python script/stage_4_script/predict_novel_generation.py --text "did you hear"

# Experimental lower-NLL mixture; coherence did not improve.
.venv-cuda/bin/python script/stage_4_script/predict_novel_generation.py --text "did you hear" --profile result/stage_4_result/generation_generalization/recommendation_predictive.json

# Experimental fixed training-copy screen; no semantic-quality guarantee.
.venv-cuda/bin/python script/stage_4_script/predict_novel_generation.py --text "did you hear" --profile result/stage_4_result/generation_generalization/recommendation_strict.json
```

`--strict-novelty` applies the same guard to an explicitly selected beam profile. `--seed` controls the sampled candidate pool; `--device cpu` and `--device cuda` select the device; `--text-file` accepts a prompt file. A custom seed changes the sampled outputs, so it is outside the exact saved-panel evidence. The older `predict_generation.py` entry point and its existing default remain available.

Profiles bind the raw dataset, split IDs, ordered vocabulary, exact training-copy reference rows, relevant source files, and checkpoint contents by SHA-256. Checkpoint loaders also verify training configuration and source provenance. Inference fails on binding drift rather than silently using a stale profile. This provides reproducibility checks, not a guarantee of semantic quality.

## Evidence and verification

Evidence is under `result/stage_4_result/generation_generalization/`: the frozen development/test panels and experiment specifications, all candidate pools and returned outputs, randomized blind files and mappings, per-item ratings with reasons, model metrics, `dev_summary.json`, `decision.json`, three recommendation profiles, and `final_test/summary.json`. Training histories, configurations, and best/last checkpoints are under `regularized_runs/`.

The implementation validation reported 69 passing tests and eight CLI integration runs. The authoritative artifact audit is `verification_development_v2.json`: it verifies 2,112 distinct saved candidates and 240 returned outputs across development and final comparison artifacts, replays eight model/split NLL calculations on CPU, checks model likelihoods and data/source bindings, and recomputes the default-selection decision. The earlier v1 audit is retained as a failed attempt with an overly strict CPU/CUDA likelihood tolerance. V2 allows 0.0002 per generated token; the maximum observed total log-probability difference was 0.001955. This tolerance change addresses floating-point replay only; it does not change outputs, ratings, similarity checks, or selection. The separate `verification_final_summary.json` audit also passed, validating the final summary fields and hashes and all 48 confirmatory ratings.

To audit the existing artifacts without retraining:

```bash
.venv-cuda/bin/python script/stage_4_script/verify_generalization.py --output /tmp/ecs170-generalization-verification.json
```

The experiment artifacts and selection are frozen. A future experiment should use a new output directory and new development criteria rather than overwrite this decision or tune against these final test outputs. The small corpus, sparse vocabulary, word-level OOV behavior, narrow three-word prompts, and assistant-only judgments remain material limits. Subword modeling or external pretrained knowledge would be separate future experiments with their own provenance and evaluation; neither was used here.
