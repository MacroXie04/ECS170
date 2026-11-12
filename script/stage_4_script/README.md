# Stage 4: recurrent text classification and generation

This implementation trains RNN, LSTM, and GRU models for IMDb sentiment classification and short-joke generation. The six full-data runs are complete on `code-server-gpu`, in `/home/ubuntu/PycharmProjects/ECS170`, branch `stage4-rnn-hongzhe`. Work remains on the server: no push or pull request has been made.

The GitHub repository is public. Keep instructor datasets, checkpoints, student identifiers/contact details, and the report PDF off GitHub. Data, checkpoint directories, and the draft PDF are ignored by Git. The report is a **local review draft**: team review and an accessible submission source-code link are still needed before submission.

## Environment and setup

Run the commands below from `/home/ubuntu/PycharmProjects/ECS170`. The existing `.venv-cuda` environment is ready; reinstalling is unnecessary.

| Component | Verified version |
|---|---|
| Python | 3.12.16 |
| PyTorch | 2.14.1+cu130 |
| CUDA / cuDNN | 13.0 / 9.24.0 |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.1 |
| Matplotlib | 3.11.2 |
| ReportLab / pypdf | 5.1.0 / 7.0.1 |
| GPU | NVIDIA RTX PRO 4500 Blackwell Server Edition |

To recreate the environment separately, without replacing the existing one:

```bash
uv venv --python 3.12.16 --seed .venv-stage4
.venv-stage4/bin/python -m pip install 'torch==2.14.1+cu130' --index-url https://download.pytorch.org/whl/cu130
.venv-stage4/bin/python -m pip install 'numpy==2.5.3' 'scikit-learn==1.9.1' 'matplotlib==3.11.2'
.venv-stage4/bin/python -m pip install -r script/stage_4_script/requirements-report.txt
```

Use `.venv-stage4` in place of `.venv-cuda` in subsequent commands when using that new environment. CUDA also requires a compatible host driver; CPU inference and tests are supported.

## Data, checks, and training

Use the instructor-provided archive already stored as `data/stage_4_data.zip`. The preparation script checks extraction paths, skips macOS metadata, and writes an archive SHA256 manifest.

```bash
.venv-cuda/bin/python script/stage_4_script/prepare_stage4_data.py data/stage_4_data.zip
.venv-cuda/bin/python -m unittest discover -s tests/stage4 -p 'test_*.py' -v
.venv-cuda/bin/python script/stage_4_script/run_stage4.py --device cuda
```

The final command runs both tasks and all three recurrent cells. Repeating the **same command** resumes interrupted runs from `last.pt` and skips completed runs. Configuration and training-source hashes must match; changed experiments need a new `--tag` or output directory. Do not overwrite the verified baseline artifacts. For a separate small functionality check:

```bash
.venv-cuda/bin/python script/stage_4_script/run_stage4.py --device cuda --smoke --output result/stage_4_result/smoke/current --tag smoke
```

Smoke outputs use small subsets and are not reportable full-data results. Run `run_stage4.py --help` for task/cell selection, dimensions, epochs, learning rate, and other options.

## Model and evaluation protocol

All models learn embeddings from scratch: 128-dimensional embedding, one unidirectional recurrent layer with 128 hidden units, and dropout 0.3. Classification uses masked mean pooling followed by two logits; generation maps each recurrent state to vocabulary logits. Training uses Adam at 0.001, batch size 128, gradient clipping at norm 1, and seed 1704. Classification allows 12 epochs with validation patience 4; generation allows 80 with patience 12. The lowest validation-loss checkpoint is evaluated on test data. Padding is excluded from pooling and next-token loss; generation uses separate BOS/EOS-bounded joke sequences.

- **IMDb:** preserve all 25,000 official test reviews. Normalize and group duplicate texts within the 25,000 official training reviews, then allocate whole groups to balanced training/validation sets of 22,500/2,500. Validation has 1,250 reviews per class. Vocabulary is fitted only on training text, capped at 20,000 tokens including four special tokens; reviews are truncated to 256 tokens. HTML is removed, Unicode normalized, and text lowercased, retaining negation, contractions, and punctuation.
- **Jokes:** parse the extensionless `text_generation/data` as CSV. Its 1,622 rows form 1,609 normalized groups. All 13 duplicate rows are retained, with each group kept entirely in one approximately 80/10/10 split. Actual counts are 1,297/164/161 rows, with no cross-split text overlaps. The training-only vocabulary has 4,201 tokens. Jokes are not truncated; maximum length is 60 tokens including BOS/EOS.
- **Evaluation:** sentiment accuracy and macro precision/recall/F1; generation non-padding next-token cross-entropy, perplexity, token accuracy, and qualitative three-word continuations. Generation references come from training text and illustrate reconstruction, not held-out semantic correctness.

## Verified full-data results

Classification scores are percentages, evaluated on all 25,000 official test reviews.

| Cell | Accuracy | Macro precision | Macro recall | Macro F1 | Best / completed epochs |
|---|---:|---:|---:|---:|---:|
| RNN | 85.824 | 85.834 | 85.824 | 85.823 | 9 / 12 |
| LSTM | 86.768 | 86.769 | 86.768 | 86.768 | 6 / 10 |
| GRU | 87.184 | 87.188 | 87.184 | 87.184 | 7 / 11 |

Generation uses 161 held-out jokes and 2,961 predicted tokens, including EOS and excluding BOS/PAD.

| Cell | Test cross-entropy | Perplexity | Token accuracy (%) | Best / completed epochs |
|---|---:|---:|---:|---:|
| RNN | 5.395665 | 220.449 | 25.836 | 44 / 56 |
| LSTM | 5.363837 | 213.543 | 27.052 | 62 / 74 |
| GRU | 5.332992 | 207.057 | 28.267 | 46 / 58 |

These are one-seed observations, not estimates of statistical significance. The shared hidden width does not equalize parameter counts. Lower perplexity is better under this fixed vocabulary and split.

**Limitations:** training and validation text groups are disjoint, but the supplied IMDb official test set overlaps with 113 unique normalized training texts and 10 validation texts; the official test set remains unchanged. About 35% of test reviews are truncated. Joke test targets contain 334 unknown tokens out of 2,961 (11.28%), limiting interpretation of perplexity. Generated samples often have recognizable sentence structure but incoherent content or punchlines; the results do not establish reliable joke generation.

## Independent result verification

```bash
.venv-cuda/bin/python script/stage_4_script/verify_stage4_results.py
```

All six runs passed. This checks source/raw-data/split/vocabulary/checkpoint binding, validation checkpoint selection, all 75,000 classification CSV rows and 483 generation CSV rows, small classifier checkpoint replays, and all nine greedy examples. The report is saved as `result/stage_4_result/report/verification.json`. CUDA classifier replay matches within 3e-8; CPU recurrent kernels differ by up to 2.24e-4 with the documented 5e-4 cross-backend tolerance. Parameters are float32 and cuDNN TF32 is allowed.

## Inference

```bash
.venv-cuda/bin/python script/stage_4_script/predict_stage4.py result/stage_4_result/runs/classification_gru_baseline/checkpoints/best.pt --text 'A wonderful movie with excellent acting.' --device cpu
.venv-cuda/bin/python script/stage_4_script/predict_stage4.py result/stage_4_result/runs/generation_gru_baseline/checkpoints/best.pt --text 'What did the' --temperature 0.8 --top-k 20 --seed 1704 --max-new-tokens 40 --device cpu
```

Use `--text-file` instead of `--text` for UTF-8 input, and `--device auto` or `cuda` when desired. Classification reports probabilities, truncation, and unknown tokens, and requires the original nearby `config.json` for its sequence-length limit. Generation requires exactly three words; punctuation is not counted as a word. It preserves the entered prefix and rejects unknown prompt tokens explicitly. Temperature 0 uses greedy decoding. PAD/UNK/BOS are excluded from generated tokens, and EOS ends generation.

## Files and report

- Source: `local_code/stage_4_code/{Dataset_Loader,Method_RNN,Experiment_Runner}.py`; command entry points and this guide: `script/stage_4_script/`; tests: `tests/stage4/`.
- Run evidence: `result/stage_4_result/runs/<task>_<cell>_baseline/`, including `config.json`, `history.json`, `metrics.json`, `test_predictions.csv`, `learning_curve.png`, and generation examples where applicable. Shared environment, split, vocabulary, and dataset metadata are under `result/stage_4_result/runs/`.
- Local model binaries: each run's ignored `checkpoints/best.pt` and `last.pt`.
- Report evidence: `result/stage_4_result/report/`, including `metric_audit.json`, `summary.json`, and `convergence.png`.

The report builder requires ReportLab and pypdf, plus the analysis dependencies above. It reads the **ignored private** `data/stage_4_data/team_information.json`, whose `members` field contains the private roster; do not copy that roster into source files or public artifacts.

```bash
.venv-cuda/bin/python -m pip install -r script/stage_4_script/requirements-report.txt
.venv-cuda/bin/python script/stage_4_script/build_stage4_report.py
```

The builder recomputes metrics from saved test evidence, rejects smoke runs and mismatched scores, and verifies a five-page PDF. Output: `output/pdf/Stage_4_Report_Draft.pdf` (ignored, local only). Team review and the submission source-code link remain pending; no Canvas submission has been made by these scripts.

## Generation follow-up

The coherent-generation profile, six additional model runs, all 12 before/after prompts, blinded assistant ratings, training-copy checks, and the prediction-loss tradeoff are documented in [Generation optimization](GENERATION_OPTIMIZATION.md). Use `predict_generation.py` for the selected profile; `predict_stage4.py` preserves the original baseline interface.

## Generalization and training-copy follow-up

[Generalization experiments](GENERATION_GENERALIZATION.md) document two regularized GRUs, fixed checkpoint mixtures, continuation-level copy checks, and new validation/test prefix panels. The best predictive mixture improves test perplexity to 128.045, but none of the new profiles preserves coherence while passing the strict novelty criterion; the existing default remains unchanged. `predict_novel_generation.py` exposes the retained default plus explicitly experimental predictive and strict-screen profiles. See the guide for commands, both blinded assistant reviews, and the verified tradeoffs.
