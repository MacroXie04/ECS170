# ECS 170 course project

- [Stage 3: CNN image classification](#stage-3-cnn-image-classification)
- [Stage 2: team integration](#stage-2-team-integration)

## Stage 3: CNN image classification

Stage 3 trains CNNs on the instructor's MNIST, ORL, and CIFAR-10 datasets and
compares configurations on each. The code follows the course template:
`local_code/stage_3_code/` holds the dataset, method, setting, evaluate, and
result classes, and `script/stage_3_script/` holds the entry points.

### Install

Use Python 3.12 or newer. For CPU training:

```bash
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, install the same PyTorch version from the CUDA index first.
Blackwell GPUs need CUDA 12.8 or newer, so use `cu130`:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu130 'torch==2.14.1'
python -m pip install -r requirements.txt
```

No dependency was added for Stage 3. Download the instructor's data from its
public link (no account needed) and verify the checksums:

```bash
python script/stage_3_script/prepare_stage3_data.py
```

### Commands

All commands run through `script/stage_3_script/run_stage3.py`. `--device`
accepts `auto` (the default: CUDA, then MPS, then CPU), `cpu`, `cuda`, `cuda:N`,
or `mps`. `--threads` sets the CPU thread count.

| Task | Command |
|---|---|
| List configurations | `python script/stage_3_script/run_stage3.py list` |
| Check data, models, and checkpoints | `python script/stage_3_script/run_stage3.py preflight` |
| Train one dataset and config | `python script/stage_3_script/run_stage3.py train --dataset cifar --config baseline --seed 2026` |
| Run all required experiments | `python script/stage_3_script/run_stage3.py run-all --jobs 2` |
| Resume from a checkpoint | `python script/stage_3_script/run_stage3.py resume --dataset cifar --config baseline --seed 2026` |
| Fix the plan before testing | `python script/stage_3_script/run_stage3.py lock-protocol` |
| Evaluate a checkpoint | `python script/stage_3_script/run_stage3.py evaluate --dataset cifar --config baseline --seed 2026` |
| Reload every checkpoint and compare | `python script/stage_3_script/run_stage3.py verify` |
| Rebuild tables and figures | `python script/stage_3_script/run_stage3.py summarize` |
| Run one dataset's configurations | `python script/stage_3_script/script_cnn_mnist.py` (also `script_cnn_orl.py`, `script_cnn_cifar.py`) |

`train` drops the test split as soon as the file is loaded, so no model or
statistic sees it. `run-all` locks the plan, trains every run, then evaluates. Rerun `run-all` to continue after a stop: finished runs are
skipped and interrupted runs resume from their last epoch. `resume --checkpoint
PATH` continues from a specific file of that run. A run with a saved test result
cannot train further, because the result would no longer describe the model on
disk. `evaluate` saves the first result for a run; later calls compare against that result instead of replacing it.
`evaluate --checkpoint PATH` reports under the run's `evaluations/` folder and
leaves the formal result alone.

To try a change without affecting formal results, use a pilot run:

```bash
python script/stage_3_script/run_stage3.py train --dataset orl --tag try_lr --set learning_rate=0.0005
```

Pilot runs go to `pilot_runs/` and cannot read the test set.

### Protocol

- **Data:** the instructor's train and test splits are used unchanged. A
  stratified validation set comes from the training set only, with split seed
  170, and every run on a dataset shares it.

  | Dataset | Optimization | Validation | Test |
  |---|---:|---:|---:|
  | MNIST | 54,000 | 6,000 | 10,000 |
  | ORL | 320 (8 per person) | 40 (1 per person) | 40 |
  | CIFAR-10 | 45,000 | 5,000 | 10,000 |

- **Preprocessing:** pixels are scaled to [0, 1], then normalized with per-channel
  statistics from the optimization images only. ORL keeps one of its three
  identical channels, and its labels 1–40 map to 0–39. Random shifts and flips
  apply to training batches only.
- **Checkpoint selection:** each run keeps the epoch with the highest validation
  accuracy, then the lowest validation cross-entropy, then the earliest epoch.
- **Configuration selection:** each dataset's final choice has the highest mean
  validation accuracy over seeds, then the lowest mean validation cross-entropy,
  then the fewest parameters. Test results play no part.
- **Fixed in advance:** `protocol.json` records the configurations and rules
  before any test evaluation, and `test_access_log.jsonl` records every model
  evaluation on the test set. `preflight` also loads the test split, but only
  to check its size, shape, labels, and overlap with training images.
- **Reproducibility:** seeds, data order, and augmentation are fixed and restored
  on resume. Exact reruns need the same PyTorch build, hardware, and thread count.
  Results from different devices are not expected to match exactly.

### Results

Each run lives in `result/stage_3_result/runs/<dataset>/<config>/seed_<seed>/`:

| File | Contents |
|---|---|
| `config.json`, `split.json`, `environment.json` | Full configuration and seeds, split and normalization, device and dependency versions |
| `history.csv`, `history.json`, `train.log` | Per-epoch losses, accuracy, and timing; the training log |
| `model_summary.txt` | Layers, output shapes, and parameter counts |
| `test_metrics.json` | Accuracy, macro and weighted precision, recall, and F1, confusion matrix |
| `test_predictions.npz` | True and predicted labels for the full test set |
| `checkpoints/best.pt`, `checkpoints/last.pt` | Selected checkpoint and latest resume state (ignored by git) |

`result/stage_3_result/summary/` holds the combined results: `SUMMARY.md`,
`config_comparison.csv`, `runs.csv`, `best_configs.json`, learning curves and
architecture diagrams in `figures/`, and per-configuration structure text in
`architecture/`. `pilot_runs/` keeps the validation-only trial runs.

`verify` writes `summary/reload_verification.json`. The formal checks were kept
as `reload_verification_cuda.json` and `reload_verification_cpu.json`. The CUDA
check matched all 72 runs exactly. The CPU check matched 71: rounding differences
changed one of 10,000 predictions in `cifar/kernel5/seed_2028`, and `verify`
reports any metric change above 1e-6 as a failure.

The formal results came from one command on an NVIDIA RTX PRO 4500 Blackwell:

```bash
python script/stage_3_script/run_stage3.py run-all --device cuda --jobs 3 --threads 4
```

Training times were measured with three runs sharing the GPU, so they compare
configurations only roughly.

### ORL follow-up

`script/stage_3_followup/run_orl_followup.py` is a supplementary experiment. It
picks an ORL configuration by 9-fold cross-validation on the 360 training images,
refits that configuration on all 360 for three seeds, and scores their ensemble.
Run its stages in order:

```bash
python script/stage_3_followup/run_orl_followup.py register
python script/stage_3_followup/run_orl_followup.py cv --jobs 3
python script/stage_3_followup/run_orl_followup.py refit
python script/stage_3_followup/run_orl_followup.py test
```

It was proposed after the formal ORL test scores were known, so its result is not
a blind estimate and does not replace the formal results. Results are in
`result/stage_3_result/orl_followup/`; start with `FOLLOWUP.md`. Its test read is
logged in that folder's own `test_access_log.jsonl`.

### Tests

```bash
python -m unittest discover -s tests/stage3
```

## Stage 2: team integration

`Dev` brings the four members' implementations together while preserving their
original source files and saved experiments. Each implementation retains its own
architecture, training settings, and evaluation protocol. This integration does
not replace them with a single algorithm or change `main`.

| Member | Directory | Source branch | Source commit |
|---|---|---|---|
| Hongzhe Xie | [implementations/hongzhe](implementations/hongzhe) | `stage2-mlp` | [`2bb9c8e`](https://github.com/MacroXie04/ECS170/commit/2bb9c8eccb24ac59f953bf5567867cc0be54b348) |
| Aileen Zheng | [implementations/aileen](implementations/aileen) | `stage-2-tuned-mlp` | [`910e5f4`](https://github.com/MacroXie04/ECS170/commit/910e5f46033b79446acf03b137442bc085a1b7be) |
| Zizheng Wang | [implementations/zizheng](implementations/zizheng) | `Zz_Wang` | [`207892a`](https://github.com/MacroXie04/ECS170/commit/207892a3d0a596120f303452a67fa74acf692ca9) |
| Xiaoyu Feng | [implementations/xiaoyu](implementations/xiaoyu) | `xy` | [`9c00e38`](https://github.com/MacroXie04/ECS170/commit/9c00e389051d90d6fe9c6fef7efdee41c83daf82) |

Xiaoyu's runnable project remains inside
`implementations/xiaoyu/ECS170_Fall_2026_Source_Code_Template/`.
The member directories are complete source snapshots, including original
documentation and archived outputs. Their README paths are relative to their
own project roots. `provenance.json` records the source files and their hashes;
`python script/integration/verify_sources.py --check-history` checks preservation
and confirms that all four source commits are in the integrated branch history.

### Run with shared data

Use Python 3.13 and install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

The following command reconstructs the instructor CSVs from public MNIST
archives and verifies their exact checksums. It can also verify CSVs already
placed in the destination. The dataset is excluded from version control.

```bash
python implementations/hongzhe/script/ci/prepare_mnist.py --output-dir data/stage_2_data
```

Run all four implementations against that same data:

```bash
python run_stage2.py --implementation all --data-dir "$PWD/data/stage_2_data" --output-dir runs/full
```

To run one member, replace `all` with `hongzhe`, `aileen`, `zizheng`, or `xiaoyu`.
Use a new output directory for each run. The launcher runs each member in a
separate temporary project, supplies the shared data, and collects fresh outputs
under the requested output directory. Archived results in `implementations/`
are preserved and are not evidence that a new run has completed.

The original protocols differ: Hongzhe runs four 15-epoch configurations;
Aileen runs the tuned 25-epoch model; Zizheng performs validation-based model
selection followed by a full-training-set refit; Xiaoyu runs five 10-epoch
configurations. See each member's source and saved configurations for details.
Cross-member differences are not controlled ablations, and new results can
differ slightly with the execution platform.

### Automated checks

- **Dev integration:** pushes to `Dev` and pull requests targeting `Dev` or
  `main` verify snapshot preservation, run the original Hongzhe and Zizheng
  unit tests, check the shared launcher, and smoke-test all four members.
  Smoke tests check that training and evaluation work on small inputs; they
  do not report official test accuracy.
- **Team full training:** pull requests targeting `Dev` or `main` run four
  independent jobs, one for each member's complete original protocol. Each job
  trains from scratch on checksum-verified course data and validates its new
  predictions and training records. A separate bot comment for each member
  reports the commit, settings, accuracy, macro F1, and artifact link. Re-runs
  update that member's existing comment; one failure does not stop the others.
  Zizheng's seven validation results are labeled separately from the final
  10,000-example test result. Each member's results and logs are retained for
  seven days. Fork PRs run training but cannot post comments with a write token.

Only the workflows under the root `.github/workflows/` are active. Copies inside
member snapshots are preserved historical files. Direct pushes to `Dev` run the
integration checks without starting the full-dataset training workflow.

## Stage 4: recurrent text models

Stage 4 implements IMDb sentiment classification and short-joke generation with RNN, LSTM, and GRU models. The six full GPU runs, validation-selected checkpoints, prediction evidence, and a five-page working report are complete on the GPU server.

See [the Stage 4 guide](script/stage_4_script/README.md) for data preparation, training/resume, inference, tests, result verification, and report generation. Instructor datasets, checkpoints, and the private report PDF remain local and are excluded from Git.

The [generation optimization follow-up](script/stage_4_script/GENERATION_OPTIMIZATION.md) adds a more coherent decoding profile and documents its memorization/generalization tradeoff.
