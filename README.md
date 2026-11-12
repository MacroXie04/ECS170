# ECS 170 — Stage 2 and Stage 3 team implementations

## Stage 3

The four Stage 3 submissions are integrated as separate runnable projects. Their
original commits remain in Git history, and their source files and committed
results are preserved byte-for-byte. Each project keeps its own CNNs, training
protocol, and evaluation code; the shared root is not assigned to one member.

| Member | Stage 3 project | Source PR | Source commit |
|---|---|---|---|
| Hongzhe Xie | [implementations/hongzhe/stage3](implementations/hongzhe/stage3) | [#6](https://github.com/MacroXie04/ECS170/pull/6) | `41cb6f0` |
| Xiaoyu Feng | [implementations/xiaoyu/stage3](implementations/xiaoyu/stage3) | [#7](https://github.com/MacroXie04/ECS170/pull/7) | `011ea2c` |
| Zizheng Wang | [implementations/zizheng/stage3](implementations/zizheng/stage3) | [#8](https://github.com/MacroXie04/ECS170/pull/8) | `5d23e34` |
| Aileen Zheng | [implementations/aileen/stage3](implementations/aileen/stage3) | [#9](https://github.com/MacroXie04/ECS170/pull/9) | `4e49b4e` |

Each directory contains its source branch tree except the inherited
`implementations/` archives, which already exist at the repository root.
Supporting base classes and Stage 2 modules are included because some Stage 3
implementations import them. Nested workflow files are historical copies and
are not active workflows. Inherited Stage 2 integration commands in these
snapshots should be run from the repository root instead.

`provenance_stage3.json` records all 1,458 imported files and source commits.
The earlier Stage 2 manifest and its 350 files remain intact. Verify both with:

```bash
python script/integration/verify_sources.py --check-history
python script/integration/verify_sources.py --manifest provenance_stage3.json --check-history
```

### Reproduce Stage 3

Install the root `requirements.txt` using Python 3.13. Keep archived source
snapshots unchanged by copying the selected project into the ignored `runs/`
directory before training. For example, from the repository root:

```bash
mkdir -p runs/stage3
cp -R implementations/xiaoyu/stage3 runs/stage3/xiaoyu
cd runs/stage3/xiaoyu
# Place the instructor's MNIST, ORL and CIFAR files in data/stage_3_data/.
python script/stage_3_script/script_experiments.py
```

Use a new destination for each copy. The commands below are relative to that
member's copied project root; see each original README for full settings.

| Member | Commands |
|---|---|
| Hongzhe | `python script/stage_3_script/run_stage3.py run-all --result-root result/fresh_stage3` |
| Xiaoyu | `python script/stage_3_script/script_experiments.py` |
| Zizheng | `python script/stage_3_script/script_cnn.py compare --output result/fresh_stage3`, then `python script/stage_3_script/script_cnn.py final --output result/fresh_stage3` |
| Aileen | `python script/stage_3_script/script_cnn.py MNIST` (repeat with `ORL` and `CIFAR`); `python script/stage_3_script/script_ablation.py` for the configuration study |

Hongzhe's archived status files mark completed runs, while checkpoints are
excluded from Git. A fresh `--result-root` is necessary to train anew;
checkpoint verification must use that same newly trained result directory.
Zizheng's fresh output directory similarly keeps archived validation/final
records separate. Aileen's original ablation script reuses the per-dataset
prediction filename across configurations; retain separate runtime copies if
individual configuration prediction files are needed.

The original experimental protocols remain unchanged. Hongzhe and Zizheng use
validation-based selection; Xiaoyu and Aileen compare configurations using test
results, so those comparisons are descriptive. This merge does not rerun full
training or create new official benchmark scores.

### Stage 3 checks

The Stage 3 integration workflow verifies both source manifests and ancestry,
runs the original Hongzhe and Zizheng unit tests, and smoke-tests Xiaoyu and
Aileen on synthetic inputs for MNIST, ORL and CIFAR. Data-dependent tests may
skip when the instructor datasets are absent. Synthetic checks validate that
training/evaluation runs and are not accuracy benchmarks.

## Stage 2

The Stage 2 integration preserves the four original implementations and saved
experiments. Each retains its own architecture, settings, and evaluation protocol.

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

## Run with shared data

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

## Automated checks

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
