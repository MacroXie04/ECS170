# ECS 170 — Stage 2 team integration

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
