# ECS170 - Zz_Wang Stage 2

Independent PyTorch MLP implementation following the course module framework.
Development is on the local `Zz_Wang` branch. The completed implementation is saved
as a local commit and synchronized to the desktop repository's `Zz_Wang` branch.
The instructor data and all original Stage 1 code are preserved; no push or main-branch merge has been performed.

- Start here: [Chinese walkthrough](STAGE2_GUIDE_ZH.md)
- [Real experiment results](STAGE2_RESULTS_ZH.md)
- [Four-page report draft](reports/Stage2_Report_Draft.pdf)
- [Editable report generator](reports/build_report.py)
- [Complete final record](result/stage_2_result/final/record.json)

Final locked-model test accuracy: **98.26%** (9,826 / 10,000).
Test macro Precision / Recall / F1: 0.982615 / 0.982334 / 0.982451.
Seven configurations were compared using a fixed validation holdout from the instructor's training set.
No test-based hyperparameter selection or cross-validation was used.
Team information and a published source link must be supplied before report submission.

## Reproduce

Place the provided `train.csv` and `test.csv` in `data/stage_2_data/` (not distributed in the code archive).
From this repository root, with Python 3.9-3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-stage2.txt
python -m unittest discover -s tests -v
python script/stage_2_script/script_mlp.py baseline --output result/my_reproduction
python script/stage_2_script/script_mlp.py compare --output result/my_reproduction
python script/stage_2_script/script_mlp.py final --output result/my_reproduction
python script/stage_2_script/script_mlp.py plot --output result/my_reproduction
```

Keep the original `result/stage_2_result` unchanged as the record of the reported run.
The supplied audit script `python tests/audit_stage2.py` verifies that original run.
All original Stage 1 code and base classes remain unchanged.
