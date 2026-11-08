# ECS170 - Stage 2: MLP Classification

**Final test accuracy: 98.26% (9,826/10,000).** Macro Precision / Recall / F1:
**0.982615 / 0.982334 / 0.982451**.

## Changes from the course template

Stage 1 and the base classes are unchanged. Stage 2 adapts the loader to the supplied
CSV files (label first, 784 features), scales features by 255, and uses logits with
cross-entropy. Training uses `fit()` to preserve PyTorch's `train()`/`eval()` modes.
Seven configurations compare network depth, loss, optimizer and regularization.
A fixed 54,000/6,000 split within the training set selects the configuration and epoch;
a fresh model then trains on all 60,000 training examples before final testing.
The instructor's test set is kept separate; no cross-validation or test-based tuning is used.

The template saves predictions with **Pickle**. This implementation currently uses
**NumPy NPZ** for predictions, **JSON** for settings/metrics, **CSV** for training history,
and **PyTorch PT** for model parameters. These are storage-format changes, not changes
to the evaluation task.

## Where to review the results

All recorded experiments are under [`result/stage_2_result/`](result/stage_2_result/).
Each named experiment folder contains its training/validation results; `final/`
contains the model refitted on the full training set and its test results.
The report is submitted separately.

| File | Contents / how to view |
| --- | --- |
| [comparison.csv](result/stage_2_result/comparison.csv) | Seven-configuration validation comparison; open in a spreadsheet or text editor. |
| [final/record.json](result/stage_2_result/final/record.json) | Final configuration, training/test metrics and per-class results; open as text. |
| [final/history.csv](result/stage_2_result/final/history.csv) | Per-epoch training loss and accuracy; open in a spreadsheet or text editor. |
| `*/learning_curve.png`, `final/confusion_matrix.png`, `comparison.png` | Open the images directly. |
| [final/predictions.npz](result/stage_2_result/final/predictions.npz) | Saved true/predicted labels; load with NumPy (example below). |
| [final/model.pt](result/stage_2_result/final/model.pt) | Load with `torch.load(path, map_location='cpu', weights_only=True)`; contains `state_dict`, `config`, `input_dim`, `classes` and preprocessing. |

From the repository root, inspect saved test predictions without retraining:

```python
import numpy as np
with np.load('result/stage_2_result/final/predictions.npz') as result:
    print(result['test_true'][:10], result['test_pred'][:10])
    print('Accuracy:', (result['test_true'] == result['test_pred']).mean())
```

## Run / verify

Place the supplied `train.csv` and `test.csv` in `data/stage_2_data/`. Run from the
repository root with Python 3.9 or later:

```bash
python -m pip install torch numpy scikit-learn
python -m unittest discover -s tests -v
python tests/audit_stage2.py
python script/stage_2_script/script_mlp.py baseline --output result/new_run
python script/stage_2_script/script_mlp.py compare --output result/new_run
python script/stage_2_script/script_mlp.py final --output result/new_run
```

The audit only reads the original records, checks their historical source hash,
and verifies saved metrics and checkpoint predictions. Use a new output directory
for reproduction; the reported artifacts and their original hashes are preserved.
