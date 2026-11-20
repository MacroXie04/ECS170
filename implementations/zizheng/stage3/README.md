# ECS170 Course Project

## Stage 3: CNN image classification

Final results on the instructor's test partitions:

| Dataset | Test accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | --- | --- | --- | --- |
| MNIST | 99.08% (9,908/10,000) | 0.990787 | 0.990643 | 0.990690 |
| ORL | 97.50% (39/40) | 0.962500 | 0.975000 | 0.966667 |
| CIFAR | 77.74% (7,774/10,000) | 0.791596 | 0.777400 | 0.781089 |

Stage 3 follows the course module framework in `local_code/stage_3_code/`.
The loader converts images to NCHW float32 tensors with pixels divided by 255 and maps ORL
identities 1-40 to indices 0-39, keeping one of its identical grayscale channels.
Five independent configurations compare convolution depth, pooling, hidden width
and loss. Each dataset uses one fixed stratified holdout within its training set.
Validation selects the configuration and epoch; a fresh model then trains on the
full training partition before testing. Test results are excluded from selection.
Only CIFAR training batches use random crops and horizontal flips.

### Review the results

Open [MNIST](result/stage_3_result/MNIST/), [ORL](result/stage_3_result/ORL/)
or [CIFAR](result/stage_3_result/CIFAR/) under `result/stage_3_result/`.
The report is submitted separately.

| File within each dataset folder | Contents / how to view |
| --- | --- |
| `comparison.csv`, `comparison.png` | Five-configuration validation comparison; open as a table or image. |
| `selection.json` | Validation selection and locked final epoch count. |
| `*/record.json`, `*/history.csv` | Settings, source/data hashes, metrics and epoch history; open as text or a table. |
| `*/learning_curve.png`, `final/confusion_matrix.png` | Open images directly. |
| `final/predictions.npz` | NumPy arrays `train_true`, `train_pred`, `test_true`, `test_pred`; labels use class indices. |
| `final/model.pt` | Checkpoint with `state_dict`, `config`, `input_shape`, original `classes`, `dataset` and preprocessing. |

Prediction storage uses NPZ rather than the course template's Pickle format.
Settings and metrics use JSON, histories use CSV, and model weights use PT.
Load checkpoints with `torch.load(path, map_location='cpu', weights_only=True)`.
Map class indices back to the instructor labels with `checkpoint['classes']`.
Existing experiment folders are protected against overwriting.

### Run and verify

Use Python 3.9 or later. Place the supplied files `MNIST`, `ORL`, and `CIFAR`
in `data/stage_3_data/`. From the repository root:

```bash
python -m pip install torch numpy scikit-learn matplotlib
python -m unittest discover -s tests -v
python tests/audit_stage3.py
python script/stage_3_script/script_cnn.py compare --output result/new_stage3
python script/stage_3_script/script_cnn.py final --output result/new_stage3
python tests/audit_stage3.py --output result/new_stage3
```

`--dataset MNIST`, `ORL` or `CIFAR` runs one dataset; the default is all three.
The default device is CPU with two threads. The audit reads saved results,
recalculates metrics, checks shared splits and selection, and reproduces validation
and final test predictions from checkpoints. Use a new output folder to retrain;
leave the reported artifacts unchanged. Figures are included as saved images.

## Stage 2: MLP classification

The earlier Stage 2 implementation and its recorded experiments are preserved.
Final test accuracy: **98.26% (9,826/10,000)**. Macro Precision / Recall / F1:
**0.982615 / 0.982334 / 0.982451**.
See [comparison](result/stage_2_result/comparison.csv),
[final metrics](result/stage_2_result/final/record.json), and the images in
[result/stage_2_result/](result/stage_2_result/). Its report is submitted separately.
Stage 1 and the course base classes are unchanged.

For Stage 2, put `train.csv` and `test.csv` in `data/stage_2_data/`:

```bash
python tests/audit_stage2.py
python script/stage_2_script/script_mlp.py compare --output result/new_stage2
python script/stage_2_script/script_mlp.py final --output result/new_stage2
```
