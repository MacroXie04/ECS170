# ECS 170 — Stage 2

PyTorch MLP for classifying handwritten digits. We started from the Stage 1 code template and kept its original files in `local_code/stage_1_code/` and `script/stage_1_script/`. Our Stage 2 work is in the corresponding `stage_2_code` and `stage_2_script` folders.

## Data

The instructor provided `data/stage_2_data/train.csv` (60,000 rows) and `test.csv` (10,000 rows). Each row has a label from 0 to 9 followed by 784 pixel values. The loader reads these files separately and scales the pixels by 255 when training. There is no extra train/test split or cross-validation.

## Files

| File | Use |
|---|---|
| `local_code/stage_2_code/Dataset_Loader.py` | Reads both CSV files. |
| `local_code/stage_2_code/Method_MLP.py` | Builds and trains the MLP. |
| `local_code/stage_2_code/Setting_Train_Test_Split.py` | Connects the provided partitions to training, prediction and evaluation. |
| `local_code/stage_2_code/Evaluate_Accuracy.py` | Accuracy, multiclass precision, recall and F1. |
| `local_code/stage_2_code/Result_Saver.py` | Saves labels and predictions. |
| `script/stage_2_script/script_mlp.py` | Main training and testing script. |
| `script/stage_2_script/script_experiments.py` | Runs several configurations and creates a comparison CSV. |

## Run

Open the project folder in PyCharm. The Python interpreter needs `torch`, `numpy`, `scikit-learn` and `matplotlib`.

```bash
python -m pip install torch numpy scikit-learn matplotlib
```

Run `script/stage_2_script/script_mlp.py` in PyCharm, or use:

```bash
python script/stage_2_script/script_mlp.py
```

The default model is `784 -> 128 -> 10`, with Adam, cross-entropy, learning rate 0.001, batch size 256 and 10 epochs. The script automatically finds the project folder. Training uses mini-batches; the network returns logits rather than applying Softmax before cross-entropy.

For a different layer size or epoch count:

```bash
python script/stage_2_script/script_mlp.py --hidden-dims 256 128 --epochs 15 --run-name deeper_15
```

To run the five parameter comparisons:

```bash
python script/stage_2_script/script_experiments.py
```

You can select one experiment by adding `--only mse_loss` or `--only lower_lr`.

For a quick code check without a full training run:

```bash
python script/stage_2_script/script_mlp.py --train-limit 256 --test-limit 100 --epochs 1 --run-name test_run
```

Don't use the small-data test result in the report. The official experiments use the full CSV files.

## Saved results

Each experiment writes files to `result/stage_2_result/<run_name>/`:

- `metrics.json`: accuracy and macro/micro/weighted precision, recall and F1
- `training_history.csv`: training loss and accuracy by epoch
- `learning_curve.png`: loss and training accuracy curves
- `predictions.pkl`: predicted and true labels
- `confusion_matrix.csv` and `classification_report.txt`: results for each digit
- `mlp_weights.pt`: model weights
- `run_config.json`: exact run settings

This project includes these saved full-dataset runs:

| Folder | Hidden layers | Loss | Optimizer / LR | Saved accuracy |
|---|---|---|---|---:|
| `baseline` | 128 | Cross-entropy | Adam / 0.001 | 97.48% |
| `deeper` | 256, 128 | Cross-entropy | Adam / 0.001 | 97.73% |
| `sgd_lr_001` | 128 | Cross-entropy | SGD / 0.01 | 89.90% |
| `mse_loss` | 128 | MSE | Adam / 0.001 | 97.29% |
| `lower_lr` | 128 | Cross-entropy | Adam / 0.0003 | 95.71% |

The baseline was run on the full dataset with Adam, learning rate 0.001, 10 epochs, batch size 256 and seed 2. Its saved test accuracy is **97.48%** and macro F1 is **97.46%**. The model settings and results are saved in the `baseline/` folder.

This archive contains the first-version Stage 2 scripts and results. The separate four-hidden-layer 10/25/40-epoch experiments from our later report are **not included in this uploaded RAR**.

## Project report

The Stage 2 report follows the instructor's sections: task, model architecture, dataset, settings, metrics, training convergence, performance and ablation studies. It must be no more than five pages. The code remains available for checking through the project repository. Use the actual result files and settings when preparing the final report; don't mix results from different configurations.

A fixed random seed helps with reproducibility, but small differences can still occur across machines and PyTorch versions. Ideally, choose hyperparameters using a validation set from the training data rather than repeatedly selecting models by test-set accuracy.
