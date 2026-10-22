# ECS170

## Stage 2 MLP experiments

The Stage 2 implementation follows the instructor-provided dataset, method, and
setting interfaces. It trains a PyTorch MLP for ten-class classification with the
instructor's fixed training and test splits.

### Setup and run

From the repository root, install the dependencies in a Python environment:

```sh
python -m pip install -r requirements.txt
```

Place the instructor-provided `train.csv` and `test.csv` in `data/`. Each CSV row
contains the label first, followed by 784 pixel values. The loader expects 60,000
training rows and 10,000 test rows, normalizes pixels to `[0, 1]`, and preserves
the supplied train/test split.

Run all four configurations:

```sh
python script/stage_2_script/run_experiments.py --device cpu
```

Use `--device mps` or `--device cuda` only when the corresponding device is
available. The recorded dependency versions were used with Python 3.13.12.

### Configurations

All configurations use seed 42, 15 epochs, and batches of 256.

| Name | Hidden layers | Optimizer | Learning rate | Loss |
| --- | --- | --- | --- | --- |
| shallow_adam_ce | 128 | Adam | 0.001 | Cross-entropy |
| deep_adam_ce | 256, 128 | Adam | 0.001 | Cross-entropy |
| deep_sgd_ce | 256, 128 | SGD with momentum 0.9 | 0.05 | Cross-entropy |
| deep_adam_mse | 256, 128 | Adam | 0.001 | MSE on softmax probabilities |

The runner writes model checkpoints, per-example test predictions, metrics,
training-loss histories, convergence plots, and a confusion matrix to `results/`.
Metrics include accuracy and macro/weighted precision, recall, and F1. Test
predictions are evaluated after training. Cross-entropy and MSE loss curves use
separate panels because their scales differ.

### Source files

- `local_code/stage_2_code/Dataset_Loader.py`: CSV validation and normalization.
- `local_code/stage_2_code/Method_MLP.py`: network construction, training, prediction.
- `local_code/stage_2_code/Setting_Prepartitioned.py`: fixed-split execution.
- `local_code/stage_2_code/Evaluate_Multiclass.py`: multiclass metrics.
- `script/stage_2_script/run_experiments.py`: experiment configurations and outputs.

The instructor dataset and generated outputs are excluded from version control.
The existing Stage 1 source and instructor base classes remain in their original
repository locations.
