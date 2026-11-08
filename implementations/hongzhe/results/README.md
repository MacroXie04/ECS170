# Stage 2 saved experiment results

This directory contains the saved outputs from four MLP experiments on the
instructor-provided training and test splits. Each configuration used seed 42,
15 epochs, a batch size of 256, and CPU execution. The test set has 10,000 examples.

| Configuration | Test accuracy |
| --- | --- |
| Shallow Adam + CE | 97.69% |
| Deep Adam + CE | 98.05% |
| Deep SGD + CE | 98.27% |
| Deep Adam + MSE | 97.98% |

## Files

- `experiments.json`: configurations, environment, data hashes, training losses,
  timings, parameter counts, metrics, and confusion matrices.
- `*.pt`: one checkpoint per configuration, containing its configuration and
  PyTorch model state dictionary.
- `*_predictions.npz`: the 10,000 true labels (`true_y`) and predicted labels
  (`pred_y`) for each configuration.
- `convergence.png`: training-loss curves, with cross-entropy and MSE in separate
  panels because their scales differ.
- `confusion.png`: test confusion matrix for the deep Adam + CE reference model.
- `validation.txt`: the saved checkpoint and metric validation note.

## Verification

Before publishing this snapshot, each checkpoint was loaded with the Stage 2
implementation and reproduced all 10,000 archived test predictions. The test
data hash matched `experiments.json`. Independently recomputed accuracy, macro
and weighted precision/recall/F1, and confusion matrices matched the saved
metrics. Each run contains 15 finite training-loss values.

This verification performed inference only; it did not retrain the models or
change the saved results. The empty training log is omitted. The instructor's
input CSV files remain excluded from the repository.

These are single-seed results. Architecture comparisons change both depth and
width, and the Adam/SGD configurations use different learning rates, so the
observed ranking does not establish that one optimizer is generally superior.
