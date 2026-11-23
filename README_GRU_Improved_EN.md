# Stage 4 — Improved GRU for IMDb Sentiment Classification

This experiment adds a separate GRU-based sentiment classifier without modifying the earlier RNN, LSTM, or GRU runs.

## Architecture

Reviews are tokenized using the existing dataset loader. For reviews longer than 200 tokens, the first and last 100 tokens are retained. The model uses 50-dimensional GloVe embeddings, a small 1D convolution, temporal pooling at stride 5, and a bidirectional GRU with 96 hidden units in each direction. Masked mean and max pooling, final GRU states, and attention-weighted convolutional features are concatenated for classification.

Training uses cross-entropy loss, Adam, gradient clipping, a validation-loss-based learning rate scheduler, and early stopping. The best checkpoint is selected using the validation split.

## Data

- Training: 22,500 IMDb reviews
- Validation: 2,500 IMDb reviews
- Testing: 25,000 IMDb reviews
- Stratified training/validation split, seed 2
- Vocabulary built from the training split only

## Running

Open the project folder in PyCharm and run:

```
script/stage_4_script/script_gru_improved.py
```

When the full-project embedding cache is present, the script uses its local 50-dimensional GloVe embeddings. Alternatively, supply the GloVe archive using `--glove PATH_TO_GLOVE_ZIP` in Script parameters. Without GloVe vectors, embeddings are initialized randomly and results may differ.

To re-evaluate the saved checkpoint on the complete IMDb test set without retraining, run:

```
script/stage_4_script/script_gru_evaluate.py
```

## Results

Saved experiment files are in `result/stage_4_result/gru_conv_pool_glove/`, including `model_weights.pt`, `metrics.json`, `training_history.csv`, `learning_curve.png`, `predictions.npz`, and the run configuration.
