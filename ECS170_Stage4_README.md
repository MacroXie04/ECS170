# ECS 170 Fall 2026 — Stage 4: RNN Text Classification and Generation

This is the Xiaoyu implementation, preserved under local_code/stage_4_code/xiaoyu/ in stage4-dev. Its command scripts remain at the paths below. See [Stage 4 DEV integration](script/stage_4_script/STAGE4_DEV_INTEGRATION.md) for the other team implementations.

## Data

- `data/stage_4_data/text_classification/`: IMDb movie-review sentiment dataset. The provided split contains 25,000 training and 25,000 testing reviews, divided into `pos` and `neg` folders. This is binary sentiment classification, not rating prediction.
- `data/stage_4_data/text_generation/`: 1,622 short jokes from the provided CSV file. The generator learns next-word prediction and extends three supplied starting words.
- The optional GloVe 2024 50-dimensional word-vector ZIP can initialize the embedding layer. Its download is separate, and the project does not require the large ZIP to run.

## Requirements

Python 3, `torch`, `numpy`, `scikit-learn`, `matplotlib`:

```bash
python -m pip install torch numpy scikit-learn matplotlib
```

Use PyCharm to open the project folder. Script paths are resolved from the script location, so the working directory does not need to be changed.

## Run classification

Open `script/stage_4_script/script_classification.py` in PyCharm and run it, or use:

```bash
python script/stage_4_script/script_classification.py --model rnn --epochs 5
python script/stage_4_script/script_classification.py --model lstm --epochs 5
python script/stage_4_script/script_classification.py --model gru --epochs 5
```

The default configuration uses a 50-dimensional trainable embedding, a 128-unit recurrent layer, Adam, learning rate 0.001, batch size 64, and a 180-token input limit. Padding is ignored using packed sequences. The vocabulary comes only from training reviews; the provided train/test split is not altered. Output labels are `0=negative` and `1=positive`.

Each run writes `metrics.json`, `training_history.csv`, `learning_curve.png`, `predictions.pkl`, `classification_report.txt`, `confusion_matrix.csv`, `run_config.json` and `model_weights.pt` in `result/stage_4_result/classification_<model>/`.

## Run text generation

Open `script/stage_4_script/script_generation.py` in PyCharm, or use:

```bash
python script/stage_4_script/script_generation.py --model rnn --epochs 15 --start-words "what did the"
python script/stage_4_script/script_generation.py --model lstm --epochs 15 --start-words "what did the"
python script/stage_4_script/script_generation.py --model gru --epochs 15 --start-words "what did the"
```

The script trains next-token prediction using the jokes, outputs generated samples and matching examples from the source dataset, and saves loss curves and model weights under `result/stage_4_result/generation_<model>/`. The three input words are supplied by `--start-words`.

## Compare all models

```bash
python script/stage_4_script/script_experiments.py --task both
```

You can run only one task with `--task classification` or `--task generation`. These commands train new models: no final test accuracies are precomputed or implied.

## Optional GloVe embeddings

Keep `glove.2024.wikigiga.50d.zip` somewhere on your computer. Pass its full path when running a script:

```bash
python script/stage_4_script/script_classification.py --model gru --glove "C:/your/path/glove.2024.wikigiga.50d.zip"
```

The program reads the ZIP directly and extracts vectors only for words present in the task vocabulary. The selected embedding matrix is cached under `result/stage_4_result/embedding_cache/`. To compare fairly, keep the other parameters unchanged.

## Small environment check (not a report result)

```bash
python script/stage_4_script/script_classification.py --train-limit 64 --test-limit 32 --epochs 1 --run-name smoke_classification
python script/stage_4_script/script_generation.py --epochs 1 --run-name smoke_generation
```

The classification smoke test uses only a small subset. Its accuracy must not be presented as a result for the full IMDb dataset. Text generated from an untrained or one-epoch model is expected to be poor.

## Report requirements

The report is limited to five pages. Include task descriptions, model diagrams and settings, dataset descriptions, Accuracy/Precision/Recall/F1 for IMDb, training loss curves for both tasks, generated joke examples with comparisons to training examples, and RNN/LSTM/GRU comparisons.
