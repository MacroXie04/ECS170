# ECS 170 Stage 4 | Zizheng Wang

We compare one-layer RNN, LSTM and GRU models on IMDb sentiment classification
and short-joke generation. The best checkpoint for each run is selected by
validation cross-entropy before evaluation on its test set.

| Cell | IMDb test accuracy | IMDb macro F1 | Joke test perplexity | Joke next-token accuracy |
| --- | ---: | ---: | ---: | ---: |
| RNN | 84.03% | 0.8402 | 39.66 | 34.68% |
| LSTM | 84.92% | 0.8491 | 37.51 | 36.10% |
| GRU | 85.18% | 0.8517 | 36.33 | 34.80% |

The baseline classifiers read the first 200 tokens of each review. In a
[follow-up GRU run](../../result/stage_4_result/head_tail/), reviews longer than
200 tokens kept their first 100 and last 100 tokens. With the same data split
and training settings, test accuracy rose from 85.18% to 86.68% and macro F1
rose from 0.8517 to 0.8668. The run records the input window in both the
checkpoint and `record.json`.

IMDb uses 22,500 training reviews, 2,500 stratified validation reviews and the
25,000 official test reviews. Tokenized jokes are deduplicated before splitting
into 1,284 training, 161 validation and 161 test examples. For generation,
the first three tokens are supplied as context and loss begins at the fourth
token. Test perplexity includes end-of-sequence and unknown-word targets.

The [classification results](../../result/stage_4_result/classify/) and
[generation results](../../result/stage_4_result/generate/) contain combined learning
curves. Each model directory contains the settings and metrics (`record.json`),
epoch history (`history.csv`), checkpoint with vocabulary (`model.pt`) and
either IMDb test predictions (`predictions.npz`) or three completions with
training-corpus comparisons (`samples.json`). The report is submitted
separately.

The Stage 4 requirements file records the package versions used with Python
3.9.6 for the reported runs. Put the instructor dataset in `data/stage_4_data/`,
with the `text_classification/` and `text_generation/` folders directly inside.

From the repository root:

```bash
python -m pip install -r script/stage_4_script/requirements.txt
python script/stage_4_script/script_rnn.py --smoke --output /tmp/ecs170-stage4-smoke
python script/stage_4_script/script_rnn.py --task classify --threads 2 --output /tmp/ecs170-stage4
python script/stage_4_script/script_rnn.py --task generate --threads 1 --output /tmp/ecs170-stage4
python script/stage_4_script/script_rnn.py --task classify --model GRU --review-window head-tail --threads 2 --output /tmp/ecs170-stage4-head-tail
```

Use `--data` to point to a different dataset directory. Use `--model RNN`,
`--model LSTM` or `--model GRU` to run one cell type. Choose a fresh output
folder for reruns.
