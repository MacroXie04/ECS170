# ECS 170 Stage 2: Tuned MLP

A PyTorch multilayer perceptron for the instructor-provided digit dataset (784 input features, 10 classes), built on the course template. The tuned model reaches **98.75% test accuracy** in the included script (seed 2); the average over seeds 1, 2 and 3 was 98.82% ± 0.06.

## Model

- Hidden layers [512, 256], each Linear → BatchNorm → ReLU, then 10 output logits (537,354 parameters)
- Loss: cross-entropy with label smoothing 0.1 (no output Softmax; the loss applies log-softmax itself)
- Optimizer: SGD with momentum 0.9, learning rate 0.05
- Learning-rate schedule: cosine decay over 25 epochs
- Batch size 128, no dropout

Hyperparameters were chosen on a validation split held out from the training data, not on the test set.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the instructor's `train.csv` and `test.csv` in `data/stage_2_data/`. The dataset is not committed to this repository.

## Run

```bash
cd script/stage_2_script
python script_mlp_tuned.py
```

The script trains for 25 epochs on all 60,000 training images, prints test accuracy plus precision, recall and F1 (macro, weighted and micro), and saves:

- `result/stage_2_result/convergence_curve_tuned.png` (training loss and training accuracy per epoch)
- `result/stage_2_result/MLP_tuned_prediction_result_None` (predictions)

Set `method_obj.use_gpu = True` in the script to use CUDA or Apple MPS if available.

## Results

| Run | Test accuracy | Macro F1 |
|---|---|---|
| `script_mlp_tuned.py` (seed 2) | 98.75% | 0.9875 |
| Mean of seeds 1, 2, 3 | 98.82% ± 0.06 | 0.9881 |

The three-seed average comes from a separate experiment script that is not part of this pull request.

## Code layout

- `local_code/stage_2_code/Dataset_Loader.py`: reads the CSVs and scales pixels to [0, 1]
- `local_code/stage_2_code/Method_MLP.py`: configurable MLP (layers, activation, batch norm, dropout, loss, optimizer, schedule)
- `local_code/stage_2_code/Setting_Pre_Split.py`: uses the instructor's train and test files
- `local_code/stage_2_code/Evaluate_Metrics.py`: accuracy, precision, recall and F1
- `local_code/stage_2_code/Result_Saver.py`, `Result_Loader.py`: save and load predictions
- `script/stage_2_script/script_mlp_tuned.py`: runs the tuned model
