# ECS 170 Stage 3: CNN Image Classification

This project uses convolutional neural networks (CNNs) to classify MNIST handwritten digits, ORL faces, and CIFAR-10 objects. We test three model configurations on each dataset, for a total of nine experiments. The provided train/test splits are unchanged. Pixel values are divided by 255; ORL uses its first image channel, with labels mapped from 1–40 to 0–39.

## How to Run

Put the provided datasets in `data/stage_3_data/` and install the dependencies:

```shell
python -m pip install -r requirements.txt
```

Run all experiments and generate the plots:

```shell
python script/stage_3_script/script_experiments.py
python script/stage_3_script/script_plot_results.py
```

To run a single dataset, use `script_mnist.py`, `script_orl.py`, or `script_cifar.py` in `script/stage_3_script/`. These scripts can also be run directly in PyCharm.

## Model and Settings

Each convolution block uses convolution, ReLU, and 2×2 pooling (stride 2), followed by flattening, a fully connected layer of 128 units, and an output layer. The baseline uses channels 16→32 for MNIST/ORL and 32→64 for CIFAR-10. The deeper models use 16→32→64 and 32→64→128, respectively. The baseline convolution uses a 3×3 kernel, stride 1, and padding 1. MNIST/ORL also test a 5×5 kernel with padding 2; CIFAR-10 also tests average pooling instead of max pooling.

All experiments use Adam, cross-entropy loss, learning rate 0.001, and random seed 2. MNIST uses 5 epochs and batch size 256; ORL uses 20 epochs and batch size 32; CIFAR-10 uses 10 epochs and batch size 256.

## Results

| Dataset | Variant | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| orl | baseline | 95.00% | 92.50% | 95.00% | 93.33% |
| orl | deeper | 97.50% | 96.25% | 97.50% | 96.67% |
| orl | kernel5 | 97.50% | 96.25% | 97.50% | 96.67% |
| mnist | baseline | 98.58% | 98.57% | 98.58% | 98.57% |
| mnist | deeper | 98.93% | 98.93% | 98.92% | 98.92% |
| mnist | kernel5 | 98.86% | 98.85% | 98.86% | 98.85% |
| cifar | baseline | 70.19% | 70.70% | 70.19% | 70.33% |
| cifar | deeper | 70.87% | 71.00% | 70.87% | 70.25% |
| cifar | avg_pool | 63.87% | 64.53% | 63.87% | 63.35% |

## Output Files

Results for each configuration are saved under `result/stage_3_result/<dataset>/<variant>/`, including model weights, training history, predictions, evaluation metrics, and plots. Summary files include `experiments.csv` and `ablation_summary.csv`. To load a saved model, recreate its architecture using `run_config.json` before loading the weights. The report is in `reports/`.
