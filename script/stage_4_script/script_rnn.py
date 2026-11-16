"""Train and evaluate the Stage 4 text models from the course datasets."""

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Setting import CELLS, plot_learning_curves, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["classify", "generate", "all"], default="all")
    parser.add_argument("--model", choices=[*CELLS, "all"], default="all")
    parser.add_argument("--data", type=Path, default=ROOT / "data/stage_4_data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)

    tasks = ["classify", "generate"]
    if args.task != "all":
        tasks = [args.task]
    cells = CELLS
    if args.model != "all":
        cells = [args.model]

    loader = Dataset_Loader(args.data)
    for task in tasks:
        splits, vocab, training_jokes = loader.load(task, smoke=args.smoke)
        output = args.output / task
        for cell in cells:
            print(f"Starting {task} / {cell}", flush=True)
            run_experiment(task, cell, splits, vocab, training_jokes, output / cell, args.smoke)
        if args.model == "all":
            plot_learning_curves(output, task)


if __name__ == "__main__":
    main()
