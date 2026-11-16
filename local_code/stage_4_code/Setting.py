"""Run one Stage 4 experiment and save its results."""

import csv
import json
import math
import re

import numpy as np
import torch
from torch.utils.data import DataLoader

from .Dataset_Loader import classify_batch, generate_batch, tokenize
from .Evaluate import Evaluate
from .Method_RNN import Method_RNN


CELLS = ("RNN", "LSTM", "GRU")
PROMPTS = ("what did the", "why did the", "how do you")
SEED = 170


def run_experiment(task, cell, splits, vocab, training_jokes, output, smoke):
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(SEED)

    if task == "classify":
        batch_size, hidden_size = 128, 64
        epochs, patience = 10, 2
        collate = classify_batch
    else:
        batch_size, hidden_size = 32, 96
        epochs, patience = 70, 8
        collate = generate_batch
    if smoke:
        epochs = 1

    model_settings = {
        "task": task,
        "cell": cell,
        "vocab_size": len(vocab),
        "embedding_size": 64,
        "hidden_size": hidden_size,
        "dropout": 0.2,
    }
    model = Method_RNN(**model_settings)
    batches = {}
    for name, rows in splits.items():
        batches[name] = DataLoader(
            rows, batch_size=batch_size, shuffle=(name == "train"), collate_fn=collate,
        )

    learning_rate = 0.002
    history, best_epoch = model.fit(
        batches["train"], batches["val"], epochs, patience, learning_rate,
    )
    evaluation = model.measure(batches["test"], keep_predictions=(task == "classify"))
    test = {
        "loss": evaluation["loss"],
        "accuracy": evaluation["accuracy"],
        "count": evaluation["count"],
    }
    if task == "classify":
        true_labels = evaluation["true_labels"]
        predicted_labels = evaluation["predicted_labels"]
        metrics = Evaluate([0, 1]).score(true_labels, predicted_labels)
        test["macro"] = metrics["macro"]
        test["confusion_matrix"] = metrics["confusion_matrix"]
        np.savez(output / "predictions.npz", true_y=true_labels, pred_y=predicted_labels)
    else:
        test["perplexity"] = math.exp(evaluation["loss"])
        samples = []
        for prompt in PROMPTS:
            samples.append(compare_generation(model, prompt, vocab, training_jokes))
        (output / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")

    split_sizes = {}
    for name, rows in splits.items():
        split_sizes[name] = len(rows)

    record = {
        "seed": SEED,
        "split_seed": 999,
        "smoke": smoke,
        "settings": model_settings,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "patience": patience,
        "threads": torch.get_num_threads(),
        "torch_version": str(torch.__version__),
        "split_sizes": split_sizes,
        "best_epoch": best_epoch,
        "validation": history[best_epoch - 1],
        "test": test,
    }

    with (output / "history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    torch.save(
        {"state_dict": model.state_dict(), "settings": model_settings, "vocab": vocab},
        output / "model.pt",
    )
    (output / "record.json").write_text(json.dumps(record, indent=2) + "\n")


def compare_generation(model, prompt, vocab, training_jokes):
    generated, ended = model.generate(prompt, vocab)
    prefix = tuple(tokenize(prompt))
    matches = 0
    best_match = ()
    shared_tokens = 0
    for joke in training_jokes:
        if joke[:3] != prefix:
            continue
        matches += 1
        overlap = 0
        for generated_token, training_token in zip(generated, joke):
            if generated_token != training_token:
                break
            overlap += 1
        if overlap > shared_tokens:
            shared_tokens = overlap
            best_match = joke
        elif overlap == shared_tokens and tuple(generated) == joke:
            best_match = joke

    return {
        "prompt": prompt,
        "generated": format_tokens(generated),
        "ended_with_eos": ended,
        "training_jokes_with_prefix": matches,
        "closest_training_joke": format_tokens(best_match),
        "shared_prefix_tokens": shared_tokens,
        "exact_training_match": tuple(generated) == best_match,
    }


def format_tokens(tokens):
    return re.sub(r" ([.!?,;:])", r"\1", " ".join(tokens))


def plot_learning_curves(output, task):
    # Import plotting only when combining the completed runs.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    colors = ("#b24b41", "#225e93", "#258064")
    for cell, color in zip(CELLS, colors):
        with (output / cell / "history.csv").open(newline="") as file:
            history = list(csv.DictReader(file))
        epochs = []
        for row in history:
            epochs.append(int(row["epoch"]))
        for axis, metric in zip(axes, ("loss", "accuracy")):
            train_values = []
            val_values = []
            for row in history:
                train_values.append(float(row[f"train_{metric}"]))
                val_values.append(float(row[f"val_{metric}"]))
            axis.plot(epochs, train_values, linestyle="--", color=color, alpha=0.65)
            axis.plot(epochs, val_values, marker=".", color=color, label=cell)

    axes[0].set_ylabel("Cross-entropy")
    axes[1].set_ylabel("Review accuracy" if task == "classify" else "Token accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    axes[1].legend(title="Validation (solid)\nTraining (dashed)")
    figure.savefig(output / "learning_curves.png", dpi=180)
    plt.close(figure)
