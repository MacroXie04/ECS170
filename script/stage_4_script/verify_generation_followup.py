#!/usr/bin/env python3
"""Independently audit the six generation-only follow-up training runs.

Bind raw data, train-only vocabularies, unchanged baseline source, configuration,
best/last checkpoints, histories, and complete held-out CSV metrics. This audit
does not select qualitative profiles, retrain models, or run GPU inference.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import SequenceModel
from script.stage_4_script.verify_stage4_results import (
    FROZEN_SOURCES, equal_number, file_hash, read_json, require, verify_metrics,
)

GRIDS = {
    "compact256": {"embedding_dim": 256, "hidden_dim": 256, "num_layers": 1,
                   "dropout": 0.1, "batch_size": 32, "max_epochs": 180, "patience": 20},
    "wide256": {"embedding_dim": 128, "hidden_dim": 256, "num_layers": 2,
                "dropout": 0.15, "batch_size": 64, "max_epochs": 160, "patience": 20},
}


def resolve_commit(reference):
    return subprocess.check_output(
        ["git", "rev-parse", f"{reference}^{{commit}}"], cwd=ROOT, text=True).strip()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path,
                        default=ROOT / "result/stage_4_result/generation_optimization/runs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/stage_4_data")
    parser.add_argument("--baseline", type=Path, default=ROOT / "result/stage_4_result/runs")
    parser.add_argument("--baseline-commit", default="f82bdcd")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "result/stage_4_result/generation_optimization/training_verification.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    require(args.threads > 0, "Thread count must be positive")
    torch.set_num_threads(args.threads)
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": file_hash(__file__),
        "metric_helper_sha256": file_hash(ROOT / "script/stage_4_script/verify_stage4_results.py"),
        "python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(),
        "checkpoint_device": "cpu", "gpu_inference_or_training": False,
        "scope": "Six training experiments only; full raw-data/CSV audit and strict CPU checkpoint loading; no qualitative profile selection",
        "expected_grids": GRIDS, "runs": {}, "failures": [],
    }
    try:
        commit = resolve_commit(args.baseline_commit)
        baseline_config = read_json(args.baseline / "generation_gru_baseline/config.json")
        frozen = baseline_config["source_sha256"]
        require(set(frozen) == FROZEN_SOURCES, "Incomplete original baseline source manifest")
        for path, expected in frozen.items():
            require(file_hash(ROOT / path) == expected, f"Frozen training source changed: {path}")
            original = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
            require(hashlib.sha256(original).hexdigest() == expected,
                    f"Frozen source does not match initial Stage 4 commit: {path}")
        report["baseline_commit"] = commit
        report["frozen_source_sha256"] = frozen
        baseline_metadata = baseline_config["data_metadata"]
        loader = Dataset_Loader(args.data_dir, seed=baseline_config["seed"],
                                max_vocab=baseline_config["max_vocab"],
                                max_length=baseline_config["classification_max_length"])
        data = loader.load_generation()
        require(data["metadata"] == baseline_metadata, "Raw dataset/splits differ from the initial baseline")
        for directory in (args.runs, args.baseline):
            require(read_json(directory / "generation_metadata.json") == data["metadata"],
                    f"Task metadata changed in {directory}")
            require(read_json(directory / "generation_split_ids.json") ==
                    {split: [row["id"] for row in data[split]] for split in ("train", "validation", "test")},
                    f"Split ID manifest mismatch in {directory}")
            require(read_json(directory / "generation_vocabulary.json") == data["vocab"].to_dict(),
                    f"Train-derived vocabulary mismatch in {directory}")
        require(not any(data["metadata"]["id_overlap_counts"].values()), "Dataset ID overlap")
        require(not any(data["metadata"]["normalized_text_overlap_counts"].values()),
                "Normalized text leaked across generation splits")
        require(data["metadata"]["vocabulary_fit_split"] == "train", "Vocabulary not fitted only to train")
        require(data["metadata"]["dropped_rows"] == 0 and
                sum(len(data[split]) for split in ("train", "validation", "test")) ==
                data["metadata"]["source_count"], "Dataset rows were dropped or duplicated")
        report["dataset"] = {key: data["metadata"][key] for key in
            ("source_sha256", "counts", "source_count", "normalized_group_count",
             "duplicate_normalized_rows", "dropped_rows", "split_id_sha256", "vocabulary_sha256",
             "vocabulary_fit_split", "id_overlap_counts", "normalized_text_overlap_counts")}
    except Exception as error:
        report["failures"].append({"scope": "shared prerequisites", "error": f"{type(error).__name__}: {error}"})
        data = None

    if data is not None:
        for tag, grid in GRIDS.items():
            for cell in ("rnn", "lstm", "gru"):
                name = f"generation_{cell}_{tag}"
                directory = args.runs / name
                entry = report["runs"][name] = {"passed": False}
                try:
                    config = read_json(directory / "config.json")
                    metrics = read_json(directory / "metrics.json")
                    history = read_json(directory / "history.json")
                    environment = read_json(directory / "environment.json")
                    require(config["model"]["task"] == metrics["task"] == "generation", "Wrong task")
                    require(config["model"]["cell"] == metrics["cell"] == cell, "Wrong recurrent cell")
                    require(config["tag"] == metrics["tag"] == tag, "Wrong experiment tag")
                    require(not config["smoke"] and not metrics["smoke"], "Smoke run supplied")
                    require(config["source_sha256"] == environment["source_sha256"] == frozen,
                            "Source provenance differs from frozen baseline")
                    require(config["data_metadata"] == data["metadata"], "Experiment data differs from baseline")
                    for key in ("seed", "max_vocab", "classification_max_length", "optimizer",
                                "learning_rate", "weight_decay", "gradient_clip_norm", "checkpoint_selection"):
                        require(config[key] == baseline_config[key], f"Unexpected common training setting: {key}")
                    for key, expected in grid.items():
                        actual = config["model"][key] if key in config["model"] else config[key]
                        require(actual == expected, f"Unexpected grid setting {key}: {actual!r}")
                    require(config["model"]["vocab_size"] == len(data["vocab"]), "Model/vocabulary size mismatch")
                    digest = canonical_hash(config)
                    best = torch.load(directory / "checkpoints/best.pt", map_location="cpu", weights_only=True)
                    last = torch.load(directory / "checkpoints/last.pt", map_location="cpu", weights_only=True)
                    require(best["config_hash"] == last["config_hash"] == digest,
                            "Best or last checkpoint/config hash mismatch")
                    require(best["model_kwargs"] == config["model"], "Best checkpoint model configuration mismatch")
                    require(best["vocab"] == data["vocab"].to_dict(), "Best checkpoint vocabulary mismatch")
                    require(history and [row["epoch"] for row in history] == list(range(1, len(history) + 1)),
                            "Epoch history is incomplete or not ordered")
                    require(history == last["history"], "Last checkpoint/history JSON mismatch")
                    require(len(history) == metrics["epochs_completed"] <= config["max_epochs"], "Epoch count mismatch")
                    minimum = min(history, key=lambda row: row["validation_loss"])
                    require(minimum["epoch"] == metrics["best_epoch"] == best["epoch"] == last["best_epoch"],
                            "Selected checkpoint is not earliest minimum validation loss")
                    equal_number(metrics["validation_loss"], minimum["validation_loss"], "Best validation loss")
                    equal_number(last["best_loss"], minimum["validation_loss"], "Last checkpoint best loss")
                    seconds = math.fsum(row["seconds"] for row in history)
                    equal_number(metrics["train_seconds"], seconds, "Training time")
                    equal_number(last["seconds"], seconds, "Last checkpoint training time")
                    require(len(history) == config["max_epochs"] or
                            len(history) - minimum["epoch"] >= config["patience"],
                            "Training stopped before its epoch budget or early-stopping patience")
                    with (directory / "test_predictions.csv").open(newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    recomputed = verify_metrics(rows, "generation", data["test"], metrics["test"])
                    with torch.random.fork_rng(devices=[]):
                        model = SequenceModel(**config["model"])
                    model.load_state_dict(best["model"], strict=True)
                    require(model.parameter_count() == metrics["parameters"], "Parameter count mismatch")
                    model.load_state_dict(last["model"], strict=True)
                    artifact_paths = ("config.json", "environment.json", "history.json", "metrics.json",
                                      "test_predictions.csv", "checkpoints/best.pt", "checkpoints/last.pt")
                    entry.update(passed=True, config_hash=digest, best_epoch=minimum["epoch"],
                        last_epoch=len(history), parameter_count=metrics["parameters"],
                        strict_checkpoint_loads=["best.pt", "last.pt"],
                        recomputed_test_metrics=recomputed,
                        artifact_sha256={path: file_hash(directory / path) for path in artifact_paths})
                    print(f"PASS {name}: {len(rows)} held-out rows; best epoch {minimum['epoch']}; last epoch {len(history)}", flush=True)
                    del model, best, last
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    entry["error"] = message
                    report["failures"].append({"run": name, "error": message})
                    print(f"FAIL {name}: {message}", file=sys.stderr, flush=True)
    report["verified_run_count"] = sum(entry["passed"] for entry in report["runs"].values())
    report["verified_test_rows"] = sum(entry.get("recomputed_test_metrics", {}).get("n_examples", 0)
                                       for entry in report["runs"].values())
    report["passed"] = not report["failures"] and report["verified_run_count"] == 6
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(f"Follow-up training verification {'PASSED' if report['passed'] else 'FAILED'}: {args.output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
