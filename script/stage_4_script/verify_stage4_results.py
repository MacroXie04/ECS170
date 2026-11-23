#!/usr/bin/env python3
"""Audit the six Stage 4 runs against source, raw data, and saved checkpoints.

Recompute complete CSV metrics independently of Experiment_Runner. Reload each
best checkpoint on CPU, check a small classifier batch, and reproduce all greedy
three-word continuations. When CUDA is available, additionally confirm the small
classifier batch against the original inference backend at tighter tolerance.
No training or result mutation is performed; only the verification JSON is
written. Any mismatch produces a nonzero exit status.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader, Vocabulary, tokenize
from local_code.stage_4_code.Method_RNN import SequenceModel, generate

FROZEN_SOURCES = {
    "local_code/stage_4_code/Dataset_Loader.py",
    "local_code/stage_4_code/Method_RNN.py",
    "local_code/stage_4_code/Experiment_Runner.py",
    "script/stage_4_script/run_stage4.py",
    "script/stage_4_script/prepare_stage4_data.py",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def equal_number(actual, expected, label, *, rtol=1e-9, atol=1e-10):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)),
            f"{label}: non-finite number")
    require(math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol),
            f"{label}: actual {actual!r} != expected {expected!r}")


def verify_metrics(rows, task, expected_records, saved):
    require([r["id"] for r in rows] == [r["id"] for r in expected_records],
            "CSV test IDs/order do not exactly match raw-data-derived test split")
    require(len(set(r["id"] for r in rows)) == len(rows), "Repeated CSV IDs")
    require(len(rows) == saved["n_examples"], "Test example count mismatch")
    if task == "classification":
        confusion = [[0, 0], [0, 0]]
        for row, source in zip(rows, expected_records):
            label, prediction = int(row["label"]), int(row["prediction"])
            require(label == source["label"], f"Wrong ground truth for {row['id']}")
            require(prediction in (0, 1), "Prediction outside binary class set")
            probability, nll = float(row["positive_probability"]), float(row["nll"])
            require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid probability")
            require(math.isfinite(nll) and nll >= 0, "Invalid NLL")
            confusion[label][prediction] += 1
        precisions, recalls, f1s = [], [], []
        for cls in (0, 1):
            tp = confusion[cls][cls]
            fp, fn = confusion[1 - cls][cls], confusion[cls][1 - cls]
            precisions.append(tp / (tp + fp) if tp + fp else 0.0)
            recalls.append(tp / (tp + fn) if tp + fn else 0.0)
            f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        recomputed = {
            "n_examples": len(rows),
            "loss": math.fsum(float(r["nll"]) for r in rows) / len(rows),
            "accuracy": (confusion[0][0] + confusion[1][1]) / len(rows),
            "macro_precision": math.fsum(precisions) / 2,
            "macro_recall": math.fsum(recalls) / 2,
            "macro_f1": math.fsum(f1s) / 2,
            "confusion_matrix": confusion,
        }
    else:
        for row, source in zip(rows, expected_records):
            count, correct = int(row["token_count"]), int(row["correct_tokens"])
            require(count == len(source["tokens"]) - 1,
                    f"Wrong next-token count for {row['id']} (BOS excluded, EOS included)")
            require(0 <= correct <= count, "Invalid correct-token count")
            require(math.isfinite(float(row["nll_sum"])) and float(row["nll_sum"]) >= 0,
                    "Invalid token NLL sum")
        n = sum(int(row["token_count"]) for row in rows)
        loss = math.fsum(float(row["nll_sum"]) for row in rows) / n
        recomputed = {
            "n_examples": len(rows), "n_tokens": n, "loss": loss,
            "perplexity": math.exp(loss),
            "token_accuracy": sum(int(row["correct_tokens"]) for row in rows) / n,
        }
    require(set(recomputed) == set(saved), "Unexpected/missing test metric keys")
    for name, value in recomputed.items():
        if isinstance(value, list):
            require(value == saved[name], f"{name} mismatch")
        else:
            equal_number(saved[name], value, name)
    return recomputed


def classifier_checkpoint_check(model, rows, records, batch_size, cpu_only=False):
    records = records[:batch_size]
    rows = rows[:batch_size]
    lengths = torch.tensor([len(r["tokens"]) for r in records])
    tokens = torch.zeros(len(records), int(lengths.max()), dtype=torch.long)
    for index, record in enumerate(records):
        tokens[index, :lengths[index]] = torch.tensor(record["tokens"])
    with torch.no_grad():
        logits = model(tokens, lengths)
        probabilities = logits.softmax(-1)[:, 1]
        predictions = logits.argmax(-1)
        losses = torch.nn.functional.cross_entropy(
            logits, torch.tensor([r["label"] for r in records]), reduction="none")
    max_probability_error = 0.0
    for index, row in enumerate(rows):
        require(int(predictions[index]) == int(row["prediction"]),
                f"Checkpoint class mismatch for {row['id']}")
        probability = float(probabilities[index])
        max_probability_error = max(max_probability_error,
                                    abs(probability - float(row["positive_probability"])))
        # cuDNN's TF32-enabled recurrent kernels and CPU float32 differ slightly.
        # Original-CUDA replay is checked much more tightly below, when available.
        equal_number(probability, float(row["positive_probability"]),
                     f"CPU checkpoint probability {row['id']}", rtol=1e-3, atol=5e-4)
        equal_number(float(losses[index]), float(row["nll"]),
                     f"CPU checkpoint NLL {row['id']}", rtol=1e-3, atol=1e-3)
    result = {"examples": len(records), "predictions_match": True,
              "max_absolute_probability_error": max_probability_error,
              "probability_absolute_tolerance": 5e-4, "probability_relative_tolerance": 1e-3,
              "note": "CPU versus TF32-enabled cuDNN float32 comparison; CUDA confirmation uses tighter tolerance."}
    if torch.cuda.is_available() and not cpu_only:
        model.to("cuda")
        with torch.no_grad():
            cuda_logits = model(tokens.cuda(), lengths.cuda())
            cuda_probabilities = cuda_logits.softmax(-1)[:, 1].cpu()
            cuda_predictions = cuda_logits.argmax(-1).cpu()
            cuda_losses = torch.nn.functional.cross_entropy(
                cuda_logits, torch.tensor([r["label"] for r in records], device="cuda"),
                reduction="none").cpu()
        maximum = 0.0
        for index, row in enumerate(rows):
            require(int(cuda_predictions[index]) == int(row["prediction"]),
                    f"CUDA checkpoint class mismatch for {row['id']}")
            probability = float(cuda_probabilities[index])
            maximum = max(maximum, abs(probability - float(row["positive_probability"])))
            equal_number(probability, float(row["positive_probability"]),
                         f"CUDA checkpoint probability {row['id']}", rtol=1e-5, atol=2e-6)
            equal_number(float(cuda_losses[index]), float(row["nll"]),
                         f"CUDA checkpoint NLL {row['id']}", rtol=1e-5, atol=2e-6)
        result["cuda_confirmation"] = {"examples": len(records), "predictions_match": True,
            "max_absolute_probability_error": maximum,
            "probability_absolute_tolerance": 2e-6, "probability_relative_tolerance": 1e-5,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
        model.to("cpu")
    else:
        result["cuda_confirmation"] = {"skipped": True, "reason": "CPU-only requested or CUDA unavailable"}
    return result


def generation_checkpoint_check(model, examples, data):
    require(len(examples) == 3, "Expected all three fixed generation examples")
    require([e["prefix"] for e in examples] == ["what do you", "why did the", "what did the"],
            "Unexpected generation prefix selection")
    training = {r["id"]: r for r in data["train"]}
    vocab = data["vocab"]
    checked = []
    for example in examples:
        prefix_tokens = tokenize(example["prefix"])
        require(len(example["prefix"].split()) == len(prefix_tokens) == 3,
                "Generation prefix must contain exactly three words")
        require(example["source_split"] == "train", "Wrong generation reference split")
        require(example["reference_id"] in training, "Generation reference is not in training")
        reference = training[example["reference_id"]]
        require(example["reference"] == reference["text"], "Reference text does not match raw data")
        require(tokenize(reference["text"])[:3] == prefix_tokens,
                "Three-word prefix does not match its reference")
        for mode in ("greedy", "sampled"):
            require(tokenize(example[mode])[:3] == prefix_tokens,
                    f"{mode} output lost its three-word prefix")
        ids = generate(model, vocab.encode(example["prefix"], bos=True),
                       max_new_tokens=example["max_new_tokens"], temperature=0)
        regenerated = vocab.decode(ids)
        require(regenerated == example["greedy"],
                f"Greedy checkpoint output mismatch for {example['prefix']!r}: {regenerated!r}")
        checked.append({"prefix": example["prefix"], "reference_id": example["reference_id"],
                        "greedy_checkpoint_match": True})
    return checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "result/stage_4_result/runs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/stage_4_data")
    parser.add_argument("--output", type=Path, default=ROOT / "result/stage_4_result/report/verification.json")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--classification-batch-size", type=int, default=8)
    parser.add_argument("--cpu-only", action="store_true", help="Skip optional strict CUDA classifier confirmation")
    args = parser.parse_args(argv)
    require(args.threads > 0 and args.classification_batch_size > 0, "Thread/batch counts must be positive")
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": file_hash(__file__),
        "torch": str(torch.__version__), "python": sys.version,
        "platform": platform.platform(), "inference_device": "cpu", "threads": args.threads,
        "cuda_classifier_confirmation": torch.cuda.is_available() and not args.cpu_only,
        "runs": {}, "failures": [],
        "scope": "Full test-CSV metric/ID audit; source/data/checkpoint binding; small classifier replay; all greedy examples",
    }
    cache = {}
    for task in ("classification", "generation"):
        for cell in ("rnn", "lstm", "gru"):
            name = f"{task}_{cell}_{args.tag}"
            run = args.runs / name
            entry = report["runs"][name] = {"passed": False}
            try:
                config = read_json(run / "config.json")
                metrics = read_json(run / "metrics.json")
                history = read_json(run / "history.json")
                environment = read_json(run / "environment.json")
                require(not config["smoke"] and not metrics["smoke"], "Smoke artifacts are not full runs")
                require(config["model"]["task"] == metrics["task"] == task, "Task identity mismatch")
                require(config["model"]["cell"] == metrics["cell"] == cell, "Cell identity mismatch")
                require(config["tag"] == metrics["tag"] == args.tag, "Tag mismatch")
                require(set(config["source_sha256"]) == FROZEN_SOURCES, "Incomplete frozen-source manifest")
                for source, digest in config["source_sha256"].items():
                    require(file_hash(ROOT / source) == digest, f"Frozen source changed: {source}")
                require(environment["source_sha256"] == config["source_sha256"], "Environment/source mismatch")
                canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False)
                digest = hashlib.sha256(canonical.encode()).hexdigest()
                checkpoint = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=True)
                require(checkpoint["config_hash"] == digest, "Best checkpoint/config hash mismatch")
                require(checkpoint["model_kwargs"] == config["model"], "Best checkpoint/model config mismatch")
                require(history and [h["epoch"] for h in history] == list(range(1, len(history) + 1)),
                        "Epoch history is not complete and contiguous")
                require(len(history) == metrics["epochs_completed"] <= config["max_epochs"], "Epoch count mismatch")
                best = min(history, key=lambda h: h["validation_loss"])
                require(best["epoch"] == metrics["best_epoch"] == checkpoint["epoch"],
                        "Best epoch is not the earliest minimum validation loss")
                equal_number(metrics["validation_loss"], best["validation_loss"], "Best validation loss")
                equal_number(metrics["train_seconds"], math.fsum(h["seconds"] for h in history), "Training seconds")
                metadata = config["data_metadata"]
                key = (task, config["seed"], config["max_vocab"], config["classification_max_length"],
                       metadata.get("validation_fraction", 0.1))
                if key not in cache:
                    loader = Dataset_Loader(args.data_dir, seed=key[1], max_vocab=key[2],
                                            max_length=key[3], validation_fraction=key[4])
                    cache[key] = getattr(loader, f"load_{task}")()
                data = cache[key]
                require(data["metadata"] == metadata, "Raw-data-derived metadata differs from run config")
                require(read_json(args.runs / f"{task}_metadata.json") == metadata,
                        "Root task metadata differs from run config")
                split_ids = read_json(args.runs / f"{task}_split_ids.json")
                require(split_ids == {s: [r["id"] for r in data[s]] for s in ("train", "validation", "test")},
                        "Saved split manifest differs from raw-data-derived split")
                require(checkpoint["vocab"] == data["vocab"].to_dict()
                        == read_json(args.runs / f"{task}_vocabulary.json"), "Vocabulary/checkpoint mismatch")
                require(metadata["normalized_text_overlap_counts"]["train_validation"] == 0,
                        "Train/validation normalized text leakage")
                if task == "generation":
                    require(not any(metadata["normalized_text_overlap_counts"].values()), "Joke split text leakage")
                with (run / "test_predictions.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                recomputed = verify_metrics(rows, task, data["test"], metrics["test"])
                model = SequenceModel(**checkpoint["model_kwargs"]).eval()
                model.load_state_dict(checkpoint["model"], strict=True)
                require(model.parameter_count() == metrics["parameters"], "Parameter count mismatch")
                if task == "classification":
                    replay = classifier_checkpoint_check(model, rows, data["test"],
                                                         args.classification_batch_size, args.cpu_only)
                else:
                    replay = generation_checkpoint_check(model, read_json(run / "generation_examples.json"), data)
                entry.update(passed=True, best_epoch=best["epoch"], config_hash=digest,
                    source_sha256=config["source_sha256"], dataset_source_sha256=metadata["source_sha256"],
                    test_id_sha256=metadata["split_id_sha256"]["test"],
                    recomputed_test_metrics=recomputed, checkpoint_replay=replay,
                    normalized_text_overlap_counts=metadata["normalized_text_overlap_counts"],
                    artifact_sha256={p: file_hash(run / p) for p in
                        ("config.json", "metrics.json", "history.json", "test_predictions.csv", "checkpoints/best.pt")})
                print(f"PASS {name}: {len(rows)} test rows; best epoch {best['epoch']}", flush=True)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                entry["error"] = message
                report["failures"].append({"run": name, "error": message})
                print(f"FAIL {name}: {message}", file=sys.stderr, flush=True)
    report["passed"] = not report["failures"] and len(report["runs"]) == 6
    report["verified_run_count"] = sum(r["passed"] for r in report["runs"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(f"Verification {'PASSED' if report['passed'] else 'FAILED'}: {args.output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
