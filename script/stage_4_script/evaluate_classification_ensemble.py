"""Evaluate or infer with the locked, equal-weight three-seed IMDb ensemble.

Validation uses saved validation predictions. The official test split is read
only by the explicit final-eval subcommand after a selection manifest is locked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from local_code.stage_4_code.Dataset_Loader import Vocabulary, truncate_tokens
from script.stage_4_script.train_classification import (
    ClassifierModel, choose_device, collate, data_loader, digest_file, write_json
)

RUN_ROOT = REPO / "result/stage_4_result/classification_optimization_20261117"
DEFAULT_SELECTION = (
    REPO / "result/stage_4_result/classification_ensemble_followup_20261117/selection.json"
)
EXPECTED_TAGS = {
    "c_bigru128_meanmax_512_headtail_s1704": 1704,
    "e_bigru128_meanmax_drop30_512_headtail_s1705": 1705,
    "e_bigru128_meanmax_drop30_512_headtail_s1706": 1706,
}
SOURCE_NAMES = (
    "local_code/stage_4_code/Dataset_Loader.py",
    "local_code/stage_4_code/Method_Classifier.py",
    "local_code/stage_4_code/Experiment_Runner.py",
)


def metrics_from_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No prediction rows")
    truth = [int(row["label"]) for row in rows]
    prediction = [int(row["prediction"]) for row in rows]
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, prediction, average="macro", zero_division=0
    )
    return {
        "n_examples": len(rows),
        "loss": float(np.mean([float(row["nll"]) for row in rows])),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1]).tolist(),
    }


def make_row(identifier: str, label: int, probability: float) -> dict:
    probability = float(probability)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid member probability")
    safe = min(max(probability, 1e-15), 1 - 1e-15)
    return {
        "id": identifier,
        "label": int(label),
        "prediction": int(probability >= 0.5),
        "positive_probability": probability,
        "nll": -math.log(safe if label else 1 - safe),
    }


def load_locked(selection_path: Path) -> tuple[dict, list[tuple[dict, dict, Path]]]:
    selection_path = selection_path.resolve()
    selection = json.loads(selection_path.read_text())
    if selection.get("locked_before_test") is not True:
        raise ValueError("Selection must be locked before official-test evaluation")
    if not isinstance(selection.get("validation_metrics"), dict):
        raise ValueError("Locked validation metrics are required")
    if selection.get("threshold") != 0.5:
        raise ValueError("The preselected decision threshold must be 0.5")
    members = selection.get("members")
    if not isinstance(members, list) or len(members) != 3:
        raise ValueError("The ensemble requires exactly three locked members")
    weights = selection.get("weights")
    if (not isinstance(weights, list) or len(weights) != 3 or
        any(not isinstance(weight, (int, float)) or
            not math.isclose(weight, 1 / 3, rel_tol=0, abs_tol=1e-12)
            for weight in weights)):
        raise ValueError("Ensemble weights must be equal and fixed at one third")
    tags = [member.get("run") for member in members]
    if set(tags) != set(EXPECTED_TAGS) or len(set(tags)) != 3:
        raise ValueError("Ensemble members do not match the three preselected runs")
    loaded = []
    common_config = None
    common_splits = None
    common_vocab = None
    for member in members:
        tag = member["run"]
        run = RUN_ROOT / tag
        if run.resolve().parent != RUN_ROOT.resolve():
            raise ValueError("Member path leaves the saved experiment directory")
        config = json.loads((run / "config.json").read_text())
        summary = json.loads((run / "summary.json").read_text())
        if (member.get("config_hash") != summary["config_hash"] or
            member.get("checkpoint_sha256") != summary["best_checkpoint_sha256"] or
            digest_file(run / "checkpoints/best.pt") != member["checkpoint_sha256"]):
            raise ValueError(f"Checkpoint or configuration binding mismatch: {tag}")
        if config["model_seed"] != EXPECTED_TAGS[tag] or config["split_seed"] != 1704:
            raise ValueError(f"Unexpected model or split seed: {tag}")
        model_spec = config["model"]
        if (model_spec["cell"] != "gru" or
            model_spec["bidirectional"] is not True or
            model_spec["pooling"] != "meanmax" or
            model_spec["dropout"] != 0.3 or
            model_spec["embedding_dim"] != 128 or
            model_spec["hidden_dim"] != 128 or
            model_spec["num_layers"] != 1 or
            config["max_length"] != 512 or
            config["truncation_mode"] != "head_tail"):
            raise ValueError(f"Wrong architecture or input view: {tag}")
        for name in SOURCE_NAMES:
            if digest_file(REPO / name) != config["source_sha256"][name]:
                raise ValueError(f"Source changed since training: {name}")
        comparable = {
            key: value for key, value in config.items()
            if key not in {"tag", "model_seed", "source_sha256"}
        }
        if common_config is None:
            common_config = comparable
            common_splits = summary["split_id_sha256"]
            common_vocab = summary["vocabulary_sha256"]
        elif (comparable != common_config or
              summary["split_id_sha256"] != common_splits or
              summary["vocabulary_sha256"] != common_vocab):
            raise ValueError("Members have different data, vocabularies, or training settings")
        loaded.append((config, summary, run))
    return selection, loaded


def validation_rows(loaded: list[tuple[dict, dict, Path]]) -> list[dict]:
    tables = []
    for _, summary, run in loaded:
        with (run / "validation_predictions.csv").open(newline="") as handle:
            table = list(csv.DictReader(handle))
        if len(table) != 2500 or len({row["id"] for row in table}) != 2500:
            raise ValueError(f"Member validation rows are incomplete or duplicated: {run.name}")
        metrics = metrics_from_rows(table)
        for key in ("accuracy", "loss"):
            if abs(metrics[key] - summary["validation"][key]) > 1e-8:
                raise ValueError(f"Member validation CSV disagrees with summary: {run.name} {key}")
        tables.append(table)
    lengths = {len(table) for table in tables}
    if lengths != {2500}:
        raise ValueError("Validation predictions must have exactly 2500 rows per member")
    rows = []
    for entries in zip(*tables):
        ids = {entry["id"] for entry in entries}
        labels = {int(entry["label"]) for entry in entries}
        if len(ids) != 1 or len(labels) != 1:
            raise ValueError("Ensemble validation rows are misaligned")
        probability = sum(float(entry["positive_probability"]) for entry in entries) / 3
        rows.append(make_row(entries[0]["id"], int(entries[0]["label"]), probability))
    if len({row["id"] for row in rows}) != 2500:
        raise ValueError("Duplicate validation IDs")
    if sum(row["label"] == 0 for row in rows) != 1250 or sum(row["label"] == 1 for row in rows) != 1250:
        raise ValueError("Validation labels differ from the balanced fixed split")
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["id"].encode("utf-8"))
        digest.update(b"\0")
    if digest.hexdigest() != loaded[0][1]["split_id_sha256"]["validation"]:
        raise ValueError("Validation prediction IDs differ from the fixed split")
    return rows


def verify_validation(selection: dict, loaded: list[tuple[dict, dict, Path]]) -> dict:
    metrics = metrics_from_rows(validation_rows(loaded))
    expected = selection["validation_metrics"]
    for key in ("accuracy", "macro_f1", "loss"):
        if key not in expected or abs(float(expected[key]) - metrics[key]) > 1e-8:
            raise ValueError(f"Locked validation metric differs from member predictions: {key}")
    return metrics


def load_models(loaded: list[tuple[dict, dict, Path]], device: torch.device) -> tuple[list[ClassifierModel], dict]:
    models = []
    vocabulary = None
    for config, summary, run in loaded:
        checkpoint = torch.load(run / "checkpoints/best.pt", map_location=device, weights_only=True)
        if checkpoint["config_hash"] != summary["config_hash"]:
            raise ValueError("Checkpoint config hash mismatch")
        if checkpoint["model_kwargs"] != config["model"]:
            raise ValueError("Checkpoint model specification mismatch")
        if vocabulary is None:
            vocabulary = checkpoint["vocab"]
        elif checkpoint["vocab"] != vocabulary:
            raise ValueError("Member checkpoint vocabularies differ")
        model = ClassifierModel(**config["model"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
    return models, vocabulary


def evaluate_models(models: list[ClassifierModel], loader: DataLoader, device: torch.device) -> tuple[dict, list[dict]]:
    rows = []
    with torch.no_grad():
        for tokens, lengths, labels, identifiers in loader:
            tokens, lengths = tokens.to(device), lengths.to(device)
            probabilities = [
                model(tokens, lengths).softmax(-1)[:, 1].cpu()
                for model in models
            ]
            averaged = torch.stack(probabilities).double().mean(dim=0).tolist()
            rows.extend(
                make_row(identifier, int(label), probability)
                for identifier, label, probability in zip(
                    identifiers, labels.tolist(), averaged
                )
            )
    return metrics_from_rows(rows), rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate(args: argparse.Namespace) -> None:
    selection, loaded = load_locked(Path(args.selection))
    metrics = verify_validation(selection, loaded)
    print("LOCKED_VALIDATION " + json.dumps(metrics), flush=True)


def final_eval(args: argparse.Namespace) -> None:
    selection_path = Path(args.selection).resolve()
    selection, loaded = load_locked(selection_path)
    validation = verify_validation(selection, loaded)
    output = selection_path.parent / "final_test"
    manifest_hash = digest_file(selection_path)
    if output.exists():
        path = output / "metrics.json"
        if not path.exists():
            raise RuntimeError("Partial ensemble test output exists; inspect before retrying")
        existing = json.loads(path.read_text())
        if existing.get("selection_sha256") != manifest_hash or not (output / "predictions.csv").exists():
            raise ValueError("Existing test output is incomplete or belongs to a different selection")
        print("SKIP existing locked ensemble final test " + str(output), flush=True)
        return
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    config, summary, _ = loaded[0]
    if config["archive_path"] is not None and digest_file(Path(config["archive_path"])) != config["archive_sha256"]:
        raise ValueError("Course archive changed since training")
    data = data_loader(config["data_dir"], config, include_test=True)
    for split in ("train", "validation"):
        if data["metadata"]["split_id_sha256"][split] != summary["split_id_sha256"][split]:
            raise ValueError(f"Locked {split} IDs differ from official data loader")
    if data["metadata"]["vocabulary_sha256"] != summary["vocabulary_sha256"]:
        raise ValueError("Vocabulary differs from locked runs")
    baseline = json.loads((REPO / "result/stage_4_result/runs/classification_metadata.json").read_text())
    if len(data["test"]) != 25000 or data["metadata"]["label_counts"]["test"] != {"0": 12500, "1": 12500}:
        raise ValueError("Official test is incomplete or unbalanced")
    if (data["metadata"]["split_id_sha256"]["test"] != baseline["split_id_sha256"]["test"] or
        data["metadata"]["source_sha256"] != baseline["source_sha256"]):
        raise ValueError("Official test IDs or source contents differ from baseline")
    models, vocabulary = load_models(loaded, device)
    if vocabulary != data["vocab"].to_dict():
        raise ValueError("Checkpoint and data vocabularies differ")
    test_loader = DataLoader(
        data["test"], batch_size=config["batch_size"], shuffle=False, collate_fn=collate,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    started = time.monotonic()
    metrics, rows = evaluate_models(models, test_loader, device)
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "predictions.csv", rows)
    write_json(output / "metrics.json", {
        "selection_sha256": manifest_hash,
        "members": selection["members"],
        "weights": selection["weights"],
        "threshold": selection["threshold"],
        "validation": validation,
        "official_test": metrics,
        "evaluation_seconds": time.monotonic() - started,
        "test_split_id_sha256": data["metadata"]["split_id_sha256"]["test"],
        "previous_official_test_exposure": "Original GRU 87.184 percent and selected single BiGRU seed 89.244 percent were previously viewed; this is a repeated official-test use.",
    })
    print("FINAL_TEST_ENSEMBLE " + json.dumps(metrics), flush=True)


def predict(args: argparse.Namespace) -> None:
    selection, loaded = load_locked(Path(args.selection))
    device = choose_device(args.device)
    models, vocabulary = load_models(loaded, device)
    config = loaded[0][0]
    vocab = Vocabulary.from_dict(vocabulary)
    ids = truncate_tokens(
        vocab.encode(args.text), config["max_length"], config["truncation_mode"]
    )
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        probabilities = [
            float(model(tokens, lengths).softmax(-1)[0, 1])
            for model in models
        ]
    average = sum(probabilities) / 3
    print(json.dumps({
        "label": "positive" if average >= 0.5 else "negative",
        "positive_probability": average,
        "member_positive_probabilities": probabilities,
        "members": [member["run"] for member in selection["members"]],
    }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_val = sub.add_parser("validate")
    p_val.add_argument("--selection", default=str(DEFAULT_SELECTION))
    p_test = sub.add_parser("final-eval")
    p_test.add_argument("--selection", default=str(DEFAULT_SELECTION))
    p_test.add_argument("--device", default="auto")
    p_test.add_argument("--threads", type=int, default=8)
    p_predict = sub.add_parser("predict")
    p_predict.add_argument("--selection", default=str(DEFAULT_SELECTION))
    p_predict.add_argument("--device", default="cpu")
    p_predict.add_argument("--text", required=True)
    args = parser.parse_args()
    if args.command == "validate":
        validate(args)
    elif args.command == "final-eval":
        final_eval(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
