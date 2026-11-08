"""Exercise each preserved implementation on synthetic data in its own process.

This checks integration and numerical plumbing, not MNIST accuracy. Outputs go
only into a fresh temporary directory; archived results are never loaded.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

MEMBERS = ("hongzhe", "aileen", "zizheng", "xiaoyu")
ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run_member(member):
    # No member module is imported until its isolated child has selected a root.
    sys.dont_write_bytecode = True
    runtime = ROOT / "implementations" / member
    if member == "xiaoyu":
        runtime /= "ECS170_Fall_2026_Source_Code_Template"
    sys.path.insert(0, str(runtime))
    import numpy as np
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    imported = []

    def module(name):
        result = importlib.import_module("local_code.stage_2_code." + name)
        require(Path(result.__file__).resolve().is_relative_to(runtime.resolve()),
                "Imported code from the wrong member: " + str(result.__file__))
        imported.append(result.__file__)
        return result

    model_class = module("Method_MLP").Method_MLP
    rng = np.random.default_rng(42)
    raw = {}
    for split, count in (("train", 200), ("test", 40)):
        labels = np.tile(np.arange(10, dtype=np.int64), count // 10)
        pixels = rng.integers(0, 256, (count, 784), dtype=np.uint8)
        raw[split] = {"X": pixels, "y": labels}
    normalized = {key: {"X": value["X"].astype(np.float32) / 255,
                        "y": value["y"]} for key, value in raw.items()}

    def check_history(history, field="train_loss"):
        losses = [row[field] for row in history]
        require(1 <= len(losses) <= 2, "Expected one or two synthetic epochs")
        require(np.isfinite(losses).all(), "Training produced non-finite loss")

    def check_predictions(true, pred, scores):
        true, pred = np.asarray(true), np.asarray(pred)
        require(pred.shape == true.shape and pred.ndim == 1, "Prediction shape mismatch")
        require(np.issubdtype(pred.dtype, np.integer), "Predictions must be class integers")
        require(np.isin(pred, np.arange(10)).all(), "Invalid predicted class")
        cm = np.bincount(10 * true + pred, minlength=100).reshape(10, 10)
        tp, support, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
        precision = np.divide(tp, predicted, out=np.zeros(10), where=predicted != 0)
        recall = np.divide(tp, support, out=np.zeros(10), where=support != 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros(10), where=(precision + recall) != 0)
        require(np.isclose(scores["accuracy"], tp.sum() / len(true)), "Accuracy mismatch")
        for average in ("macro", "weighted", "micro"):
            for key, values in (("precision", precision), ("recall", recall), ("f1", f1)):
                if average == "micro":
                    expected = tp.sum() / len(true)
                else:
                    expected = (values.mean() if average == "macro"
                                else np.average(values, weights=support))
                actual = (scores[average][key] if average in scores
                          else scores.get(f"{key}_{average}"))
                if actual is not None:
                    require(np.isfinite(actual) and np.isclose(actual, expected),
                            f"{average} {key} mismatch")
        require("macro" in scores or "f1_macro" in scores, "Missing multiclass metrics")
        if "confusion_matrix" in scores:
            require(np.array_equal(cm, scores["confusion_matrix"]), "Confusion matrix mismatch")

    cases = []
    with tempfile.TemporaryDirectory(prefix="ecs170-synthetic-") as temporary:
        temp = Path(temporary)
        for split, values in raw.items():
            np.savetxt(temp / (split + ".csv"), np.column_stack((values["y"], values["X"])),
                       fmt="%d", delimiter=",")

        if member == "hongzhe":
            setting_class = module("Setting_Prepartitioned").Setting_Prepartitioned
            # This loader deliberately requires 60,000/10,000 rows. Inject the
            # small normalized arrays at the setting's documented input API.
            for optimizer, loss in (("Adam", "CE"), ("SGD", "CE"), ("Adam", "MSE")):
                config = dict(name="synthetic", hidden=[32, 16], optimizer=optimizer,
                              loss=loss, lr=.01, seed=42, epochs=2, batch_size=40)
                model = model_class(config, device="cpu")
                result = setting_class(normalized, model).load_run_save_evaluate()
                check_history(result["history"])
                check_predictions(result["true_y"], result["pred_y"], result["metrics"])
                cases.append(optimizer + "/" + loss)

        elif member == "aileen":
            loader_class = module("Dataset_Loader").Dataset_Loader
            setting_class = module("Setting_Pre_Split").Setting_Pre_Split
            saver_class = module("Result_Saver").Result_Saver
            evaluator_class = module("Evaluate_Metrics").Evaluate_Metrics
            loaders = []
            for split in ("train", "test"):
                loader = loader_class("synthetic " + split, "")
                loader.dataset_source_folder_path = str(temp) + os.sep
                loader.dataset_source_file_name = split + ".csv"
                loaders.append(loader)
            for loss, validation in (("ce_ls", False), ("mse", True)):
                model = model_class("synthetic", "")
                model.hidden_sizes, model.batch_norm = [512, 256], True
                model.optimizer_name, model.learning_rate = "sgd", .05
                model.loss_name, model.scheduler = loss, "cosine"
                model.max_epoch, model.batch_size, model.verbose = 2, 40, False
                saver = saver_class("synthetic", "")
                saver.result_destination_folder_path = str(temp) + os.sep
                saver.result_destination_file_name = loss + "_prediction"
                evaluator = evaluator_class("synthetic", "")
                setting = setting_class("synthetic", "")
                setting.val_fraction, setting.tune_mode = (.2 if validation else 0), validation
                setting.prepare(loaders[0], model, saver, evaluator, loaders[1])
                scores = setting.load_run_save_evaluate()
                require(model.num_parameters() == 537354, "Tuned architecture changed")
                check_history([{"train_loss": value} for value in model.loss_history])
                check_predictions(saver.data["true_y"], saver.data["pred_y"], scores)
                require(len(saver.data["pred_y"]) == 40, "Incorrect evaluation split")
                require(len(model.val_history) == (2 if validation else 0),
                        "Validation history does not match setting")
                require((temp / (loss + "_prediction_None")).is_file(), "Missing temporary result")
                cases.append("SGD/" + loss + ("/validation" if validation else "/test"))

        elif member == "zizheng":
            loader_class = module("Dataset_Loader").Dataset_Loader
            setting_class = module("Setting").Setting
            setting = setting_class(loader_class(temp), temp / "outputs", device="cpu")
            names = []
            for optimizer, loss in (("adam", "cross_entropy"), ("sgd", "squared_error")):
                name = optimizer + "_" + loss
                config = dict(hidden=[32, 16], dropout=0., optimizer=optimizer, loss=loss,
                              lr=.01, weight_decay=0., batch_size=40, epochs=2, patience=3, seed=42)
                record = setting.load_run_save_evaluate(name, config)
                folder = temp / "outputs" / name
                history = np.genfromtxt(folder / "history.csv", delimiter=",", names=True)
                check_history([{"train_loss": value} for value in np.atleast_1d(history["train_loss"])])
                with np.load(folder / "predictions.npz", allow_pickle=False) as predictions:
                    check_predictions(predictions["validation_true"], predictions["validation_pred"],
                                      record["validation"])
                require(record["training_rows"] == 180 and record["validation_rows"] == 20,
                        "Incorrect validation partition")
                names.append(name)
                cases.append(optimizer + "/" + loss)
            setting.select(names)
            record = setting.final(temp / "outputs" / "selection.json")
            with np.load(temp / "outputs/final/predictions.npz", allow_pickle=False) as predictions:
                check_predictions(predictions["test_true"], predictions["test_pred"], record["test"])
            require(record["training_rows"] == 200 and record["test_rows"] == 40,
                    "Final evaluation did not preserve separate test data")
            cases.append("validation-selection-final")

        else:
            loader_class = module("Dataset_Loader").Dataset_Loader
            setting_class = module("Setting_Train_Test_Split").Setting_Train_Test_Split
            saver_class = module("Result_Saver").Result_Saver
            evaluator_class = module("Evaluate_Accuracy").Evaluate_Accuracy
            loader = loader_class("synthetic", "")
            loader.dataset_source_folder_path = str(temp)
            for optimizer, loss in (("adam", "cross_entropy"), ("sgd", "mse"), ("adam", "nll")):
                model = model_class("synthetic", "", hidden_dims=(32, 16), max_epoch=2,
                                    learning_rate=.01, batch_size=40, optimizer_name=optimizer,
                                    loss_name=loss, device="cpu")
                saver = saver_class("synthetic", "")
                saver.result_destination_folder_path = str(temp / (optimizer + "_" + loss))
                evaluator = evaluator_class("synthetic", "")
                setting = setting_class("synthetic", "")
                setting.prepare(loader, model, saver, evaluator)
                scores, _ = setting.load_run_save_evaluate()
                check_history(model.history)
                check_predictions(saver.data["true_y"], saver.data["pred_y"], scores)
                require((Path(saver.result_destination_folder_path) / "predictions.pkl").is_file(),
                        "Missing temporary predictions")
                cases.append(optimizer + "/" + loss)

    # Also check transitive imports, where a crossed implementation can hide.
    for name, loaded in tuple(sys.modules.items()):
        if name == "local_code" or name.startswith("local_code."):
            location = getattr(loaded, "__file__", None)
            if location:
                require(Path(location).resolve().is_relative_to(runtime.resolve()),
                        "Crossed implementation import: " + location)
    print(json.dumps({"implementation": member, "status": "passed", "cases": cases,
                      "runtime": str(runtime), "imported_modules": len(imported),
                      "data": "synthetic only; not MNIST accuracy"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=(*MEMBERS, "all"), default="all")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        require(args.implementation in MEMBERS, "Worker requires one implementation")
        run_member(args.implementation)
        return
    members = MEMBERS if args.implementation == "all" else (args.implementation,)
    for member in members:
        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                        "--implementation", member, "--worker"], check=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=120)
    print("PASS: isolated synthetic functionality checks; no archived results modified.")


if __name__ == "__main__":
    main()
