"""Exercise Xiaoyu and Aileen's Stage 3 pipelines on small synthetic datasets.

Each member runs in a separate process and a temporary copy of their source.
The archived code, results, and course data are never used as writable runtime
directories. Hongzhe and Zizheng have their own Stage 3 unit tests.

Usage: python tests/integration/smoke_stage3.py [--member xiaoyu|aileen]
These one-epoch checks validate integration, not benchmark accuracy.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[2]
MEMBERS = ("xiaoyu", "aileen")
DATASETS = {"MNIST": ((28, 28), 10, 0),
            "ORL": ((112, 92, 3), 40, 1),
            "CIFAR": ((32, 32, 3), 10, 0)}


def make_data(runtime):
    import numpy as np

    folder = runtime / "data" / "stage_3_data"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for name, (shape, classes, offset) in DATASETS.items():
        data = {}
        for split, count in (("train", classes * 2), ("test", classes)):
            samples = []
            for i in range(count):
                image = rng.integers(0, 256, size=shape, dtype=np.uint8)
                if name == "ORL":
                    image = np.repeat(image[..., :1], 3, axis=-1)
                samples.append({"image": image, "label": i % classes + offset})
            data[split] = samples
        with (folder / name).open("wb") as stream:
            pickle.dump(data, stream)


def check_predictions(path, name, accuracy):
    import numpy as np

    with path.open("rb") as stream:
        saved = pickle.load(stream)
    classes = DATASETS[name][1]
    truth, pred = np.asarray(saved["true_y"]), np.asarray(saved["pred_y"])
    assert truth.shape == pred.shape == (classes,), (name, truth.shape, pred.shape)
    assert np.array_equal(truth, np.arange(classes)), name
    assert np.isfinite(pred).all() and np.isin(pred, np.arange(classes)).all(), name
    assert math.isclose(float((truth == pred).mean()), accuracy), name
    return saved


def smoke_xiaoyu(runtime):
    from script_cnn import parse_args, run_experiment
    import numpy as np

    for name in DATASETS:
        args = parse_args(["--dataset", name.lower(), "--epochs", "1",
                           "--batch-size", "8", "--channels", "2", "4",
                           "--hidden-dim", "8", "--device", "cpu",
                           "--run-name", "integration_smoke"])
        scores = run_experiment(args)
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in scores.values()), name
        output = runtime / "result" / "stage_3_result" / name.lower() / args.run_name
        check_predictions(output / "predictions.pkl", name, scores["accuracy"])
        with (output / "training_history.csv").open() as stream:
            history = list(csv.DictReader(stream))
        assert len(history) == 1 and math.isfinite(float(history[0]["train_loss"])), name
        config = json.loads((output / "run_config.json").read_text())
        assert config["train_count"] == DATASETS[name][1] * 2, name
        assert config["test_count"] == DATASETS[name][1], name
        matrix = np.loadtxt(output / "confusion_matrix.csv", delimiter=",")
        assert matrix.shape == (DATASETS[name][1], DATASETS[name][1]), name
        assert matrix.sum() == DATASETS[name][1], name
        assert (output / "cnn_weights.pt").is_file(), name
        print(f"PASS xiaoyu {name}: synthetic training, metrics, predictions, checkpoint", flush=True)


def smoke_aileen(runtime):
    from stage3_common import run_one
    import numpy as np

    for name in DATASETS:
        scores, model = run_one(name, max_epoch=1, use_gpu=False, batch_size=8,
                                conv_channels=[2, 4], hidden_dim=8,
                                track_test=False, verbose=False)
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in scores.values()), name
        assert len(model.loss_history) == 1 and math.isfinite(model.loss_history[0]), name
        assert len(model.data["train"]["y"]) == DATASETS[name][1] * 2, name
        output = runtime / "result" / "stage_3_result" / f"CNN_{name}_prediction_result_2"
        saved = check_predictions(output, name, scores["accuracy"])
        np.testing.assert_array_equal(saved["pred_y"], model.predict(model.data["test"]["X"]))
        print(f"PASS aileen {name}: synthetic training, metrics, saved predictions", flush=True)


def worker(member, runtime):
    import torch

    runtime = Path(runtime).resolve()
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(runtime / "script" / "stage_3_script"))
    os.chdir(runtime)
    torch.set_num_threads(2)
    make_data(runtime)
    {"xiaoyu": smoke_xiaoyu, "aileen": smoke_aileen}[member](runtime)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", choices=(*MEMBERS, "all"), default="all")
    args = parser.parse_args(argv)
    selected = MEMBERS if args.member == "all" else (args.member,)
    for member in selected:
        source = REPO / "implementations" / member / "stage3"
        with tempfile.TemporaryDirectory(prefix=f"ecs170-stage3-{member}-") as folder:
            runtime = Path(folder)
            for name in ("local_code", "script"):
                shutil.copytree(source / name, runtime / name,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            env.pop("PYTHONPATH", None)
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--worker", member, str(runtime)], cwd=runtime,
                           env=env, check=True)
    print(f"PASS: {len(selected) * len(DATASETS)} isolated Stage 3 synthetic smoke runs")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker" and sys.argv[2] in MEMBERS:
        worker(sys.argv[2], sys.argv[3])
    else:
        main()
