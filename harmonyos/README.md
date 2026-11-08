# Stage 3 training on a physical HarmonyOS computer

The source protocol is [PR #6](https://github.com/MacroXie04/ECS170/pull/6),
commit `41cb6f084a7722b2fb46f9c9cffb45132f9eb15d`. The initial requested subset is
ORL kernel5 / MNIST kernel5 / CIFAR avgpool, seed 2026, with 120 / 15 / 40 epochs.
`protocol.json` retains the complete 72-run matrix; `selected_runs.json` records
this initial subset. Successful smoke tests are not completed formal experiments.

## Execution and protocol

All forward/backward passes and optimizer updates execute on the connected
HarmonyOS computer, through its native MindSpore Lite C API. The Mac prepares
real data, the original PyTorch random initialization, sampling schedules and
graphs, then transfers files and collects evidence. It does not perform the
formal training. The app displays a monotonic elapsed timer and per-epoch progress.

The original split seed, train-only normalization, full 112×92 ORL resolution,
network, loss, optimizer, cosine schedule and validation selection are retained.
Sampling and augmentation choices reproduce the PR's CPU random streams;
native preprocessing is compared against the original loader's first batch.
Different frameworks need not generate identical Dropout masks or bitwise results.
Evaluation CE after each epoch is labelled `trainEvalLoss`; it is not the PR's
stochastic minibatch training loss. Final test data is evaluated only after
validation-based model selection. Short final training batches have their own
graph and exchange optimizer state with the full-batch graph.

## Compatibility repairs

- BatchNorm is decomposed into supported arithmetic with an explicit rsqrt
  derivative. Each training graph writes batch mean and unbiased variance;
  the native bridge applies the original 0.9/0.1 moving-statistics update after
  each step. This avoids folding the old running statistics into constants.
- Adam uses the native optimizer operation, with epsilon adjusted for the
  framework's bias-correction convention. Its power accumulators use one-element
  tensors so state can be copied between full and partial batches.
- Converter inputs are explicitly NCHW. The app checks shapes and learning-rate
  readback. Native library linking includes the C++ runtime statically.
- Native model handles separate training and evaluation. Evaluation models are
  rebuilt to avoid stale packed weights. Weight copies use owned tensor storage.
  Inference export occurs after training; reload is checked in a new app process.

## Reproduction

Use DevEco Studio with HarmonyOS SDK API 26 and a signed development app. The
bundle identifier is `com.example.mnistprobe`, matching the local signing profile.
The generated `Stage3App/build-profile.json5` and credentials must remain ignored.

Install `requirements-host.txt` in the host preparation environment, and
`requirements-export.txt` in a separate Linux environment. Supply the official
Stage 3 MNIST/ORL/CIFAR pickle directory to the unchanged PR loader:

```sh
python harmonyos/prepare_protocol.py
python harmonyos/prepare_data.py /path/to/official/pickles
python harmonyos/prepare_initial_weights.py --run orl_kernel5_seed2026
python harmonyos/prepare_schedule.py --run orl_kernel5_seed2026
python harmonyos/prepare_batch_reference.py --run orl_kernel5_seed2026
# Linux, with harmonyos/data and harmonyos/models/initial available:
python harmonyos/export_models.py harmonyos --run orl_kernel5_seed2026
python harmonyos/convert_models.py harmonyos/models/orl_kernel5_seed2026 \
  --converter /path/to/mindspore-lite-2.7.0/tools/converter/converter/converter_lite
# Mac:
sh harmonyos/build_app.sh
hdc -t USB_SERIAL install harmonyos/Stage3App/entry/build/default/outputs/default/entry-default-signed.hap
python harmonyos/run_device.py --target USB_SERIAL --run orl_kernel5_seed2026 --smoke
python harmonyos/run_device.py --target USB_SERIAL --run orl_kernel5_seed2026
```

Repeat preparation/export for the other selected runs. The controller rejects
network/emulator targets and checks that the physical target reports `2in1`.

## Evidence and limits

Each local `results/<run-id>/` contains the exact job, protocol, data/model/source
hashes, per-epoch progress, before/after weights, best inference model, saved state,
independent-process reload outputs and timing. `optimizerMs` measures native
steps, `trainingMs` the training loops, and `elapsedMs` the device job including
loading/evaluation/checkpoint work. Host transfer time is excluded and labelled.
An interrupted or failed run must never be reported as completed. Binary datasets,
models, credentials, build outputs and raw logs are intentionally ignored.
Verified compact experiment reports are published in [reports/README.md](reports/README.md).
ORL has completed all 120 epochs; subsequent runs are listed there only after verification.

`legacy/` preserves earlier emulator/probe implementations. Their smaller CNNs,
different protocols and historical metrics are not physical-device PR #6 results.

## Arithmetic checks

`tests/make_batchnorm_reference.py` creates a small synthetic PyTorch reference.
`tests/test_batchnorm.py` and `tests/test_optimizers.py` compare graph arithmetic;
`tests/export_diagnostics.py` exports the three-step device diagnostics. Run
`tests/run_device_diagnostics.py --target USB_SERIAL --models /path/to/converted/diagnostics
--reference /path/to/bn_reference.npz` in the host preparation environment.
These synthetic checks do not train Stage 3 data on the Mac. Current device
BN, Adam and Nesterov results are in `validation/native_arithmetic.json`.

`run_selected.py --target USB_SERIAL` runs smoke checks and formal experiments
serially, rejecting missing sample coverage, nonfinite weights or reload failures.
Use repeated `--run` arguments to select only remaining runs.
