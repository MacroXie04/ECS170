# Stage 3 on HarmonyOS

This directory contains the HarmonyOS ArkTS app, native batch preprocessing,
model-export compatibility probe, and the earlier emulator training implementation.
All optimizer steps in the app execute on HarmonyOS. Host Python exports graphs,
prepares data and collects evidence; it must not be used to substitute host training.

## Current status

- Source configuration: PR #6, commit `41cb6f084a7722b2fb46f9c9cffb45132f9eb15d`.
- `protocol.json`: 72 requested runs (3 datasets × 8 configurations × 3 seeds).
- `prepare_protocol.py`: regenerates the protocol directly from the repository registry.
- Signed app built and installed on a physical HUAWEI MateBook Pro (2in1, aarch64).
- Full CIFAR baseline export has 814,570 parameters, matching the PR's PyTorch model.
- Full graph conversion currently fails in MindSpore Lite 2.7.0 with an unsupported
  `None` tensor type. Formal PR6 training has **not** started and no real-device
  accuracy or full training duration is claimed.
- `Stage3App` currently runs compatibility probes only and rejects formal jobs.
  Its timer uses `CLOCK_MONOTONIC`; optimizer-step time is recorded separately from
  graph loading and total probe duration. The display refreshes between synchronous steps.

## Build

Install Huawei DevEco Studio with HarmonyOS SDK API 26. Set `DEVECO_HOME` to its
`Contents` directory if it is not installed in the default user Applications folder.

```sh
python3 harmonyos/prepare_protocol.py
sh harmonyos/build_app.sh
```

The build copies `Stage3App/build-profile.template.json5` to a local ignored
`build-profile.json5` if needed. Open the app in DevEco Studio and configure official
automatic signing for the connected physical device. The current bundle identifier
is `com.example.mnistprobe` because it uses the existing local development profile.
Never commit the generated signing configuration or developer credentials.

## Graph compatibility probe

`export_pr6_probe.py` requires Python 3.11, MindSpore 2.7.0 and NumPy 1.26.4.
It exports a random-initialized full CIFAR baseline training graph. It does not yet
copy the exact PyTorch initialization; therefore this graph is only a compatibility
probe, not a formal experiment. Conversion uses the official Lite 2.7.0 converter:

```sh
python export_pr6_probe.py protocol.json models/pr6_cifar_probe
converter_lite --fmk=MINDIR --trainModel=true \
  --modelFile=models/pr6_cifar_probe.mindir \
  --outputFile=models/pr6_cifar_probe --inputDataFormat=NCHW
```

Formal implementation must retain the original train/validation split, train-only
normalization, full-resolution ORL, BatchNorm, Dropout, optimizer semantics,
cosine learning-rate schedule, partial final batches and validation-based selection.
The training app must not silently replace these with the old small CNN protocol.

## Earlier emulator implementation

`legacy/` archives the earlier, different experiment matrix and orchestration.
`legacy/Index.ets` is the original full-training app page. Its 3 configurations per
dataset are **not** the 8 configurations from PR #6. Its scripts retain their original
workspace layout assumptions; the original README describes that layout. Historical
emulator metrics must not be presented as physical-device or PR6 results.

Datasets, model binaries, build outputs, credentials and raw device logs are ignored.
Add verified compact metrics and reproducibility metadata explicitly after real runs.
