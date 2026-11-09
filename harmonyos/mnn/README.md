# MNN training capability probe on a physical HarmonyOS computer

This optional build tests another training runtime after the system-provided
MindSpore Lite runtime rejected GPU contexts. It does not replace the completed
Stage 3 CPU experiments or their reported accuracy.

MNN is pinned to `5b74a931887ba5b4d9f65b222cc18dd1a16599a8` (3.6.2). The
OpenHarmony ARM64 build enables its training/autodiff module, OpenCL, Vulkan
buffer backend, and GPU timestamp profiling. No MNN upstream files are patched.

## Current validation status (2026-11-08)

Both the optional MNN app and the original app build successfully with the
HarmonyOS SDK. The signed MNN app is installed on the physical MOR-M1 computer,
and real input data passes the provenance checks below. Device execution is
currently blocked at app launch: HarmonyOS returns `10106102` because the screen
is locked. **No MNN CPU or GPU training step has yet been verified.** This is an
application-launch restriction, not evidence that OpenCL or Vulkan is unsupported.
See [preparation status](../reports/mnn_probe_preparation.json).

## What the probe measures

- Real instructor MNIST data from the existing Stage 3 optimization split.
  The first 1,024 shuffled examples are preprocessed by the unchanged PR #6
  loader, using seed 2026. Input hashes, sample order, and exact first-batch
  parity are recorded. The eight augmented batches are frozen and repeated.
- A port of the PR #6 MNIST `kernel5` architecture: 32/64-channel 5x5
  convolutions, BatchNorm, ReLU, max pooling, a 128-unit hidden layer, dropout
  0.25, and ten logits. Cross-entropy and Adam use learning rate 0.001,
  beta1 0.9, beta2 0.999, and zero weight decay. Batch size is 128.
- Explicit CPU, OpenCL, and Vulkan requests, each in a new app process.
  A missing requested runtime is rejected instead of silently accepting a
  CPU-only run. Execution callbacks record the output tensor backend for each
  operator and phase. CPU=0, OpenCL=3, Vulkan=7; -1 means no backend attached.
- Finite losses, changed learned weights (excluding BatchNorm running statistics),
  inference graph/parameter export, same-process reload, and a separately
  launched app for independent-process reload.
- Device monotonic elapsed time and per-step wall time, including graph creation,
  loss materialization, differentiation, optimizer work, and synchronization.
  UI/host polling time is outside `stepMs`. `lastGpuMs` samples the last GPU
  execution at the forward and optimizer boundaries, **not total step GPU time**.

The architecture port does not establish numerical equivalence to PyTorch or the
MindSpore implementation. MNN initialization, BatchNorm running statistics, Adam
epsilon conventions, and random masks can differ. This subset probe has no
cosine schedule, validation selection, or test accuracy measurement. GPU
profiling, callbacks, first-use shader compilation, and tuning also affect
timing. Do not put these times into the existing four-platform benchmark as if
they measured the same execution path.

## Build and run

Use the same DevEco SDK, signed app configuration, connected physical target,
and prepared Stage 3 data as described in the parent README. `MNN_ROOT` and
`DEVECO_HOME` can override the local defaults. A fresh MNN clone is fetched at
the pinned commit; an existing checkout at another commit is left untouched.

```sh
sh harmonyos/mnn/build_mnn.sh
sh harmonyos/mnn/build_probe.sh
python harmonyos/mnn/prepare_data.py --output harmonyos/results/mnn-inputs
python harmonyos/mnn/run_probe.py --target PHYSICAL_USB_TARGET \
  --data harmonyos/results/mnn-inputs --output harmonyos/results/mnn-smoke \
  --steps 3 --install
```

After checking backend availability, repeat with `--steps 40` and a new output
directory. The device must be unlocked to launch the app. `--wait-unlock 600`
can wait up to ten minutes for the user to unlock it; it does not unlock the
device. The runner backs up and restores `job.json` and stops the app afterward.

For an independent-process reload, use the saved run's `id` from its JSON result:

```sh
python harmonyos/mnn/run_probe.py --target PHYSICAL_USB_TARGET \
  --data harmonyos/results/mnn-inputs --output harmonyos/results/mnn-reload \
  --backends opencl --reload-model SAVED_RUN_ID
```

The reload result must report the same saved run, a different `device_pid`, and
a small finite `reloadMaxLogitError`. The saved parameter file is a model
snapshot; resuming Adam optimizer history is not part of this probe.

`sh harmonyos/build_app.sh` still builds the original application without MNN.
Raw datasets, binaries, signed packages, and device logs remain ignored by Git.

## Upstream references

- [MNN MNIST training example](https://github.com/alibaba/MNN/blob/5b74a931887ba5b4d9f65b222cc18dd1a16599a8/tools/train/source/demo/MnistUtils.cpp)
- [OpenHarmony OpenCL loader paths](https://github.com/alibaba/MNN/blob/5b74a931887ba5b4d9f65b222cc18dd1a16599a8/source/backend/opencl/core/runtime/OpenCLWrapper.cpp)
- [GPU/CPU build options](https://github.com/alibaba/MNN/blob/5b74a931887ba5b4d9f65b222cc18dd1a16599a8/CMakeLists.txt)
