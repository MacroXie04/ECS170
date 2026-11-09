# MNN training on a physical HarmonyOS computer

**MNN + OpenCL buffer mode passed GPU-assisted training checks on the MOR-M1
computer (Maleoon 935 GPU). This is mixed CPU/GPU execution. The tested Vulkan
paths failed numerical validation and must not be used for formal training.**

This optional port does not replace the completed Stage 3 MindSpore CPU
experiments or their accuracy results. MNN is pinned to
`5b74a931887ba5b4d9f65b222cc18dd1a16599a8` (3.6.2), with no upstream source patches.
The native OpenHarmony ARM64 build enables training/autodiff, OpenCL, and Vulkan.

## Verified results, 2026-11-08

The device trained on 1,024 genuine instructor Stage 3 MNIST optimization
examples, using eight frozen augmented batches of 128, repeated for five passes
(40 optimizer steps). All model execution and optimization occurred on the
physical HarmonyOS computer. The Mac prepared data, built the app, and collected
and compared results.

| MNN mode | First → fifth pass mean CE | Changed learned weights | Total device job | Mean step, steps 6–40 | Independent reload max error |
|---|---:|---:|---:|---:|---:|
| CPU, 4 threads | 2.36185 → 0.478834 | 445,220 | 9.008 s | 214.919 ms | 0 |
| OpenCL buffer + CPU fallback | 2.36224 → 0.479628 | 445,227 | 63.448 s | 784.459 ms | 0 |

OpenCL was **3.65× slower per steady training step** in this implementation.
These final timing runs disable GPU timestamp profiling. Operator callbacks
remain enabled. Total time includes initial shader compilation/tuning, first-step
gradient auditing, setup, snapshots, export, and same-process reload; host
transfers and polling are excluded. The separate profiled run also completed
40 steps and recorded positive GPU timestamps, with 848.738 ms mean steady step
time. The slowdown is not a peak hardware rating.

The exact initial learned weights matched the CPU control. The first-step
OpenCL gradient had maximum absolute error `2.38419e-6` and relative L2 error
`1.54647e-6`, passing the per-tensor CPU comparison. Every learned parameter
tensor changed during training; BatchNorm running statistics are excluded from
that check. Saved inference graphs reloaded with zero logit error in new app
processes, after the training processes were stopped.

[Machine-readable results, operator traces, data provenance and artifact hashes](../reports/mnn_gpu_training.json)

## GPU placement and rejected modes

- **OpenCL buffer (`opencl_buffer`)**: forward convolution, filter-gradient
  matrix multiplication, reductions, and optimizer arithmetic are on OpenCL.
  One dynamic-weight `Deconvolution` per backward pass, used for the convolution
  input gradient, falls back to CPU. The extra first-step gradient audit adds
  one more backward pass. All other traced training operators use OpenCL.
- **OpenCL automatic/image (`opencl`)**: completed a three-step smoke test, but
  training convolutions and additional operations fell back to CPU. Buffer mode
  has better coverage and is the validated default GPU candidate.
- **Vulkan (`vulkan`)**: operator traces marked GPU execution, but the first-step
  gradient relative L2 error was `2.11471`; the second convolution's weight
  gradient was entirely zero when the CPU reference was nonzero. The 40-step
  run ended with nonfinite learned weights. It is rejected, despite the reported
  loss decreasing toward the uniform-class baseline.
- **Vulkan batch recording (`vulkan_batch`)**: produced an initial loss around
  `4.92e22` and a saved/reloaded logit mismatch around `6.98e8`. It is rejected.
  The probe now explicitly rejects nonfinite or mismatched reload outputs.

A requested runtime must exist; silent complete fallback to CPU is rejected.
Callbacks record output tensor backends per operator/phase: CPU=0, OpenCL=3,
Vulkan=7. These counts are **not percentages of computation time**. CPU graph
construction, scheduling, materialization and CPU/GPU transfers remain part of
the application. `lastGpuMs` samples the last GPU execution at forward and
optimizer boundaries; it is not total step GPU time. It is -1 in unprofiled builds.

The Vulkan adapter query resolves public loader functions with `dlsym` to avoid
a symbol collision with MNN's same-named Vulkan function-pointer variables.
Native work runs asynchronously, keeping the app's elapsed timer responsive.

## Data and protocol limits

Input hashes, sample order, and exact first-batch preprocessing parity with the
unchanged PR #6 loader are recorded. Seed is 2026. The architecture port uses
32/64-channel 5x5 convolutions, BatchNorm, ReLU, max pooling, a 128-unit hidden
layer, dropout 0.25, and ten logits. Cross-entropy and Adam use learning rate
0.001, beta1 0.9, beta2 0.999, and zero weight decay; precision is High/FP32.

This is a subset capability experiment, not the full Stage 3 protocol. It has
no cosine schedule, validation selection, or held-out test accuracy. MNN
initialization, BatchNorm statistics, Adam epsilon conventions and random masks
can differ from PyTorch and the original MindSpore port. Gradient agreement is
against MNN CPU, not proof of PyTorch equivalence. Do not insert these timings
into the existing four-platform benchmark as an equivalent GPU row.

## Reproduction

Use the DevEco SDK API 26, signed app configuration, physical USB target and
prepared Stage 3 data from the parent README. `MNN_ROOT` and `DEVECO_HOME`
override local defaults. The build script fetches a fresh MNN clone at the pinned
commit and leaves an existing checkout at another commit unchanged.

For a profiled backend audit:

```sh
sh harmonyos/mnn/build_mnn.sh
sh harmonyos/mnn/build_probe.sh
python harmonyos/mnn/prepare_data.py --output harmonyos/results/mnn-inputs
python harmonyos/mnn/run_probe.py --target PHYSICAL_USB_TARGET \
  --data harmonyos/results/mnn-inputs --output harmonyos/results/mnn-smoke \
  --backends cpu opencl_buffer --steps 3 --install
python harmonyos/mnn/compare_gradients.py harmonyos/results/mnn-smoke \
  --backends opencl_buffer --require opencl_buffer \
  --output harmonyos/results/mnn-smoke/numerical_comparison.json
```

Then repeat with `--steps 40` and a new output directory. The default backends
are CPU and OpenCL buffer. Vulkan modes are available only for explicit
failure reproduction; a successful launch or `completed` one-step smoke state
must never substitute for the gradient and long-run validation gates.

For the unprofiled timing build:

```sh
export MNN_BUILD_DIR="$HOME/Developer/MNN-harmony/build-ohos-fast"
MNN_GPU_PROFILE=OFF sh harmonyos/mnn/build_mnn.sh
sh harmonyos/mnn/build_probe.sh
# Run the same device test with --install and a new output directory.
```

The device must be unlocked to launch the app. `--wait-unlock 600` waits for the
user to unlock it; it does not unlock the device. The runner restores the prior
`job.json` and stops the app afterward.

For independent-process reload, use the saved training run's `id` from its JSON:

```sh
python harmonyos/mnn/run_probe.py --target PHYSICAL_USB_TARGET \
  --data harmonyos/results/mnn-inputs --output harmonyos/results/mnn-reload \
  --backends opencl_buffer --reload-model SAVED_RUN_ID
```

Require a different `device_pid`, the expected `reloadedFrom` run ID, and finite
logits matching at `atol=1e-4, rtol=1e-4`. A model parameter snapshot is also
saved; resuming Adam optimizer history is not part of this probe.

`sh harmonyos/build_app.sh` builds the original app without MNN. Raw datasets,
models, signed packages, signing credentials and device logs remain ignored.

## Upstream references

- [MNN MNIST training example](https://github.com/alibaba/MNN/blob/5b74a931887ba5b4d9f65b222cc18dd1a16599a8/tools/train/source/demo/MnistUtils.cpp)
- [OpenHarmony OpenCL loader paths](https://github.com/alibaba/MNN/blob/5b74a931887ba5b4d9f65b222cc18dd1a16599a8/source/backend/opencl/core/runtime/OpenCLWrapper.cpp)
- [GPU/CPU build options](https://github.com/alibaba/MNN/blob/5b74a931887ba5b4d9f65b222cc18dd1a16599a8/CMakeLists.txt)
