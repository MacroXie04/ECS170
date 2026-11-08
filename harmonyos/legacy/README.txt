ECS 170 Stage 3 — HarmonyOS PC local training

This project runs all optimizer steps, full-set evaluation, model export, and model reload inside the local HarmonyOS PC/2in1 emulator. Python on the Mac only prepares data, exports random training graphs, orchestrates the app, and verifies/plots the returned results.

Official partitions: MNIST 60,000 / 10,000; ORL 360 / 40; CIFAR 50,000 / 10,000.
ORL uses a single identical RGB channel and 2x2 area resize from 112x92 to 56x46. Labels are mapped to zero-based class indices. Per-channel normalization statistics are computed from training data only.

Prespecified experiments: three architectures per dataset (baseline, wide, deep), in plan.json. MNIST 10 epochs; ORL 60 epochs; CIFAR 20 epochs. All examples are used every epoch. Each epoch shuffles the full training partition. CIFAR uses on-device random horizontal flip and a crop shifted by up to four pixels with zero padding. Cross-entropy, SGD momentum 0.9; learning rate drops to 0.3x after 55% of epochs, to 0.1x after 80%. There is no test-based model selection or early stopping. Final-epoch models are evaluated on all official test examples. Test metrics across architectures are descriptive comparisons.

Architecture: each convolution uses 3x3 kernels, stride 1, padding 1, ReLU, and 2x2 max pooling. Baseline widths 8,16 with dense 64; wide widths 16,32 with dense 128; deep widths 16,32,64 with dense 128. Final dense output matches each dataset's class count. All initialization is random, seed 170.

Implementation files:
prepare_stage3.py + local_code/stage_3_code/Dataset_Loader.py: trusted course archive loading, preprocessing, binary export.
local_code/base_class/: original instructor template base classes, copied unchanged.
local_code/stage_3_code/Method_CNN.py: template-compatible dataset-specific method adapters delegating training to HarmonyOS.
export_models.py: MindSpore random training-graph exporter; no optimizer step on host.
convert_models.py: official MindSpore Lite converter invocation.
Stage3App/entry/src/main/ets/pages/Index.ets: actual HarmonyOS training implementation.
run_experiments.py: deploy full data, run each experiment, restart app, reload model, collect evidence.
summarize.py: independent numerical checks, confusion matrices, precision/recall/F1, and learning curves.

Compatibility decisions:
Use FixedRateSGD without unused int32 global_step AssignAdd, unsupported in the device Lite runtime. Epoch-based LR is set by the app. Explicitly enable trainMode immediately after loading. Keep a dedicated training instance and use a weight-copy model for evaluation/export, avoiding evaluation changes to the training instance.

Expected evidence per experiment in results/<id>/:
*_result.json: full epoch curve, counts, optimizer steps, final test score.
*_weights_before.bin, *_weights_after.bin and metadata: actual learned parameter snapshots (optimizer states excluded).
*_test_logits.bin and *_reloaded_logits.bin: outputs before/after process restart.
*_checkpoint.ms: saved training checkpoint (resuming optimization is not tested).
*_inference.ms: saved deployment model.
*_reload.json + device.log: separate-process reload evidence.
verification.json and confusion_matrix.csv: independent host checks.

Use the already installed Stage3 Harmony app and MNIST_PC emulator. To orchestrate the entire fixed plan:
  python3 run_experiments.py
To run a named configuration:
  python3 run_experiments.py mnist_baseline
Existing verified results are skipped. Do not mix stale device files with a new run using the same IDs; archive prior results or use a clean app data directory for a new experiment.
After all experiments:
  ../harmony-mnist-probe/.venv/bin/python summarize.py --require-all

Training uses MindSpore Lite rather than the original PyTorch training runtime. This is a HarmonyOS implementation of the course CNN experiments, not a claim that unmodified Python/PyTorch runs natively on HarmonyOS. No course submission, repository commit, push, or PR is performed.
