# Verified physical-device results

Source: PR #6, seed 2026. Each row passed full epoch/sample coverage, finite and
changed weights, and inference reload in an independent application process.
Total device time includes loading, evaluation and saving; optimizer time records
native training steps. Data-transfer time is excluded. Training CE is evaluated
on the full optimization split after each epoch, with augmentation disabled.

| Run | Epochs | Initial → final train CE | Best validation | Test | Total device time | Optimizer time |
|---|---:|---:|---:|---:|---:|---:|
| orl_kernel5_seed2026 | 120 | 3.69066 → 7.04073e-05 | 100.00% | 95.00% | 530.976 s | 450.069 s |

Only completed, verified runs are listed. Raw models/data remain local; artifact hashes identify them.
