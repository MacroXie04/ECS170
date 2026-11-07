# ORL follow-up experiment

Supplementary experiment. It does not replace the formal ORL results.

Proposed after the formal ORL test scores were known: the formal final model scored 95.00% and the per-config test means were visible. Any result here is therefore not a blind estimate.

## Cross-validation on the 360 training images

Selection rule: Most correct of the 360 pooled held-out predictions; then lowest mean held-out cross-entropy; then fewest parameters.

| Config | Correct | Accuracy % | Mean cross-entropy | Train s |
|---|---:|---:|---:|---:|
| depth_3stage | 360/360 | 100.00 | 0.0033 | 13 |
| stride2 **(selected)** | 360/360 | 100.00 | 0.0011 | 10 |
| baseline | 359/360 | 99.72 | 0.0053 | 13 |
| kernel5 | 359/360 | 99.72 | 0.0056 | 16 |
| avgpool | 359/360 | 99.72 | 0.0140 | 13 |
| no_dropout | 359/360 | 99.72 | 0.0109 | 13 |
| mse_loss | 358/360 | 99.44 | 0.0329 | 14 |
| hidden_small | 300/360 | 83.33 | 1.0287 | 13 |

## Test results

| Model | Accuracy % | Precision (macro) % | Recall (macro) % | F1 (macro) % |
|---|---:|---:|---:|---:|
| 3-seed ensemble (primary) | 97.50 | 96.25 | 97.50 | 96.67 |
| Seed 2026 | 97.50 | 96.25 | 97.50 | 96.67 |
| Seed 2027 | 100.00 | 100.00 | 100.00 | 100.00 |
| Seed 2028 | 95.00 | 92.50 | 95.00 | 93.33 |

Per-seed mean: 97.50 ± 2.50%.

Formal result, for comparison: `kernel5`, final model 95.00%, config mean 97.50%.

The test set has 40 images, so one image is 2.5 percentage points.
