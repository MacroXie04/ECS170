# Stage 4 joke-generation held-out recheck

This 2026-11-22 recheck runs classmates' existing generators on a held-out joke partition. It does not replace the original PR or PDF results. The instructor joke data and new checkpoints remain local; only aggregate metrics and reproduction scripts are committed.

## Protocol

- The frozen split IDs in result/stage_4_result/runs/generation_split_ids.json allocate the 1,622 source rows to 1,297 training, 164 validation and 161 test jokes, keeping normalized duplicates together. Aileen's loader excludes one short validation joke, leaving 163 validation jokes for that implementation.
- Each implementation builds its own vocabulary from its training jokes only and uses its own tokenizer and model. No test targets are used for checkpoint selection. Aileen restores the lowest-validation-loss checkpoint from a 60-epoch run; Xiaoyu scores the fixed final epoch 15 from her original training schedule.
- Zizheng's three previously saved checkpoints were independently reloaded and scored on his own deterministic, disjoint 1,284/161/161 split. His test set is not the frozen split above.
- The new Aileen and Xiaoyu checkpoints were reloaded after training and reproduced their test loss. Raw-data, split and source hashes are recorded in the aggregate files.

| Implementation and evidence | Test jokes | Scored targets | Vocabulary | RNN perplexity | LSTM perplexity | GRU perplexity |
|---|---:|---:|---:|---:|---:|---:|
| Hongzhe original baseline | 161 | 2,961 | 4,201 | 220.449 | 213.543 | 207.057 |
| Zizheng saved-checkpoint replay | 161 | 2,529 | 1,616 | 39.665 | 37.512 | 36.329 |
| Aileen new held-out run | 161 | 2,683 | 1,618 | 36.518 | 33.574 | 34.401 |
| Xiaoyu new held-out run | 161 | 2,696 | 4,184 | 250.969 | 280.363 | 255.173 |

Perplexity depends on vocabulary and the exact predicted tokens. Those differ across these implementations, so the rows are not a direct ranking. Compare the recurrent cells within a row, subject to each row's own training settings. The new Aileen run selected best validation epochs 20/33/24 for RNN/LSTM/GRU. The new Xiaoyu run used 15 epochs for each. Aileen's PR listed validation perplexity only; Xiaoyu's shared PDF listed training cross-entropy on all 1,622 jokes. Neither earlier number was a held-out test score.

## Reproduce

From the repository root, use a PyTorch environment and the instructor data at data/stage_4_data/:

~~~bash
python script/stage_4_script/replay_zizheng_generation.py --repo . --data-root data/stage_4_data --result-root result/stage_4_result/generate --output /tmp/ecs170_zizheng_replay.json
python script/stage_4_script/evaluate_azheng_heldout.py --output /tmp/ecs170_azheng_heldout --run --device cuda
python script/stage_4_script/evaluate_xiaoyu_heldout.py --output /tmp/ecs170_xiaoyu_heldout --device cuda
~~~

Pass --data-dir to either script if the course data live outside this checkout. Use a fresh output directory for each run. The scripts write model checkpoints to that local directory. The committed aggregate records are in result/stage_4_result/team_generation_recheck_20261122/. The Aileen summary excludes generated text; Xiaoyu's four JSON files contain only hashes, settings, histories and aggregate scores.
