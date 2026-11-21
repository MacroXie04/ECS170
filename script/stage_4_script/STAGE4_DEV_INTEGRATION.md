# Stage 4 DEV integration

The stage4-dev branch contains the histories of Stage 4 pull requests #11, #12, #13, and #14, merged from main. They address the same two course tasks with different preparation and training protocols. Each contributor retains a separate working entry point, source modules, and results.

| Contribution | Python modules | Entry points | Notes |
|---|---|---|---|
| Hongzhe (#11) | local_code/stage_4_code/ | run_stage4.py; predict_stage4.py; optimization and verification scripts | Verified full-data RNN/LSTM/GRU baseline and later comparisons; see README.md. |
| Zizheng (#12) | local_code/stage_4_code/zizheng/ | script_rnn.py | Independent classifier and generator with saved metrics/models; see README_ZIZHENG.md. |
| Azheng (#13) | local_code/stage_4_code/azheng/ | script_text_classification.py; script_text_generation.py; ablation scripts | Standalone evaluator and existing Stage 1 saver replace imports to absent Stage 2 modules. |
| Xiaoyu (#14) | local_code/stage_4_code/xiaoyu/ | script_classification.py; script_generation.py; script_experiments.py | Own loaders, models, optional GloVe, and plots; see ECS170_Stage4_README.md. |

Entry point paths are relative to script/stage_4_script/. Instructor data are local under data/stage_4_data/ and are not committed. Existing result folders under result/stage_4_result/ remain available. Five Stage 3 CI and integration scripts deleted in #14 were restored during integration.

## Quick checks

From the repository root, use the project Python environment and run:

~~~bash
python -m unittest discover -s tests/stage4 -p 'test_*.py' -q
python script/stage_4_script/run_stage4.py --smoke --device cpu --output result/stage_4_result/smoke/integration --tag integration
python script/stage_4_script/script_rnn.py --task all --model all --smoke --output /tmp/stage4-zizheng-smoke
python script/stage_4_script/script_classification.py --model rnn --epochs 1 --train-limit 200 --test-limit 100 --device cpu --run-name smoke/xiaoyu_classification
python script/stage_4_script/script_generation.py --model rnn --epochs 1 --device cpu --run-name smoke/xiaoyu_generation
~~~

These exercise small data or one-epoch training, so their metrics are not full-data report results. Azheng's scripts have no smoke option and write shared result paths; check these modules on in-memory synthetic data before a full run.

## Reading the saved experiments

Compare RNN, LSTM, and GRU within one contribution. Cross-contribution metrics differ in split, tokenizer, vocabulary, input length, and validation or test usage. In particular, joke perplexities from separate vocabularies or validation versus test sets are not one benchmark. The report distinguishes recorded metrics, PR self-reports, and smoke checks.

The #11 baseline checkpoints bind to the original source revision f82bdcd. Later classifier optimization changed the canonical loader while retaining the historical checkpoint and result files. The strict baseline verifier therefore reports a source-hash mismatch on the current branch. To recheck that baseline, use revision f82bdcd with the original local ignored checkpoints.

Source branch: https://github.com/MacroXie04/ECS170/tree/stage4-dev
