"""Freeze the PR #6 registry without running training on the host."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from local_code.stage_3_code.Experiment_Config import (
    REGISTRY, FORMAL_SEEDS, SPLIT_SEED, DATASET_ORDER,
    CHECKPOINT_RULE, BEST_CONFIG_RULE,
)


def main():
    source = ROOT.parent / 'local_code/stage_3_code/Experiment_Config.py'
    runs = []
    for dataset in DATASET_ORDER:
        for name, config in REGISTRY[dataset].items():
            for seed in FORMAL_SEEDS:
                runs.append(dict(id=f'{dataset}_{name}_seed{seed}', seed=seed,
                                 config=config.to_dict(), fingerprint=config.fingerprint()))
    protocol = dict(source_pr='https://github.com/MacroXie04/ECS170/pull/6',
                    source_commit='41cb6f084a7722b2fb46f9c9cffb45132f9eb15d',
                    registry_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    split_seed=SPLIT_SEED, checkpoint_rule=CHECKPOINT_RULE,
                    best_config_rule=BEST_CONFIG_RULE, runs=runs)
    (ROOT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(f'Frozen {len(runs)} runs; this is configuration, not completed training.')


if __name__ == '__main__':
    main()
