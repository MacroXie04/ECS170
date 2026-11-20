#!/usr/bin/env python3
"""Run from any directory; relative paths are resolved from repository root."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from local_code.stage_4_code.Experiment_Runner import main

if __name__ == '__main__':
    main()
