"""Train and evaluate every MNIST configuration, then rebuild the summaries.

    python script/stage_3_script/script_cnn_mnist.py --configs baseline --seeds 2026
"""
import sys

from run_stage3 import dataset_entry_point

if __name__ == '__main__':
    sys.exit(dataset_entry_point('mnist'))
