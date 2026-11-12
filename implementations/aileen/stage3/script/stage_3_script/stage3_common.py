import os, sys
# make `local_code` importable no matter where the script is launched from
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # keep the relative data/result paths valid

import numpy as np
import torch
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Setting_Pre_Loaded import Setting_Pre_Loaded
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Result_Saver import Result_Saver

METHODS = {'MNIST': Method_CNN_MNIST, 'ORL': Method_CNN_ORL, 'CIFAR': Method_CNN_CIFAR}
RESULT_DIR = '../../result/stage_3_result/'


def data_folder():
    # the zip unpacks to data/stage_3_data/ or data/stage_3_data/stage_3_data/
    for p in ['../../data/stage_3_data/stage_3_data/', '../../data/stage_3_data/']:
        if os.path.exists(p + 'MNIST'):
            return p
    raise FileNotFoundError('put the MNIST / ORL / CIFAR files in data/stage_3_data/')


def run_one(name, seed=2, **overrides):
    """Train one CNN on dataset `name`; overrides are Method_CNN hyper-parameters. Returns (scores, method)."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    data_obj = Dataset_Loader(name, '')
    data_obj.dataset_source_folder_path = data_folder()
    data_obj.dataset_source_file_name = name
    data_obj.single_channel = (name == 'ORL')       # ORL is gray-scale stored as 3 identical channels

    method_obj = METHODS[name]('CNN ' + name, '')
    method_obj.seed = seed
    method_obj.use_gpu = True                        # CUDA (Colab) or MPS if available, otherwise CPU
    for k, v in overrides.items():
        setattr(method_obj, k, v)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = RESULT_DIR + 'CNN_%s_' % name
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.fold_count = seed

    setting_obj = Setting_Pre_Loaded('pre-partitioned train/test', '')
    evaluate_obj = Evaluate_Metrics('multiclass metrics', '')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    scores = setting_obj.load_run_save_evaluate()
    return scores, method_obj
