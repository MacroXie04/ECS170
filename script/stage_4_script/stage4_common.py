import os, sys
# make `local_code` importable no matter where the script is launched from
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # keep the relative data/result paths valid

import numpy as np
import torch
from local_code.stage_4_code.azheng.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.azheng.Method_RNN import Method_RNN
from local_code.stage_4_code.azheng.Setting_Pre_Loaded import Setting_Pre_Loaded
from local_code.stage_4_code.azheng.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_1_code.Result_Saver import Result_Saver

RESULT_DIR = '../../result/stage_4_result/'


def review_folder():
    for p in ['../../data/stage_4_data/text_classification/', '../../data/stage_4_data/']:
        if os.path.isdir(p + 'train'):
            return p
    raise FileNotFoundError('put the review data (train/ and test/) in data/stage_4_data/text_classification/')


def run_classifier(rnn_type='rnn', seed=2, loader_options=None, **overrides):
    """Train one recurrent classifier. loader_options: Dataset_Loader attributes (max_len, ...); overrides: Method_RNN attributes."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    data_obj = Dataset_Loader('reviews', '')
    data_obj.dataset_source_folder_path = review_folder()
    data_obj.cache_folder = RESULT_DIR + 'cache/'
    for k, v in (loader_options or {}).items():
        setattr(data_obj, k, v)

    method_obj = Method_RNN('recurrent classifier', '')
    method_obj.rnn_type = rnn_type
    method_obj.seed = seed
    method_obj.use_gpu = os.environ.get('DEVICE') != 'cpu'      # CUDA / MPS if available; DEVICE=cpu forces the CPU
    if os.environ.get('EPOCHS'):
        method_obj.max_epoch = int(os.environ['EPOCHS'])
    for k, v in overrides.items():
        setattr(method_obj, k, v)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = RESULT_DIR + 'RNN_%s_' % rnn_type
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.fold_count = seed

    setting_obj = Setting_Pre_Loaded('pre-partitioned train/test + validation split', '')
    setting_obj.val_fraction = 0.1                  # 10% of the TRAIN reviews: learning curves + best-epoch choice
    evaluate_obj = Evaluate_Metrics('classification metrics', '')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    scores = setting_obj.load_run_save_evaluate()
    return scores, method_obj
