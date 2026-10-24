import os, sys
# make `local_code` importable no matter where the script is launched from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # keep the relative data/result paths valid

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Pre_Split import Setting_Pre_Split
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

#---- Multi-Layer Perceptron script (stage 2): the tuned model from the 2-4 search ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    #---- objection initialization section ---------------
    train_obj = Dataset_Loader('mnist train', '')
    train_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    train_obj.dataset_source_file_name = 'train.csv'

    test_obj = Dataset_Loader('mnist test', '')
    test_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    test_obj.dataset_source_file_name = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron (tuned)', '')
    method_obj.hidden_sizes = [512, 256]
    method_obj.activation = 'relu'
    method_obj.batch_norm = True
    method_obj.dropout = 0.0
    method_obj.loss_name = 'ce_ls'          # cross-entropy with label smoothing 0.1
    method_obj.optimizer_name = 'sgd'       # SGD with momentum 0.9
    method_obj.learning_rate = 0.05
    method_obj.weight_decay = 0.0
    method_obj.batch_size = 128
    method_obj.scheduler = 'cosine'
    method_obj.max_epoch = 25
    method_obj.use_gpu = False              # set True to try CUDA / MPS

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_tuned_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Pre_Split('pre-partitioned train/test', '')
    evaluate_obj = Evaluate_Metrics('multiclass metrics', '')
    #------------------------------------------------------

    #---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(train_obj, method_obj, result_obj, evaluate_obj, test_obj)
    setting_obj.print_setup_summary()
    scores = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    for k, v in scores.items():
        print('%-20s %.4f' % (k, v))
    print('training time: %.1fs, parameters: %d' % (method_obj.method_training_time, method_obj.num_parameters()))

    # learning convergence curves: training loss and training accuracy
    epochs = range(1, len(method_obj.loss_history) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, method_obj.loss_history, marker='o', ms=3)
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Training loss (label-smoothed cross-entropy)')
    ax[0].set_title('Learning convergence'); ax[0].grid(alpha=0.3)
    ax[1].plot(epochs, method_obj.train_acc_history, marker='o', ms=3, color='tab:green')
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Training accuracy')
    ax[1].set_title('Training accuracy'); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../../result/stage_2_result/convergence_curve_tuned.png', dpi=150)
    print('saved ../../result/stage_2_result/convergence_curve_tuned.png')
    print('************ Finish ************')
    #------------------------------------------------------