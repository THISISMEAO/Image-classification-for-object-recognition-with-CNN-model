import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy

import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2) #Neural networks usually start with random weights, these fix the randomness
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- object initialization section -------------------
    data_obj = Dataset_Loader(
    'CIFAR dataset',
    'object dataset',
    folder_path='./data/stage_3_data',
    file_name='CIFAR'
    )

    method_obj = Method_CNN(
    'cnn', '',
    input_channels=3,
    num_classes=10,
    input_height=32,
    input_width=32,
    learning_rate=0.001,
    max_epoch=20,
    batch_size=64,
    dropout_rate=0.5,
    optimizer_name='adam',
    dataset_name='CIFAR',
)   

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = './result/stage_3_result/CNN_' #changed result folder pathway so it is inside of project folder
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    # ---- evaluate section ---------------------------------
    evaluate_obj = Evaluate_Accuracy('multi-metric evaluator', '') #changed accuracy to multi-metric
 

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    result = setting_obj.load_run_save_evaluate() #changed to result since no longer doing k-fold
    print('************ Overall Performance ************')
    print('CNN Accuracy for CIFAR: ' + str(result)) #changed to only print result because there is no std
    print('************ Finish ************')

    method_obj.plot_metrics(method_obj.epoch_numbers,
                            method_obj.train_losses,
                            method_obj.train_accuracies,
                            method_obj.data['test']['y'],
                            method_obj.y_probs)
    # ------------------------------------------------------
    