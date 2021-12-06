import argparse
import numpy as np
import os
from pathlib import Path

from sklearn.model_selection import KFold
import pdb

speaker_id_arr_dict = {'msp-improv': np.arange(0, 12, 1), 
                       'crema-d': np.arange(1001, 1092, 1),
                       'iemocap': np.arange(0, 10, 1)}

if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--feature_type', default='mel_spec')
    parser.add_argument('--norm', default='min_max')
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--data_dir', default='/media/data/public-data/SER')
    parser.add_argument('--training_save_dir', default='/media/data/projects/speech-privacy')
    args = parser.parse_args()

    data_set_str, feature_type = args.dataset, args.feature_type
    
    # get the 5 different test folds
    speaker_id_arr = speaker_id_arr_dict[data_set_str]    
    train_array, test_array = [], []
    
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(speaker_id_arr):
        
        # 80% are training (80% of data on a client is for training, rest validation), and 20% are test
        train_array.append(speaker_id_arr[train_index])
        test_array.append(speaker_id_arr[test_index])

    # if we dont have data ready for experiments, preprocess them first
    for fold_idx in range(5):
        cmd_str = 'python3 preprocess_federate_data.py --dataset ' + data_set_str
        cmd_str += ' --norm ' + args.norm
        cmd_str += ' --test_fold ' + 'fold' + str(fold_idx+1)
        cmd_str += ' --feature_type ' + feature_type
        cmd_str += ' --pred ' + args.pred
        cmd_str += ' --data_dir ' + args.data_dir
        cmd_str += ' --training_save_dir ' + args.training_save_dir

        # we dont have these speaker array when combining dataset
        cmd_str += ' --train_arr '
        for train_idx in train_array[fold_idx]:
            cmd_str += str(train_idx) + ' '
        cmd_str += ' --test_arr '
        for test_idx in test_array[fold_idx]:
            cmd_str += str(test_idx) + ' '
        
        print(cmd_str)
        os.system(cmd_str)
