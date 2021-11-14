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
    parser.add_argument('--input_channel', default=1)
    parser.add_argument('--input_spec_size', default=128)
    parser.add_argument('--norm', default='min_max')
    parser.add_argument('--win_len', default=200)
    parser.add_argument('--shift', default=1)
    parser.add_argument('--pred', default='emotion')
    
    args = parser.parse_args()

    data_set_str, feature_type = args.dataset, args.feature_type
    win_len, feature_len = int(args.win_len), int(args.input_spec_size)
    root_path = Path('/media/data/projects/speech-privacy')
    
    if 'combine' not in data_set_str and data_set_str != 'msp-podcast':
        # get the cross validation sets
        speaker_id_arr = speaker_id_arr_dict[data_set_str]    
        train_array, validate_array, test_array = [], [], []
        
        kf = KFold(n_splits=5, random_state=None, shuffle=False)
        for other_index, test_index in kf.split(speaker_id_arr):
            
            # 60% are training, 20% are validation, and 20% are test
            tmp_arr = []
            if test_index[-1]+1 != len(speaker_id_arr):
                for i in range(test_index[-1]+1, len(speaker_id_arr)):
                    tmp_arr.append(speaker_id_arr[i])
            for i in range(0, test_index[0]):
                tmp_arr.append(speaker_id_arr[i])
            
            validate_len = int(np.round(len(tmp_arr) * 0.2))
            train_arr = tmp_arr[validate_len:]
            validate_arr = [tmp for tmp in tmp_arr if tmp not in train_arr]
    
            train_array.append(train_arr)
            validate_array.append(validate_arr)
            test_array.append(speaker_id_arr[test_index])

    # if we dont have data ready for experiments, preprocess them first
    fold_num = 1 if args.dataset == 'msp-podcast' else 5
    for i in range(5):
        cmd_str = 'python3 preprocess_federate_data.py --dataset ' + data_set_str
        
        if args.feature_type == 'mel_spec':
            cmd_str += ' --feature_len ' + str(feature_len)
            cmd_str += ' --win_len ' + str(win_len)
            cmd_str += ' --shift ' + args.shift
        
        cmd_str += ' --norm ' + args.norm
        cmd_str += ' --test_fold ' + 'fold' + str(i+1)
        cmd_str += ' --feature_type ' + feature_type
        cmd_str += ' --pred ' + args.pred

        # we dont have these speaker array when combining dataset
        if 'combine' not in data_set_str and data_set_str != 'msp-podcast':
            cmd_str += ' --train_arr '
            for train_idx in train_array[i]:
                cmd_str += str(train_idx) + ' '
            cmd_str += ' --validation_arr '
            for validate_idx in validate_array[i]:
                cmd_str += str(validate_idx) + ' '
            cmd_str += ' --test_arr '
            for test_idx in test_array[i]:
                cmd_str += str(test_idx) + ' '
        
        print(cmd_str)
        os.system(cmd_str)
