import torch
import torch.nn as nn
import argparse
import torch.multiprocessing
from copy import deepcopy
from torch.nn.modules import dropout
from torch.utils.data import DataLoader, dataset
from re import L

import numpy as np
import torch
import pickle
from pathlib import Path
import pandas as pd
import copy
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'model'))
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'utils'))

from training_tools import EarlyStopping, ReturnResultDict
from training_tools import setup_seed
from baseline_models import dnn_classifier
from update import LocalUpdate, average_weights, average_gradients, DatasetSplit

import pdb

# define label mapping
emo_dict = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
affect_dict = {'low': 0, 'med': 1, 'high': 2}
gender_dict = {'F': 0, 'M': 1}

# define speaker mapping
speaker_id_arr_dict = {'msp-improv': np.arange(0, 12, 1), 
                       'crema-d': np.arange(1001, 1092, 1),
                       'iemocap': np.arange(0, 10, 1)}

feature_len_dict = {'emobase': 988, 'ComParE': 6373, 'wav2vec': 9216, 
                    'apc': 512, 'distilhubert': 768, 'tera': 768, 'wav2vec2': 768,
                    'decoar2': 768, 'cpc': 256, 'audio_albert': 768, 
                    'mockingjay': 768, 'npc': 512, 'vq_apc': 512, 'vq_wav2vec': 512}

data_name_dict = {'emobase': 'Emo-Base', 'ComParE': 6373, 'wav2vec': 9216, 
                  'apc': 'APC', 'distilhubert': 'DistilHuBERT', 'tera': 'Tera', 'wav2vec2': 768,
                  'decoar2': 'DeCoAR 2.0', 'cpc': 256, 'audio_albert': 768, 
                  'mockingjay': 'Mockingjay', 'npc': 'NPC', 'vq_apc': 'Vq-APC', 'vq_wav2vec': 512}


if __name__ == '__main__':

    torch.cuda.empty_cache() 
    torch.multiprocessing.set_sharing_strategy('file_system')

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--feature_type', default='mel_spec')
    parser.add_argument('--dropout', default=0.2)
    parser.add_argument('--learning_rate', default=0.05)
    parser.add_argument('--batch_size', default=20)
    parser.add_argument('--num_epochs', default=200)
    parser.add_argument('--local_epochs', default=5)
    parser.add_argument('--norm', default='znorm')
    parser.add_argument('--model_type', default='fed_sgd')
    parser.add_argument('--pred', default='emotion')

    args = parser.parse_args()
    setup_seed(8)

    root_path = Path('/media/data/projects/speech-privacy')
    pred = 'affect' if args.pred == 'arousal' or args.pred == 'valence' else 'emotion'
    
    for feature_type in ['emobase', 'apc', 'vq_apc', 'tera', 'npc', 'decoar2', 'distilhubert']:
        print('\\textbf{%s} &' % data_name_dict[feature_type])
        for model_type in ['fed_sgd', 'fed_avg']:
            model_setting_str = 'local_epoch_'+str(args.local_epochs) if model_type == 'fed_avg' else 'local_epoch_1'
            model_setting_str += '_dropout_' + str(args.dropout).replace('.', '')
            lr = 0.0001 if model_type == 'fed_avg' else 0.05
            model_setting_str += '_lr_' + str(lr)[2:]
            for dataset_str in ['iemocap', 'crema-d', 'msp-improv']:
                model_result_csv_path = Path.cwd().parents[0].joinpath('results', args.pred, model_type, feature_type, model_setting_str)
                save_result_df = pd.read_csv(str(model_result_csv_path.joinpath('private_'+ str(dataset_str) + '.csv')), index_col=0)
                acc = float(save_result_df.loc['average', 'acc']) * 100
                rec = float(save_result_df.loc['average', 'rec']) * 100
                if model_type == 'fed_avg' and dataset_str == 'msp-improv':
                    print('%.2f\\%% & ' % (acc))
                    print('%.2f\\%% ' % (rec))
                    print('\\rule{0pt}{2.25ex} \\\\')
                    print()
                else:
                    print('%.2f\\%% & ' % (acc))
                    print('%.2f\\%% & ' % (rec))
                
            