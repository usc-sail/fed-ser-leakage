from pathlib import Path
import pandas as pd
import argparse

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'model'))
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'utils'))

import pdb

# define feature name mapping
data_name_dict = {'emobase': 'Emo-Base', 'ComParE': 6373, 'wav2vec': 9216, 
                  'apc': 'APC', 'distilhubert': 'DistilHuBERT', 'tera': 'Tera', 'wav2vec2': 768,
                  'decoar2': 'DeCoAR 2.0', 'cpc': 256, 'audio_albert': 768, 
                  'mockingjay': 'Mockingjay', 'npc': 'NPC', 'vq_apc': 'Vq-APC', 'vq_wav2vec': 512}


if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--feature_type', default='mel_spec')
    parser.add_argument('--dropout', default=0.2)
    parser.add_argument('--learning_rate', default=0.05)
    parser.add_argument('--batch_size', default=20)
    parser.add_argument('--num_epochs', default=200)
    parser.add_argument('--local_epochs', default=1)
    parser.add_argument('--norm', default='znorm')
    parser.add_argument('--model_type', default='fed_sgd')
    parser.add_argument('--pred', default='emotion')

    args = parser.parse_args()
    
    # 1. results of the general fed results
    for feature_type in ['emobase', 'apc', 'vq_apc', 'npc', 'decoar2', 'tera']:
        print('\\textbf{%s} &' % data_name_dict[feature_type])
        for model_type in ['fed_sgd', 'fed_avg']:
            model_setting_str = 'local_epoch_'+str(args.local_epochs) if model_type == 'fed_avg' else 'local_epoch_1'
            model_setting_str += '_dropout_' + str(args.dropout).replace('.', '')
            lr = 0.0005 if model_type == 'fed_avg' else 0.05
            model_setting_str += '_lr_' + str(lr)[2:]
            for dataset_str in ['iemocap', 'crema-d', 'msp-improv']:
                model_result_csv_path = Path.cwd().parents[0].joinpath('results', args.pred, model_type, feature_type, model_setting_str)
                save_result_df = pd.read_csv(str(model_result_csv_path.joinpath('private_'+ str(dataset_str) + '.csv')), index_col=0)
                acc = float(save_result_df.loc['average', 'acc']) * 100
                if 'rec' in list(save_result_df.columns):
                    rec = float(save_result_df.loc['average', 'rec']) * 100
                else:
                    rec = float(save_result_df.loc['average', 'uar']) * 100
                    
                if model_type == 'fed_avg' and dataset_str == 'msp-improv':
                    print('%.2f\\%% & ' % (acc))
                    print('%.2f\\%% ' % (rec))
                    print('\\rule{0pt}{2.25ex} \\\\')
                    print()
                else:
                    print('%.2f\\%% & ' % (acc))
                    print('%.2f\\%% & ' % (rec))
    
    # 2. results in readme
    for feature_type in ['emobase', 'apc', 'vq_apc', 'npc', 'decoar2', 'tera']:
        print('\\textbf{%s} &' % data_name_dict[feature_type])
        for local_epoch in [1, 5]:
            lr = 0.0005 if local_epoch == 1 else 0.0001
            
            model_setting_str = 'local_epoch_'+str(local_epoch)
            model_setting_str += '_dropout_' + str(args.dropout).replace('.', '')
            model_setting_str += '_lr_' + str(lr)[2:]
            
            for dataset_str in ['iemocap', 'crema-d', 'msp-improv']:
                model_result_csv_path = Path.cwd().parents[0].joinpath('results', args.pred, model_type, feature_type, model_setting_str)
                save_result_df = pd.read_csv(str(model_result_csv_path.joinpath('private_'+ str(dataset_str) + '.csv')), index_col=0)
                acc = float(save_result_df.loc['average', 'acc']) * 100
                if 'rec' in list(save_result_df.columns):
                    rec = float(save_result_df.loc['average', 'rec']) * 100
                else:
                    rec = float(save_result_df.loc['average', 'uar']) * 100
                    
                if model_type == 'fed_avg' and dataset_str == 'msp-improv':
                    print('%.2f\\%% & ' % (acc))
                    print('%.2f\\%% ' % (rec))
                    print('\\rule{0pt}{2.25ex} \\\\')
                    print()
                else:
                    print('%.2f\\%% & ' % (acc))
                    print('%.2f\\%% & ' % (rec))
            