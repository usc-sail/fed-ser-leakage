from pathlib import Path
import pandas as pd
import argparse

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'model'))
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'utils'))

import pdb

# define feature name mapping
data_name_dict = {'emobase': 'Emo-Base', 'apc': 'APC', 'distilhubert': 'DistilHuBERT', 
                  'tera': 'Tera', 'decoar2': 'DeCoAR 2.0', 'cpc': 256, 'audio_albert': 768, 
                  'mockingjay': 'Mockingjay', 'npc': 'NPC', 'vq_apc': 'Vq-APC', 'vq_wav2vec': 512}

attack_name_dict = {'iemocap': 'MSP-Improv + CREMA-D', 'crema-d': 'IEMOCAP + MSP-Improv', 'msp-improv': 'IEMOCAP + CREMA-D'}
private_name_dict = {'iemocap': 'IEMOCAP', 'crema-d': 'CREMA-D', 'msp-improv': 'MSP-Improv'}


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
    parser.add_argument('--leak_layer', default='first')
    parser.add_argument('--norm', default='znorm')
    parser.add_argument('--model_type', default='fed_sgd')
    parser.add_argument('--pred', default='emotion')

    args = parser.parse_args()
    root_path = Path('/media/data/projects/speech-privacy')
    pred = 'affect' if args.pred == 'arousal' or args.pred == 'valence' else 'emotion'
    
    feature_list = ['emobase', 'apc', 'vq_apc', 'npc', 'decoar2', 'tera']
    for dataset_str in ['iemocap', 'crema-d', 'msp-improv']:

        print('\\multirow{%d}{*}{\\textbf{%s}} & ' % (len(feature_list), private_name_dict[dataset_str]))
        print('&')
        
        for feature_type in feature_list:
            
            if feature_type == feature_list[2]:
                print('&')
                print('\\textbf{%s} &' % (attack_name_dict[dataset_str].split(' + ')[0]))
            elif feature_type == feature_list[3]:
                print('&')
                print('\\textbf{%s} &' % (attack_name_dict[dataset_str].split(' + ')[1]))
            elif feature_type != feature_list[0]:
                print('&')
                print('&')
            
            print('\\textbf{%s} & ' % data_name_dict[feature_type])

            # for local_epochs in [1, 5]:
            local_epochs = 1
            for dropout in [0.2, 0.4, 0.6]:

                model_setting_str = 'local_epoch_'+str(local_epochs)
                model_setting_str += '_dropout_' + str(dropout).replace('.', '')
                lr = 0.0005
                model_setting_str += '_lr_' + str(lr)[2:]
                
                model_result_csv_path = Path.cwd().parents[0].joinpath('results', args.pred, 'fed_avg', feature_type, model_setting_str)
                ser_result_df = pd.read_csv(str(model_result_csv_path.joinpath('private_'+ str(dataset_str) + '.csv')), index_col=0)
                if 'rec' in list(ser_result_df.columns):
                    ser_uar = float(ser_result_df.loc['average', 'rec']) * 100
                else:
                    ser_uar = float(ser_result_df.loc['average', 'uar']) * 100
                print('%.2f\\%% & ' % (ser_uar))
                

            for dropout in [0.2, 0.4, 0.6]:

                model_setting_str = 'local_epoch_'+str(local_epochs)
                model_setting_str += '_dropout_' + str(dropout).replace('.', '')
                lr = 0.0005
                model_setting_str += '_lr_' + str(lr)[2:]
                
                attack_model_result_csv_path = Path.cwd().parents[0].joinpath('results', 'attack', 'first', 'fed_avg', feature_type, model_setting_str)
                save_result_df = pd.read_csv(str(attack_model_result_csv_path.joinpath('private_'+ str(dataset_str) + '_result.csv')), index_col=0)
                uar = float(save_result_df.loc['average', 'uar']) * 100
                if dropout == 0.6:
                    print('%.2f\\%%' % (uar))
                    print('\\rule{0pt}{2.25ex} \\\\')
                    print()
                else:
                    print('%.2f\\%% & ' % (uar))
        if dataset_str == 'msp-improv':
            print('\\bottomrule')
        else:
            print('\\midrule')




    '''
    feature_list = ['emobase', 'apc', 'vq_apc', 'npc', 'decoar2', 'tera']
    for dataset_str in ['iemocap', 'crema-d', 'msp-improv']:

        print('\\multirow{%d}{*}{\\textbf{%s}} & ' % (len(feature_list), private_name_dict[dataset_str]))
        # print('\\multirow{%d}{*}{\\textbf{%s}} & ' % (len(feature_list), attack_name_dict[dataset_str]))
        print('&')

        for feature_type in feature_list:
            
            if feature_type == feature_list[2]:
                print('&')
                print('\\textbf{%s} &' % (attack_name_dict[dataset_str].split(' + ')[0]))
            elif feature_type == feature_list[3]:
                print('&')
                print('\\textbf{%s} &' % (attack_name_dict[dataset_str].split(' + ')[1]))
            elif feature_type != feature_list[0]:
                print('&')
                print('&')
            
            print('\\textbf{%s} & ' % data_name_dict[feature_type])
            for model_type in ['fed_sgd', 'fed_avg']:
                for layer in ['first', 'second', 'last']:

                    model_setting_str = 'local_epoch_1_dropout_02_lr_05' if model_type == 'fed_sgd' else 'local_epoch_1_dropout_02_lr_0005'
                    attack_model_result_csv_path = Path.cwd().parents[0].joinpath('results', 'attack', layer, model_type, feature_type, model_setting_str)
                    save_result_df = pd.read_csv(str(attack_model_result_csv_path.joinpath('private_'+ str(dataset_str) + '_result.csv')), index_col=0)
                    acc = float(save_result_df.loc['average', 'acc']) * 100
                    uar = float(save_result_df.loc['average', 'uar']) * 100
                    if layer == 'last' and model_type == 'fed_avg':
                        # print('%.2f\\%% & ' % (acc))
                        print('%.2f\\%%' % (uar))
                        print('\\rule{0pt}{2.25ex} \\\\')
                        # if feature_type != feature_list[-1]:
                            # print('\\cline{3-7}')
                        print()
                    else:
                        # print('%.2f\\%% & ' % (acc))
                        print('%.2f\\%% & ' % (uar))
            
        if dataset_str == 'msp-improv':
            print('\\bottomrule')
        else:
            print('\\midrule')
    '''
    

    '''
    feature_list = ['emobase', 'apc', 'vq_apc', 'npc', 'decoar2', 'tera']
    for dataset_str in ['iemocap', 'crema-d', 'msp-improv']:

        print('\\multirow{%d}{*}{\\textbf{%s}} & ' % (len(feature_list), private_name_dict[dataset_str]))
        print('&')
        # print('\\multirow{%d}{*}{\\textbf{%s}} & ' % (len(feature_list), attack_name_dict[dataset_str]))
        
        for feature_type in feature_list:
            if feature_type == feature_list[2]:
                print('&')
                print('\\textbf{%s} &' % (attack_name_dict[dataset_str].split(' + ')[0]))
            elif feature_type == feature_list[3]:
                print('&')
                print('\\textbf{%s} &' % (attack_name_dict[dataset_str].split(' + ')[1]))
            elif feature_type != feature_list[0]:
                print('&')
                print('&')
            
            print('\\textbf{%s} & ' % data_name_dict[feature_type])

            for model_type in ['fed_sgd', 'fed_avg']:

                model_setting_str = 'local_epoch_1'
                model_setting_str += '_dropout_' + str(args.dropout).replace('.', '')
                lr = 0.0005 if model_type == 'fed_avg' else 0.05
                model_setting_str += '_lr_' + str(lr)[2:]

                attack_model_result_csv_path = Path.cwd().parents[0].joinpath('results', 'attack', 'fusion', model_type, feature_type, model_setting_str)
                save_result_df = pd.read_csv(str(attack_model_result_csv_path.joinpath('private_'+ str(dataset_str) + '_result.csv')), index_col=0)
                acc = float(save_result_df.loc['average', 'acc']) * 100
                uar = float(save_result_df.loc['average', 'uar']) * 100
                if model_type == 'fed_avg':
                    # print('%.2f\\%% & ' % (acc))
                    print('%.2f\\%%' % (uar))
                    print('\\rule{0pt}{2.25ex} \\\\')
                    # if feature_type != feature_list[-1]:
                        # print('\\cline{3-7}')
                    print()
                else:
                    # print('%.2f\\%% & ' % (acc))
                    print('%.2f\\%% & ' % (uar))
        if dataset_str == 'msp-improv':
            print('\\bottomrule')
        else:
            print('\\midrule')
    '''


    '''
    for feature_type in ['emobase', 'apc', 'vq_apc', 'tera', 'npc', 'decoar2']:
        print('\\multirow{3}{*}{\\textbf{%s}} & ' % data_name_dict[feature_type])
        for dataset_str in ['iemocap', 'crema-d', 'msp-improv']:
            if dataset_str != 'iemocap':
                print('&')
            
            print('\\textbf{%s} & ' % private_name_dict[dataset_str])
            print('\\textbf{%s} & ' % attack_name_dict[dataset_str])

            for model_type in ['fed_sgd', 'fed_avg']:

                model_setting_str = 'local_epoch_'+str(args.local_epochs) if model_type == 'fed_avg' else 'local_epoch_1'
                model_setting_str += '_dropout_' + str(args.dropout).replace('.', '')
                lr = 0.0001 if model_type == 'fed_avg' else 0.05
                model_setting_str += '_lr_' + str(lr)[2:]

                attack_model_result_csv_path = Path.cwd().parents[0].joinpath('results', 'attack', args.leak_layer, model_type, feature_type, model_setting_str)
                save_result_df = pd.read_csv(str(attack_model_result_csv_path.joinpath('private_'+ str(dataset_str) + '_result.csv')), index_col=0)
                acc = float(save_result_df.loc['average', 'acc']) * 100
                rec = float(save_result_df.loc['average', 'uar']) * 100
                if model_type == 'fed_avg':
                    print('%.2f\\%% & ' % (acc))
                    print('%.2f\\%% ' % (rec))
                    print('\\rule{0pt}{2.25ex} \\\\')
                    if dataset_str != 'msp-improv':
                        print('\\cline{2-7}')
                    print()
                else:
                    print('%.2f\\%% & ' % (acc))
                    print('%.2f\\%% & ' % (rec))
        print('\\hline')
    '''     
            