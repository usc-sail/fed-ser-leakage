import numpy as np
import os, pdb, sys
from pathlib import Path
import configparser


if __name__ == '__main__':

    # read config files
    config = configparser.ConfigParser()
    config.sections()
    config.read('config.ini')

    # 1. feature processing
    if config['mode'].getboolean('process_feature') is True:
        for dataset in ['iemocap', 'iemocap', 'crema-d']:
            if config['feature']['feature'] == 'emobase':
                cmd_str = 'taskset 100 python3 feature_extraction/opensmile_feature_extraction.py --dataset ' + dataset
            else:
                cmd_str = 'taskset 100 python3 feature_extraction/pretrained_audio_feature_extraction.py --dataset ' + dataset
            cmd_str += ' --feature_type ' + config['feature']['feature']
            cmd_str += ' --data_dir ' + config['dir'][dataset]
            cmd_str += ' --save_dir ' + config['dir']['save_dir']
            
            print('Extract features')
            print(cmd_str)
            pdb.set_trace()
            os.system(cmd_str)
    
    # 2. process training data
    if config['mode'].getboolean('process_training') is True:
        for dataset in ['msp-improv', 'iemocap', 'crema-d']:
            cmd_str = 'taskset 100 python3 preprocess_data/preprocess_federate_data.py --dataset ' + dataset
            cmd_str += ' --feature_type ' + config['feature']['feature']
            cmd_str += ' --data_dir ' + config['dir'][dataset]
            cmd_str += ' --save_dir ' + config['dir']['save_dir']
            cmd_str += ' --norm znorm'

            print('Process training data')
            print(cmd_str)
            os.system(cmd_str)

    # 3.1 Training SER model
    if config['mode'].getboolean('ser_training') is True:
        # for dataset in [config['dataset']['private_dataset'], config['dataset']['adv_dataset']]:
        # for dataset in [config['dataset']['adv_dataset']]:
        # for dataset in ['iemocap', 'crema-d', 'msp-improv', 'iemocap_crema-d', 'iemocap_msp-improv', 'msp-improv_crema-d']:
        # for dataset in ['iemocap']:
        # for dataset in ['msp-improv', 'iemocap', 'crema-d']:
        for dataset in ['iemocap']:
        # for dataset in ['crema-d']:
        # for dataset in ['msp-improv']:
        # for dataset in ['iemocap_crema-d', 'iemocap_msp-improv', 'msp-improv_crema-d']:
        # for dataset in ['iemocap_msp-improv']:
        # for dataset in ['iemocap_crema-d']:
            # for feature in ['tera', 'decoar2', 'npc']:
            # for feature in ['emobase', 'apc', 'vq_apc']:
            # for feature in ['emobase', 'apc', 'vq_apc', 'tera', 'decoar2']:
            for feature in ['emobase']:
            # for feature in ['tera']:
            # for feature in ['decoar2']:
            # for feature in ['apc']:
            # for feature in ['vq_apc']:
                if config['model'].getboolean('udp'):
                    cmd_str = 'taskset 300 python3 train/federated_ser_classifier_udp.py --dataset ' + dataset
                    cmd_str += ' --privacy_budget ' + config['model']['privacy_budget']
                else:
                    cmd_str = 'taskset 300 python3 train/federated_ser_classifier.py --dataset ' + dataset
                # cmd_str += ' --feature_type ' + config['feature']['feature']
                cmd_str += ' --feature_type ' + feature
                cmd_str += ' --dropout ' + config['model']['dropout']
                cmd_str += ' --norm znorm --optimizer adam'
                cmd_str += ' --model_type ' + config['model']['fed_model']
                cmd_str += ' --learning_rate ' + config[config['model']['fed_model']]['lr']
                cmd_str += ' --local_epochs ' + config[config['model']['fed_model']]['local_epochs']
                cmd_str += ' --num_epochs ' + config[config['model']['fed_model']]['global_epochs']
                cmd_str += ' --save_dir ' + config['dir']['save_dir']
                
                print('Traing SER model')
                print(cmd_str)
                # pdb.set_trace()
                os.system(cmd_str)

    # 4. Training attack model
    if config['mode'].getboolean('attack_training') is True:
        
        for dataset_list in [['iemocap', 'msp-improv_crema-d'], 
                             ['crema-d', 'iemocap_msp-improv'], 
                             ['msp-improv', 'iemocap_crema-d']]:
        # for dataset_list in [['msp-improv', 'iemocap_crema-d']]:
        # for dataset_list in [['crema-d', 'iemocap_msp-improv']]:
        # for dataset_list in [['iemocap', 'msp-improv_crema-d'],
        #                     ['msp-improv', 'iemocap_crema-d']]:
            # for feature in ['emobase', 'apc', 'vq_apc', 'tera', 'decoar2', 'npc']:
            for feature in ['tera', 'decoar2', 'emobase', 'apc', 'vq_apc']:
            # for feature in ['emobase']:
            # for feature in ['vq_apc']:
            # for feature in ['tera', 'decoar2', 'npc']:
            # for feature in ['emobase', 'apc', 'vq_apc']:
                if config['model'].getint('attack_sample') == 1:
                    cmd_str = 'taskset 500 python3 train/federated_attribute_attack.py'
                else:
                    cmd_str = 'taskset 500 python3 train/federated_attribute_attack_multiple.py --num_sample ' + config['model']['attack_sample']
                cmd_str += ' --norm znorm --optimizer adam'
                # cmd_str += ' --dataset ' + config['dataset']['private_dataset']
                # cmd_str += ' --adv_dataset ' + config['dataset']['adv_dataset']
                # cmd_str += ' --feature_type ' + config['feature']['feature']

                cmd_str += ' --dataset ' + dataset_list[0]
                cmd_str += ' --adv_dataset ' + dataset_list[1]
                cmd_str += ' --feature_type ' + feature
                
                cmd_str += ' --dropout ' + config['model']['dropout']
                cmd_str += ' --model_type ' + config['model']['fed_model']
                cmd_str += ' --learning_rate ' + config[config['model']['fed_model']]['lr']
                cmd_str += ' --local_epochs ' + config[config['model']['fed_model']]['local_epochs']
                cmd_str += ' --num_epochs ' + config[config['model']['fed_model']]['global_epochs']
                cmd_str += ' --leak_layer first --model_learning_rate 0.0001'
                # cmd_str += ' --leak_layer first --model_learning_rate 0.0001'
                cmd_str += ' --device 0'
                cmd_str += ' --save_dir ' + config['dir']['save_dir']

                # if config['model'].getboolean('udp'):
                #    cmd_str += ' --privacy_budget ' + config['model']['privacy_budget']
                
                print('Traing Attack model')
                print(cmd_str)
                # pdb.set_trace()
                os.system(cmd_str)

    # 5.1 Loading attack model
    if config['mode'].getboolean('attack_result') is True:
        for dataset_list in [['iemocap', 'msp-improv_crema-d'], 
                             ['crema-d', 'iemocap_msp-improv'], 
                             ['msp-improv', 'iemocap_crema-d']]:
        # for dataset_list in [['crema-d', 'iemocap_msp-improv']]:
        # for dataset_list in [['msp-improv', 'iemocap_crema-d']]:
        # for dataset_list in [['iemocap', 'msp-improv_crema-d']]:
        # for dataset_list in [['msp-improv', 'iemocap_crema-d']]:
            for feature in ['tera', 'decoar2', 'emobase', 'apc', 'vq_apc']:
                for privacy_budget in [0, 5, 10, 25, 50]:
                # for privacy_budget in [5]:
                    cmd_str = 'taskset 500 python3 train/federated_attribute_attack_result_per_speaker.py'
                    cmd_str += ' --norm znorm'
                    cmd_str += ' --dataset ' + dataset_list[0]
                    cmd_str += ' --adv_dataset ' + dataset_list[1]
                    cmd_str += ' --feature_type ' + feature
                    cmd_str += ' --dropout ' + config['model']['dropout']
                    cmd_str += ' --model_type ' + config['model']['fed_model']
                    cmd_str += ' --learning_rate ' + config[config['model']['fed_model']]['lr']
                    cmd_str += ' --local_epochs ' + config[config['model']['fed_model']]['local_epochs']
                    cmd_str += ' --num_epochs ' + config[config['model']['fed_model']]['global_epochs']
                    cmd_str += ' --leak_layer first --device 0'
                    cmd_str += ' --save_dir ' + config['dir']['save_dir']
                    cmd_str += ' --num_sample ' + config['model']['num_sample']

                    # if config['model'].getboolean('udp'):
                    if privacy_budget is not 0:
                        cmd_str += ' --privacy_budget ' + str(privacy_budget)
                    
                    print('Attack model result')
                    print(cmd_str)
                    # pdb.set_trace()
                    os.system(cmd_str)
    
    # 5.2 Loading attack model and finetune
    if config['mode'].getboolean('attack_result_finetune') is True:
        # for dataset_list in [['iemocap', 'msp-improv_crema-d'], 
        #                      ['crema-d', 'iemocap_msp-improv'], 
        #                     ['msp-improv', 'iemocap_crema-d']]:
        # for dataset_list in [['crema-d', 'iemocap_msp-improv']]:
        # for dataset_list in [['msp-improv', 'iemocap_crema-d']]:
        for dataset_list in [['iemocap', 'msp-improv_crema-d']]:
        # for dataset_list in [['msp-improv', 'iemocap_crema-d']]:
            for feature in ['tera', 'decoar2', 'emobase', 'apc', 'vq_apc']:
            # for feature in ['tera']:
                for privacy_budget in [10, 25, 50, 5]:
                # for privacy_budget in [5]:
                    cmd_str = 'taskset 500 python3 train/federated_attribute_attack_result_finetune.py'
                    cmd_str += ' --norm znorm'
                    cmd_str += ' --dataset ' + dataset_list[0]
                    cmd_str += ' --adv_dataset ' + dataset_list[1]
                    cmd_str += ' --feature_type ' + feature
                    cmd_str += ' --dropout ' + config['model']['dropout']
                    cmd_str += ' --model_type ' + config['model']['fed_model']
                    cmd_str += ' --learning_rate ' + config[config['model']['fed_model']]['lr']
                    cmd_str += ' --local_epochs ' + config[config['model']['fed_model']]['local_epochs']
                    cmd_str += ' --num_epochs ' + config[config['model']['fed_model']]['global_epochs']
                    cmd_str += ' --num_sample ' + config['model']['num_sample']
                    
                    cmd_str += ' --leak_layer first --device 0'
                    cmd_str += ' --save_dir ' + config['dir']['save_dir']
                    
                    # if config['model'].getboolean('udp'):
                    if privacy_budget is not 0:
                        cmd_str += ' --privacy_budget ' + str(privacy_budget)
                    
                    print('Attack model result')
                    print(cmd_str)
                    # pdb.set_trace()
                    os.system(cmd_str)
