import argparse
import numpy as np
import os, pdb
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


