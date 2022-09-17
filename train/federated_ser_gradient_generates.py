from moviepy.tools import verbose_print
import torch
import torch.nn as nn
import argparse, logging
import torch.multiprocessing
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

import numpy as np
from pathlib import Path
import pandas as pd
import copy, time, pickle, shutil, sys, os, pdb
from copy import deepcopy

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model'))

from dnn_models import dnn_classifier
from update import average_weights, average_gradients, local_trainer


# define label mapping
emo_dict = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
affect_dict = {'low': 0, 'med': 1, 'high': 2}
gender_dict = {'F': 0, 'M': 1}
                       
# define feature len mapping
feature_len_dict = {'emobase': 988, 'ComParE': 6373, 'wav2vec': 9216, 
                    'apc': 512, 'distilhubert': 768, 'tera': 768, 'wav2vec2': 768,
                    'decoar2': 768, 'cpc': 256, 'audio_albert': 768, 
                    'mockingjay': 768, 'npc': 512, 'vq_apc': 512, 'vq_wav2vec': 512}

def save_result(save_index, acc, uar, best_epoch, dataset):
    row_df = pd.DataFrame(index=[save_index])
    row_df['acc'], row_df['uar'], row_df['epoch'], row_df['dataset']  = acc, uar, best_epoch, dataset
    return row_df


class DatasetGenerator():
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset['dataset'])

    def __getitem__(self, item):
        data = self.dataset['data'][item]
        label = self.dataset['label'][item]
        dataset_str = self.dataset['dataset'][item]
        return torch.tensor(data), torch.tensor(int(label)), dataset_str

def read_data_dict_by_client(dataset_list, fold_idx):
    
    return_train_dict, return_val_dict, return_test_dict = {}, {}, {}
    dataset_label_list = []
    # prepare the data for the training
    for dataset in dataset_list:
        with open(preprocess_path.joinpath(dataset, 'fold'+str(int(fold_idx+1)), 'training_'+args.norm+'.pkl'), 'rb') as f:
            train_dict = pickle.load(f)
        with open(preprocess_path.joinpath(dataset, 'fold'+str(int(fold_idx+1)), 'test_'+args.norm+'.pkl'), 'rb') as f:
            test_dict = pickle.load(f)
        for tmp_dict in [train_dict, test_dict]:
            for key in tmp_dict: tmp_dict[key]['dataset'] = dataset

        # test set will be the same
        x_test, y_test = np.zeros([len(test_dict), feature_len_dict[args.feature_type]]), np.zeros([len(test_dict)])
        for key_idx, key in enumerate(list(test_dict.keys())): 
            x_test[key_idx], y_test[key_idx] = test_dict[key]['data'], int(emo_dict[test_dict[key]['label']])
            dataset_label_list.append(test_dict[key]['dataset'])
        
        if len(return_test_dict) == 0:
            return_test_dict['data'], return_test_dict['label'] = x_test, y_test
            return_test_dict['dataset'] = dataset_label_list
        else:
            return_test_dict['data'] = np.append(return_test_dict['data'], x_test, axis=0)
            return_test_dict['label'] = np.append(return_test_dict['label'], y_test, axis=0)
            return_test_dict['dataset'] = dataset_label_list

        # we remake the data dict per speaker for the ease of local training
        train_speaker_data_dict = {}
        for key in train_dict:
            speaker_id = str(train_dict[key]['speaker_id'])
            if speaker_id not in train_speaker_data_dict: train_speaker_data_dict[speaker_id] = []
            train_speaker_data_dict[speaker_id].append(key)
            
        # in federated setting that the norm validation dict will be based on local client data
        # so we combine train_dict and validate_dict in centralized setting
        # then choose certain amount data per client as local validation set
        if dataset == 'crema-d':
            for speaker_id in train_speaker_data_dict:
                speaker_data_key_list = train_speaker_data_dict[speaker_id]
                x, y = np.zeros([len(speaker_data_key_list), feature_len_dict[args.feature_type]]), np.zeros([len(speaker_data_key_list)])
                dataset_list = []
                
                for idx, data_key in enumerate(speaker_data_key_list):
                    x[idx], y[idx] = train_dict[data_key]['data'], int(emo_dict[train_dict[data_key]['label']])
                    dataset_list.append(train_dict[data_key]['dataset'])
                
                idx_array = np.random.permutation(len(train_speaker_data_dict[speaker_id]))
                perm_array = np.random.permutation(len(idx_array))
                return_train_dict[speaker_id] = {}
                return_train_dict[speaker_id]['data'] = x[perm_array[:int(0.8*len(x))]].copy()
                return_train_dict[speaker_id]['label'] = y[perm_array[:int(0.8*len(x))]].copy()
                return_train_dict[speaker_id]['gender'] = train_dict[data_key]['gender']
                return_train_dict[speaker_id]['dataset'] = [dataset_list[idx] for idx in perm_array[:int(0.8*len(x))]]

                # We save the utterance keys used for training and validation per speaker (client)
                return_val_dict[speaker_id] = {}
                return_val_dict[speaker_id]['data'] = x[perm_array[int(0.8*len(x)):]].copy()
                return_val_dict[speaker_id]['label'] = y[perm_array[int(0.8*len(x)):]].copy()
                return_val_dict[speaker_id]['gender'] = train_dict[data_key]['gender']
                return_val_dict[speaker_id]['dataset'] = [dataset_list[idx] for idx in perm_array[int(0.8*len(x)):]]
        else:
            # we want to divide speaker data if the dataset is iemocap or msp-improv to increase client size
            for speaker_id in train_speaker_data_dict:
                # in iemocap and msp-improv
                # we spilit each speaker data into 10 parts in order to create more clients
                idx_array = np.random.permutation(len(train_speaker_data_dict[speaker_id]))
                speaker_data_key_list = train_speaker_data_dict[speaker_id]
                split_array = np.array_split(idx_array, 10)
                for split_idx in range(len(split_array)):
                    # we randomly pick 10% of data
                    idxs_train = split_array[split_idx]

                    x, y = np.zeros([len(idxs_train), feature_len_dict[args.feature_type]]), np.zeros([len(idxs_train)])
                    dataset_list = []
                    for idx, key_idx in enumerate(idxs_train):
                        data_key = speaker_data_key_list[key_idx]
                        x[idx], y[idx] = train_dict[data_key]['data'], int(emo_dict[train_dict[data_key]['label']])
                        dataset_list.append(train_dict[data_key]['dataset'])
                        # if speaker_id+'_'+str(split_idx) not in return_train_dict: return_train_dict[speaker_id+'_'+str(split_idx)] = {}
                        # return_train_dict[speaker_id+'_'+str(split_idx)][key] = train_dict[key].copy()
                    
                    perm_array = np.random.permutation(len(idxs_train))
                    return_train_dict[speaker_id+'_'+str(split_idx)] = {}
                    return_train_dict[speaker_id+'_'+str(split_idx)]['data'] = x[perm_array[:int(0.8*len(x))]].copy()
                    return_train_dict[speaker_id+'_'+str(split_idx)]['label'] = y[perm_array[:int(0.8*len(x))]].copy()
                    return_train_dict[speaker_id+'_'+str(split_idx)]['gender'] = train_dict[data_key]['gender']
                    return_train_dict[speaker_id+'_'+str(split_idx)]['dataset'] = [dataset_list[idx] for idx in perm_array[:int(0.8*len(x))]]

                    # We save the utterance keys used for training and validation per speaker (client)
                    return_val_dict[speaker_id+'_'+str(split_idx)] = {}
                    return_val_dict[speaker_id+'_'+str(split_idx)]['data'] = x[perm_array[int(0.8*len(x)):]].copy()
                    return_val_dict[speaker_id+'_'+str(split_idx)]['label'] = y[perm_array[int(0.8*len(x)):]].copy()
                    return_val_dict[speaker_id+'_'+str(split_idx)]['gender'] = train_dict[data_key]['gender']
                    return_val_dict[speaker_id+'_'+str(split_idx)]['dataset'] = [dataset_list[idx] for idx in perm_array[int(0.8*len(x)):]]
                    
    return return_train_dict, return_val_dict, return_test_dict

if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--adv_dataset', default='iemocap')
    parser.add_argument('--feature_type', default='emobase')
    parser.add_argument('--dropout', default=0.2)
    parser.add_argument('--learning_rate', default=0.05)
    parser.add_argument('--batch_size', default=20)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--num_epochs', default=200)
    parser.add_argument('--local_epochs', default=1)
    parser.add_argument('--norm', default='znorm')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--model_type', default='fed_sgd')
    parser.add_argument('--privacy_budget', default=10, type=int)
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--epoch', nargs='+', type=int)
    parser.add_argument('--save_dir', default='/media/data/projects/speech-privacy')
    args = parser.parse_args()

    preprocess_path = Path(args.save_dir).joinpath('federated_learning', args.feature_type, args.pred)
    
    # set seeds
    seed_everything(8, workers=True)
    save_result_df = pd.DataFrame()
    dataset_list = args.adv_dataset.split('_')

    # find device
    device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    # we are not interested in the training here
    for fold_idx in range(1):
        # save folder details
        save_row_str = 'fold'+str(int(fold_idx+1))
        row_df = pd.DataFrame(index=[save_row_str])
        
        model_setting_str = 'local_epoch_'+str(args.local_epochs) if args.model_type == 'fed_avg' else 'local_epoch_1'
        model_setting_str += '_dropout_' + str(args.dropout).replace('.', '')
        model_setting_str += '_lr_' + str(args.learning_rate)[2:]
        model_setting_str += '_udp_' + str(args.privacy_budget)
        
        # Read the data per speaker
        train_speaker_dict, val_speaker_dict, test_speaker_dict = read_data_dict_by_client(dataset_list, 0)
        num_of_speakers, speaker_list = len(train_speaker_dict), list(set(train_speaker_dict.keys()))
        
        # Define the model
        global_model = dnn_classifier(pred='emotion', input_spec=feature_len_dict[args.feature_type], dropout=float(args.dropout))
        global_model = global_model.to(device)
        global_weights = global_model.state_dict()

        # copy weights
        criterion = nn.NLLLoss().to(device)
        
        # log saving path
        model_result_path = Path(args.save_dir).joinpath('federated_model_params_fixed_global', args.model_type, args.pred, args.feature_type, args.adv_dataset, model_setting_str, save_row_str)
        udp_model_result_path = Path(args.save_dir).joinpath('federated_model_params', args.model_type, args.pred, args.feature_type, args.dataset, model_setting_str, save_row_str)
        Path.mkdir(model_result_path, parents=True, exist_ok=True)
        
        # test loader
        dataset_test = DatasetGenerator(test_speaker_dict)
        test_dataloaders = DataLoader(dataset_test, batch_size=20, num_workers=0, shuffle=False)

        # Training steps
        result_dict, best_score = {}, 0
        for epoch in args.epoch:
        
            gradient_hist_dict = {}
            epoch_model_path = str(udp_model_result_path.joinpath('global_hist_'+str(epoch)+'.pt'))
            global_model.load_state_dict(torch.load(epoch_model_path, map_location=device))
            global_model = global_model.to(device)
            
            # 1. Local training, return weights in fed_avg, return gradients in fed_sgd
            for idx in range(num_of_speakers):
                speaker_id = speaker_list[idx]
                
                # 1.1 Local training
                dataset_train = DatasetGenerator(train_speaker_dict[speaker_id])
                train_dataloaders = DataLoader(dataset_train, batch_size=20, num_workers=0, shuffle=True)
                trainer = local_trainer(args, device, criterion, args.model_type, train_dataloaders)
            
                # read shared updates: parameters in fed_avg and gradients for fed_sgd
                if args.model_type == 'fed_avg':
                    local_update, train_result = trainer.update_weights(model=copy.deepcopy(global_model))
                else:
                    local_update, train_result = trainer.update_gradients(model=copy.deepcopy(global_model))
                print('epoch %d, speaker id %s sample size %d, privacy_budget %.3f' % (epoch, speaker_id, train_result['num_samples'], float(args.privacy_budget)))
                
                # 1.2 calculate and save the raw gradients or pseudo gradients
                gradients = []
                if args.model_type == 'fed_avg':
                    # 'fake' gradients saving code
                    # iterate all layers in the classifier model
                    original_model = copy.deepcopy(global_model).state_dict()

                    # calculate how many updates per local epoch 
                    local_update_per_epoch = int(train_result['num_samples'] / int(args.batch_size)) + 1
                    
                    for key in original_model:
                        original_params = original_model[key].detach().clone().cpu().numpy()
                        update_params = local_update[key].detach().clone().cpu().numpy()
                        
                        # calculate 'fake' gradients
                        tmp_gradients = (original_params - update_params)/(float(args.learning_rate)*local_update_per_epoch*int(args.local_epochs))
                        gradients.append(tmp_gradients)
                        del tmp_gradients, original_params, update_params
                else:
                    for g_idx in range(len(local_update)):
                        gradients.append(local_update[g_idx].cpu().numpy())
                
                # 1.3 save the attack features
                gradient_hist_dict[speaker_id] = {}
                gradient_hist_dict[speaker_id]['gradient'] = gradients
                gradient_hist_dict[speaker_id]['gender'] = train_speaker_dict[speaker_id]['gender']
                del trainer

            # 1.4 dump the gradients for the later usage
            # pdb.set_trace()
            f = open(str(model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl')), "wb")
            pickle.dump(gradient_hist_dict, f)
            f.close()
                