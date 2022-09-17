import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.multiprocessing
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
import copy
import shutil

import numpy as np
from pathlib import Path
import pandas as pd

import sys, os, pdb, pickle, argparse
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'utils'))

from training_tools import setup_seed, result_summary
from attack_model import attack_model

emo_dict = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
affect_dict = {'low': 0, 'med': 1, 'high': 2}

gender_dict = {'F': 0, 'M': 1}
speaker_id_arr_dict = {'msp-improv': np.arange(0, 12, 1), 
                       'crema-d': np.arange(1001, 1092, 1),
                       'iemocap': np.arange(0, 10, 1)}
leak_layer_dict = {'full': ['w0', 'b0', 'w1', 'b1', 'w2', 'b2'],
                   'first': ['w0', 'b0'], 'second': ['w1', 'b1'], 'last': ['w2', 'b2']}

# define feature len mapping
feature_len_dict = {'emobase': 988, 'ComParE': 6373, 'wav2vec': 9216, 
                    'apc': 512, 'distilhubert': 768, 'tera': 768, 'wav2vec2': 768,
                    'decoar2': 768, 'cpc': 256, 'audio_albert': 768, 
                    'mockingjay': 768, 'npc': 512, 'vq_apc': 512, 'vq_wav2vec': 512}

class WeightDataGenerator():
    def __init__(self, dict_keys, data_dict = None):
        self.dict_keys = dict_keys
        self.data_dict = data_dict

    def __len__(self):
        return len(self.dict_keys)

    def __getitem__(self, idx):

        w0 = []
        b0 = []
        gender = []

        data_file_str = self.dict_keys[idx]
        tmp_data = (self.data_dict[data_file_str]['w0'] - weight_norm_mean_dict['w0']) / (weight_norm_std_dict['w0'] + 0.00001)
        w0 = torch.from_numpy(np.ascontiguousarray(tmp_data))
        tmp_data = (self.data_dict[data_file_str]['b0'] - weight_norm_mean_dict['b0']) / (weight_norm_std_dict['b0'] + 0.00001)
        b0 = torch.from_numpy(np.ascontiguousarray(tmp_data))
        gender = gender_dict[self.data_dict[data_file_str]['gender']]
        return w0, b0, gender
    

def finetune_one_epoch(model, data_loader, optimizer, loss_func):
    
    model.train()
    for batch_idx, data_batch in enumerate(data_loader):
        weights, bias, y = data_batch
        weights, bias, y = weights.to(device), bias.to(device), y.to(device)
        logits = model(weights.float().unsqueeze(dim=1), bias.float())
        loss = loss_func(logits, y)

        # step the loss back
        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del data_batch, logits, loss
        torch.cuda.empty_cache()

def run_one_epoch(model, data_loader):
    model.eval()
    
    # feature size from all layers, attack input
    first_layer_features = feature_len_dict[args.feature_type] * 256
    second_layer_features = 256 * 128
    last_layer_features = 128 * 4
    total_features = first_layer_features + second_layer_features + last_layer_features

    for batch_idx, data_batch in enumerate(data_loader):
        w0, b0, y = data_batch
        w0 = w0.to(device)
        b0, y = b0.to(device), y.to(device)
        logits0 = model(w0.float().unsqueeze(dim=1), b0.float())
        final_logits = logits0
        del logits0
        
        final_logits = torch.exp(final_logits)
        prediction = np.argmax(np.mean(final_logits.detach().cpu().numpy(), axis=0))
        del data_batch
        torch.cuda.empty_cache()
    return prediction


if __name__ == '__main__':

    torch.cuda.empty_cache() 
    torch.multiprocessing.set_sharing_strategy('file_system')

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--adv_dataset', default='msp-improv_crema-d')
    parser.add_argument('--feature_type', default='apc')
    parser.add_argument('--learning_rate', default=0.0005)
    parser.add_argument('--batch_size', default=20)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--num_epochs', default=200)
    parser.add_argument('--local_epochs', default=1)
    parser.add_argument('--norm', default='znorm')
    parser.add_argument('--device', default='1')
    parser.add_argument('--model_type', default='fed_avg')
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--dropout', default=0.2)
    parser.add_argument('--leak_layer', default='fusion')
    parser.add_argument('--privacy_budget', default=None)
    parser.add_argument('--num_sample', default=5)
    parser.add_argument('--save_dir', default='/media/data/projects/speech-privacy')
    args = parser.parse_args()
    preprocess_path = Path(args.save_dir).joinpath('federated_learning', args.feature_type, args.pred)
    
    setup_seed(8)
    torch.manual_seed(8)
    pred = 'affect' if args.pred == 'arousal' or args.pred == 'valence' else 'emotion'
    
    save_result_df = pd.DataFrame()
    device = torch.device("cuda:"+str(args.device)) if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    model_setting_str = 'local_epoch_'+str(args.local_epochs) if args.model_type == 'fed_avg' else 'local_epoch_1'
    model_setting_str += '_dropout_' + str(args.dropout).replace('.', '')
    model_setting_str += '_lr_' + str(args.learning_rate)[2:]
    
    weight_file_list = []
    weight_norm_mean_dict, weight_norm_std_dict = {}, {}
    weight_sum, weight_sum_square = {}, {}
    for key in ['w0', 'w1', 'w2', 'b0', 'b1', 'b2']:
        weight_norm_mean_dict[key], weight_norm_std_dict[key] = [], []
        weight_sum[key] = 0
        weight_sum_square[key] = 0

    shadow_training_sample_size = 0
    for shadow_idx in range(5):
        for epoch in range(int(args.num_epochs)):
            adv_federated_model_result_path = Path(args.save_dir).joinpath('tmp_model_params', args.model_type, args.pred, args.feature_type, args.adv_dataset, model_setting_str, 'fold'+str(int(shadow_idx+1)))
            weight_file_str = str(adv_federated_model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl'))
            weight_file_list.append(weight_file_str)
            if epoch % 20 == 0:
                print('reading shadow model %d, epoch %d' % (shadow_idx, epoch))
            with open(weight_file_str, 'rb') as f:
                adv_fed_weight_hist_dict = pickle.load(f)
            for speaker_id in adv_fed_weight_hist_dict:
                gradients = adv_fed_weight_hist_dict[speaker_id]['gradient']
                shadow_training_sample_size += 1
                
                weight_sum['w0'] += gradients[0]
                weight_sum['b0'] += gradients[1]
                weight_sum_square['w0'] += gradients[0]**2
                weight_sum_square['b0'] += gradients[1]**2
                    
                weight_sum['w1'] += gradients[2]
                weight_sum['b1'] += gradients[3]
                weight_sum_square['w1'] += gradients[2]**2
                weight_sum_square['b1'] += gradients[3]**2
                    
                weight_sum['w2'] += gradients[4]
                weight_sum['b2'] += gradients[5]
                weight_sum_square['w2'] += gradients[4]**2
                weight_sum_square['b2'] += gradients[5]**2

    for key in leak_layer_dict['full']:
        weight_norm_mean_dict[key] = weight_sum[key] / shadow_training_sample_size
        tmp_data = weight_sum_square[key] / shadow_training_sample_size - weight_norm_mean_dict[key]**2
        weight_norm_std_dict[key] = np.sqrt(tmp_data)

    # load the evaluation model
    first_layer_model = attack_model('first', args.feature_type)
    first_layer_model = first_layer_model.to(device)
    attack_model_path = Path(os.path.realpath(__file__)).parents[1].joinpath('results', 'attack', 'first', args.model_type, args.feature_type, model_setting_str, 'private_' + str(args.dataset) + '.pt')
    first_layer_model.load_state_dict(torch.load(str(attack_model_path), map_location=device))
    
    # we evaluate the attacker performance on service provider training
    if args.privacy_budget:
        model_setting_str += '_udp_' + str(args.privacy_budget)

    for fold_idx in range(0, 5):
        test_data_dict = {}
        for epoch in range(int(args.num_epochs)):
            torch.cuda.empty_cache()
            save_row_str = 'fold'+str(int(fold_idx+1))
            row_df = pd.DataFrame(index=[save_row_str])
            
            # Model related
            federated_model_result_path = Path(args.save_dir).joinpath('federated_model_params', args.model_type, args.pred, args.feature_type, args.dataset, model_setting_str, 'fold'+str(int(fold_idx+1)))
            weight_file_str = str(federated_model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl'))

            with open(weight_file_str, 'rb') as f:
                test_fed_weight_hist_dict = pickle.load(f)
            for speaker_id in test_fed_weight_hist_dict:
                gradients = test_fed_weight_hist_dict[speaker_id]['gradient']
                if speaker_id not in test_data_dict: 
                    test_data_dict[speaker_id] = {}
                
                test_data_dict[speaker_id][str(fold_idx)+'_'+str(epoch)] = {}
                test_data_dict[speaker_id][str(fold_idx)+'_'+str(epoch)]['w0'] = gradients[0]
                test_data_dict[speaker_id][str(fold_idx)+'_'+str(epoch)]['b0'] = gradients[1]
                test_data_dict[speaker_id][str(fold_idx)+'_'+str(epoch)]['gender'] = test_fed_weight_hist_dict[speaker_id]['gender']
                test_data_dict[speaker_id][str(fold_idx)+'_'+str(epoch)]['epoch'] = epoch
        
        results_dict = {}
        predictions, truths = [], []
        
        for speaker_id in test_data_dict:
            predictions_per_speaker, truths_per_speaker = [], []
            for idx_trial in range(10):
                speaker_dict = {}
                key_list = list(test_data_dict[speaker_id])
                num_sample = int(args.num_sample) if len(key_list) > int(args.num_sample) else len(key_list)
                np.random.seed(idx_trial)
                idx_array = np.random.choice(range(len(key_list)), num_sample, replace=False)
                
                speaker_dict[idx_array[0]] = deepcopy(test_data_dict[speaker_id][key_list[idx_array[0]]])
                gender = test_data_dict[speaker_id][key_list[idx_array[0]]]['gender']
                if num_sample > 1:
                    for idx in idx_array[1:]:
                        key = key_list[idx]
                        speaker_dict[idx_array[0]]['w0'] += test_data_dict[speaker_id][key]['w0']
                        speaker_dict[idx_array[0]]['b0'] += test_data_dict[speaker_id][key]['b0']
                        
                speaker_dict[idx_array[0]]['w0'] = speaker_dict[idx_array[0]]['w0'] / num_sample
                speaker_dict[idx_array[0]]['b0'] = speaker_dict[idx_array[0]]['b0'] / num_sample
                
                # generate gradients
                epoch_str = ''
                for idx in idx_array:
                    key = key_list[idx]
                    epoch_str += str(test_data_dict[speaker_id][key]['epoch']) + ' '
                    
                cmd_str = 'taskset 300 python3 train/federated_ser_gradient_generates.py --dataset ' + args.dataset
                cmd_str += ' --adv_dataset ' + args.adv_dataset
                cmd_str += ' --privacy_budget ' + args.privacy_budget
                cmd_str += ' --feature_type ' + args.feature_type
                cmd_str += ' --dropout ' + str(args.dropout)
                cmd_str += ' --norm znorm --optimizer adam'
                cmd_str += ' --model_type fed_avg'
                cmd_str += ' --learning_rate ' + str(str(args.learning_rate))
                cmd_str += ' --local_epochs 1'
                cmd_str += ' --epoch ' + epoch_str
                cmd_str += ' --num_epochs 200'
                cmd_str += ' --save_dir /media/data/projects/speech-privacy'
                
                print('Traing SER model for one epoch')
                print(cmd_str)
                os.system(cmd_str)
                
                # finetune model
                loss = nn.NLLLoss().to(device)
                optimizer = optim.Adam(first_layer_model.parameters(), lr=float(args.learning_rate), weight_decay=1e-04, betas=(0.9, 0.98), eps=1e-9)
                
                finetune_data_dict = {}
                finetune_data_path = Path(args.save_dir).joinpath('federated_model_params_fixed_global', args.model_type, args.pred, args.feature_type, args.adv_dataset, model_setting_str, 'fold1')
                for idx in idx_array:
                    key = key_list[idx]
                    finetune_epoch = test_data_dict[speaker_id][key]['epoch']                  
                        
                    finetune_weight_file_str = str(finetune_data_path.joinpath('gradient_hist_'+str(finetune_epoch)+'.pkl'))
                    with open(finetune_weight_file_str, 'rb') as f:
                        finetune_weight_hist_dict = pickle.load(f)
                    for finetune_speaker_id in finetune_weight_hist_dict:
                        gradients = finetune_weight_hist_dict[finetune_speaker_id]['gradient']
                        data_key = finetune_speaker_id + '_' + str(finetune_epoch)
                        
                        finetune_data_dict[data_key] = {}
                        finetune_data_dict[data_key]['gender'] = finetune_weight_hist_dict[finetune_speaker_id]['gender']
                        finetune_data_dict[data_key]['w0'] = gradients[0]
                        finetune_data_dict[data_key]['b0'] = gradients[1]
                
                finetune_key_list = list(finetune_data_dict.keys())
                dataset_finetune = WeightDataGenerator(finetune_key_list, finetune_data_dict)
                finetune_loader = DataLoader(dataset_finetune, batch_size=20, num_workers=0, shuffle=True)
                finetuned_model = copy.deepcopy(first_layer_model)
                for epoch in range(5):
                    # perform the training, validate, and test
                    print('finetune epoch %d' % epoch)
                    finetune_one_epoch(finetuned_model, finetune_loader, optimizer, loss)
                shutil.rmtree(finetune_data_path)
                
                dataset_test = WeightDataGenerator(list(speaker_dict.keys()), speaker_dict)
                test_loader = DataLoader(dataset_test, batch_size=20, num_workers=0, shuffle=False)
                prediction = run_one_epoch(finetuned_model, test_loader)
                
                predictions.append(prediction)
                predictions_per_speaker.append(prediction)
                truths.append(gender_dict[gender])
                truths_per_speaker.append(gender_dict[gender])
                del dataset_test, test_loader
        
                print('test fold %d, idx_trial %d' % (fold_idx, idx_trial))
                step_outputs = []
                step_outputs.append({'loss': 0, 'pred': predictions, 'truth': truths})
                test_result = result_summary(step_outputs, mode='test', epoch=0)
                
        step_outputs = []
        step_outputs.append({'loss': 0, 'pred': predictions, 'truth': truths})
        test_result = result_summary(step_outputs, mode='test', epoch=0)
        row_df['acc'], row_df['uar'] = test_result['acc'], test_result['uar']
        save_result_df = pd.concat([save_result_df, row_df])
        
        attack_model_result_csv_path = Path(os.path.realpath(__file__)).parents[1].joinpath('results', 'attack_by_client_finetune', args.leak_layer, args.model_type, args.feature_type, model_setting_str, str(args.num_sample))
        Path.mkdir(attack_model_result_csv_path, parents=True, exist_ok=True)
        save_result_df.to_csv(str(attack_model_result_csv_path.joinpath('private_' + str(args.dataset) + '_result.csv')))
            
    row_df = pd.DataFrame(index=['average'])
    row_df['acc'], row_df['uar'] = np.mean(save_result_df['acc']), np.mean(save_result_df['uar'])
    save_result_df = pd.concat([save_result_df, row_df])
    
    attack_model_result_csv_path = Path(os.path.realpath(__file__)).parents[1].joinpath('results', 'attack_by_client_finetune', args.leak_layer, args.model_type, args.feature_type, model_setting_str, str(args.num_sample))
    Path.mkdir(attack_model_result_csv_path, parents=True, exist_ok=True)
    save_result_df.to_csv(str(attack_model_result_csv_path.joinpath('private_' + str(args.dataset) + '_result.csv')))
