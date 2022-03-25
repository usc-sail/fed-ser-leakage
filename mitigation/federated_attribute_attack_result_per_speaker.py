import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.multiprocessing
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
import pickle

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
        w1 = []
        b1 = []
        w2 = []
        b2 = []
        gender = []

        data_file_str = self.dict_keys[idx]

        tmp_data = (self.data_dict[data_file_str]['w0'] - weight_norm_mean_dict['w0']) / (weight_norm_std_dict['w0'] + 0.00001)
        w0 = torch.from_numpy(np.ascontiguousarray(tmp_data))
        tmp_data = (self.data_dict[data_file_str]['b0'] - weight_norm_mean_dict['b0']) / (weight_norm_std_dict['b0'] + 0.00001)
        b0 = torch.from_numpy(np.ascontiguousarray(tmp_data))

        tmp_data = (self.data_dict[data_file_str]['w1'] - weight_norm_mean_dict['w1']) / (weight_norm_std_dict['w1'] + 0.00001)
        w1 = torch.from_numpy(np.ascontiguousarray(tmp_data))
        tmp_data = (self.data_dict[data_file_str]['b1'] - weight_norm_mean_dict['b1']) / (weight_norm_std_dict['b1'] + 0.00001)
        b1 = torch.from_numpy(np.ascontiguousarray(tmp_data))
        
        tmp_data = (self.data_dict[data_file_str]['w2'] - weight_norm_mean_dict['w2']) / (weight_norm_std_dict['w2'] + 0.00001)
        w2 = torch.from_numpy(np.ascontiguousarray(tmp_data))
        tmp_data = (self.data_dict[data_file_str]['b2'] - weight_norm_mean_dict['b2']) / (weight_norm_std_dict['b2'] + 0.00001)
        b2 = torch.from_numpy(np.ascontiguousarray(tmp_data))
    
        gender = gender_dict[self.data_dict[data_file_str]['gender']]
        return w0, w1, w2, b0, b1, b2, gender


def run_one_epoch(data_loader):
    
    first_layer_model.eval()
    if args.leak_layer == 'second' or args.leak_layer == 'last' or args.leak_layer == 'fusion':
        second_layer_model.eval()
        last_layer_model.eval()
    step_outputs = []
    
    # feature size from all layers, attack input
    first_layer_features = feature_len_dict[args.feature_type] * 256
    second_layer_features = 256 * 128
    last_layer_features = 128 * 4
    total_features = first_layer_features + second_layer_features + last_layer_features

    for batch_idx, data_batch in enumerate(data_loader):
        w0, w1, w2, b0, b1, b2, y = data_batch
        w0, w1, w2 = w0.to(device), w1.to(device), w2.to(device)
        b0, b1, b2, y = b0.to(device), b1.to(device), b2.to(device), y.to(device)
        
        if args.leak_layer == 'fusion':
            logits0 = first_layer_model(w0.float().unsqueeze(dim=1), b0.float())
            logits1 = second_layer_model(w1.float().unsqueeze(dim=1), b1.float())
            logits2 = last_layer_model(w2.float().unsqueeze(dim=1), b2.float())
            logits0, logits1, logits2 = torch.exp(logits0), torch.exp(logits1), torch.exp(logits2)

            final_logits = (first_layer_features/total_features)*logits0
            final_logits += (second_layer_features/total_features)*logits1
            final_logits += (last_layer_features/total_features)*logits2
            del logits0, logits1, logits2
        elif args.leak_layer == 'first':
            logits0 = first_layer_model(w0.float().unsqueeze(dim=1), b0.float())
            final_logits = logits0
            del logits0
        elif args.leak_layer == 'second':
            logits1 = second_layer_model(w1.float().unsqueeze(dim=1), b1.float())
            final_logits = logits1
            del logits1
        elif args.leak_layer == 'last':
            logits2 = last_layer_model(w2.float().unsqueeze(dim=1), b2.float())
            final_logits = logits2
            del logits2
        
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
    parser.add_argument('--num_sample', default=5)
    parser.add_argument('--attack_dropout', default=0.2)
    parser.add_argument('--privacy_budget', default=None)
    parser.add_argument('--save_dir', default='/media/data/projects/speech-privacy')
    args = parser.parse_args()
    
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
    attack_dropout_str = '' if float(args.attack_dropout) == 0.2 else '_attack_dropout_' + str(args.attack_dropout).replace('.', '')
    first_layer_model = attack_model('first', args.feature_type)
    first_layer_model = first_layer_model.to(device)
    attack_model_path = Path(os.path.realpath(__file__)).parents[1].joinpath('results', 'attack', 'first', args.model_type, args.feature_type, model_setting_str+attack_dropout_str, 'private_' + str(args.dataset) + '.pt')
    first_layer_model.load_state_dict(torch.load(str(attack_model_path), map_location=device))
    
    if args.leak_layer == 'second' or args.leak_layer == 'fusion':
        second_layer_model = attack_model('second', args.feature_type)
        second_layer_model = second_layer_model.to(device)
        attack_model_path = Path(os.path.realpath(__file__)).parents[1].joinpath('results', 'attack', 'second', args.model_type, args.feature_type, model_setting_str+attack_dropout_str, 'private_' + str(args.dataset) + '.pt')
        second_layer_model.load_state_dict(torch.load(str(attack_model_path), map_location=device))

    if args.leak_layer == 'last' or args.leak_layer == 'fusion':
        last_layer_model = attack_model('last', args.feature_type)
        last_layer_model = last_layer_model.to(device)
        attack_model_path = Path(os.path.realpath(__file__)).parents[1].joinpath('results', 'attack', 'last', args.model_type, args.feature_type, model_setting_str+attack_dropout_str, 'private_' + str(args.dataset) + '.pt')
        last_layer_model.load_state_dict(torch.load(str(attack_model_path), map_location=device))

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
                # pdb.set_trace()
                gradients = test_fed_weight_hist_dict[speaker_id]['gradient']
                if speaker_id not in test_data_dict: 
                    test_data_dict[speaker_id] = {}
                
                test_data_dict[speaker_id][str(fold_idx)+'_'+str(epoch)] = {}
                test_data_dict[speaker_id][str(fold_idx)+'_'+str(epoch)]['w0'] = gradients[0]
                test_data_dict[speaker_id][str(fold_idx)+'_'+str(epoch)]['b0'] = gradients[1]
                test_data_dict[speaker_id][str(fold_idx)+'_'+str(epoch)]['w1'] = gradients[2]
                test_data_dict[speaker_id][str(fold_idx)+'_'+str(epoch)]['b1'] = gradients[3]
                test_data_dict[speaker_id][str(fold_idx)+'_'+str(epoch)]['w2'] = gradients[4]
                test_data_dict[speaker_id][str(fold_idx)+'_'+str(epoch)]['b2'] = gradients[5]
                test_data_dict[speaker_id][str(fold_idx)+'_'+str(epoch)]['gender'] = test_fed_weight_hist_dict[speaker_id]['gender']
        
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
                        speaker_dict[idx_array[0]]['w1'] += test_data_dict[speaker_id][key]['w1']
                        speaker_dict[idx_array[0]]['b1'] += test_data_dict[speaker_id][key]['b1']
                        speaker_dict[idx_array[0]]['w2'] += test_data_dict[speaker_id][key]['w2']
                        speaker_dict[idx_array[0]]['b2'] += test_data_dict[speaker_id][key]['b2']
                        
                speaker_dict[idx_array[0]]['w0'] = speaker_dict[idx_array[0]]['w0'] / num_sample
                speaker_dict[idx_array[0]]['b0'] = speaker_dict[idx_array[0]]['b0'] / num_sample
                speaker_dict[idx_array[0]]['w1'] = speaker_dict[idx_array[0]]['w1'] / num_sample
                speaker_dict[idx_array[0]]['b1'] = speaker_dict[idx_array[0]]['b1'] / num_sample
                speaker_dict[idx_array[0]]['w2'] = speaker_dict[idx_array[0]]['w2'] / num_sample
                speaker_dict[idx_array[0]]['b2'] = speaker_dict[idx_array[0]]['b2'] / num_sample
                
                dataset_test = WeightDataGenerator(list(speaker_dict.keys()), speaker_dict)
                test_loader = DataLoader(dataset_test, batch_size=20, num_workers=0, shuffle=False)
                prediction = run_one_epoch(test_loader)
                
                predictions.append(prediction)
                predictions_per_speaker.append(prediction)
                truths.append(gender_dict[gender])
                truths_per_speaker.append(gender_dict[gender])
                del dataset_test, test_loader
            
        step_outputs = []
        step_outputs.append({'loss': 0, 'pred': predictions, 'truth': truths})
        test_result = result_summary(step_outputs, mode='test', epoch=0)
        row_df['acc'], row_df['uar'] = test_result['acc'], test_result['uar']
        save_result_df = pd.concat([save_result_df, row_df])
        
    row_df = pd.DataFrame(index=['average'])
    row_df['acc'], row_df['uar'] = np.mean(save_result_df['acc']), np.mean(save_result_df['uar'])
    save_result_df = pd.concat([save_result_df, row_df])
    
    attack_model_result_csv_path = Path(os.path.realpath(__file__)).parents[1].joinpath('results', 'attack_by_client', args.leak_layer, args.model_type, args.feature_type, model_setting_str, str(args.num_sample))
    Path.mkdir(attack_model_result_csv_path, parents=True, exist_ok=True)
    save_result_df.to_csv(str(attack_model_result_csv_path.joinpath('private_' + str(args.dataset) + '_result.csv')))

