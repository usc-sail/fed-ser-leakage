import torch
import torch.nn as nn
import argparse
import torch.multiprocessing
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
import shutil
import numpy as np
import torch
import pickle
from pathlib import Path
import pandas as pd

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'model'))
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'utils'))

import pytorch_lightning as pl
from attack_model import attack_model
from pytorch_lightning import seed_everything
from sklearn.model_selection import train_test_split

import pdb

# some general mapping for this script
gender_dict = {'F': 0, 'M': 1}
leak_layer_dict = {'full': ['w0', 'b0', 'w1', 'b1', 'w2', 'b2'],
                   'first': ['w0', 'b0'], 'second': ['w1', 'b1'], 'last': ['w2', 'b2']}
leak_layer_idx_dict = {'w0': 0, 'w1': 2, 'w2': 4, 'b0': 1, 'b1': 3, 'b2': 5}


class WeightDataGenerator():
    def __init__(self, dict_keys, data_dict = None):
        self.dict_keys = dict_keys
        self.data_dict = data_dict

    def __len__(self):
        return len(self.dict_keys)

    def __getitem__(self, idx):

        data_file_str = self.dict_keys[idx]
        gender = gender_dict[self.data_dict[data_file_str]['gender']]

        tmp_data = (self.data_dict[data_file_str][weight_name] - weight_norm_mean_dict[weight_name]) / (weight_norm_std_dict[weight_name] + 0.00001)
        weights = torch.from_numpy(np.ascontiguousarray(tmp_data))
        tmp_data = (self.data_dict[data_file_str][bias_name] - weight_norm_mean_dict[bias_name]) / (weight_norm_std_dict[bias_name] + 0.00001)
        bias = torch.from_numpy(np.ascontiguousarray(tmp_data))

        return weights, bias, gender

class AttackDataModule(pl.LightningDataModule):
    def __init__(self, train, val):
        super().__init__()
        self.train = train
        self.val = val

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=20, num_workers=0, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=20, num_workers=0, shuffle=False)
    


if __name__ == '__main__':

    torch.cuda.empty_cache() 
    torch.multiprocessing.set_sharing_strategy('file_system')

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--adv_dataset', default='iemocap')
    parser.add_argument('--feature_type', default='apc')
    parser.add_argument('--input_channel', default=1)
    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--model_learning_rate', default=0.0005)
    parser.add_argument('--input_spec_size', default=128)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--aug', default=None)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--num_epochs', default=200)
    parser.add_argument('--local_epochs', default=5)
    parser.add_argument('--norm', default='znorm')
    parser.add_argument('--device', default='1')
    parser.add_argument('--win_len', default=200)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--model_type', default='fed_sgd')
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--leak_layer', default='full')
    parser.add_argument('--dropout', default=0.2)
    args = parser.parse_args()
    
    seed_everything(8, workers=True)

    root_path = Path('/media/data/projects/speech-privacy')
    device = torch.device("cuda:"+str(args.device)) if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    model_setting_str = 'local_epoch_'+str(args.local_epochs) if args.model_type == 'fed_avg' else 'local_epoch_1'
    model_setting_str += '_dropout_' + str(args.dropout).replace('.', '')
    model_setting_str += '_lr_' + str(args.learning_rate)[2:]
    
    # 1. normalization tmp computations
    weight_norm_mean_dict, weight_norm_std_dict = {}, {}
    weight_sum, weight_sum_square = {}, {}
    for key in ['w0', 'w1', 'w2', 'b0', 'b1', 'b2']:
        weight_norm_mean_dict[key], weight_norm_std_dict[key] = 0, 0
        weight_sum[key], weight_sum_square[key] = 0, 0
    
    # the updates layer name and their idx in gradient file
    weight_name, bias_name = leak_layer_dict[args.leak_layer][0], leak_layer_dict[args.leak_layer][1]
    weight_idx, bias_idx = leak_layer_idx_dict[weight_name], leak_layer_idx_dict[bias_name]

    # 1.1 read all data and compute the tmp variables
    shadow_training_sample_size = 0
    shadow_data_dict = {}
    for shadow_idx in range(5):
        for epoch in range(int(args.num_epochs)):
            adv_federated_model_result_path = root_path.joinpath('federated_model_params', args.model_type, args.pred, args.feature_type, args.adv_dataset, model_setting_str, 'fold'+str(int(shadow_idx+1)))
            file_str = str(adv_federated_model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl'))
            # if shadow_idx == 0 and epoch < 10:
            if epoch % 20 == 0:
                print('reading shadow model %d, epoch %d' % (shadow_idx, epoch))
            with open(file_str, 'rb') as f:
                adv_gradient_dict = pickle.load(f)
            for speaker_id in adv_gradient_dict:
                data_key = str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id
                gradients = adv_gradient_dict[speaker_id]['gradient']
                gender = adv_gradient_dict[speaker_id]['gender']
                shadow_training_sample_size += 1
                
                # calculate running stats for computing std and mean
                shadow_data_dict[data_key] = {}
                shadow_data_dict[data_key]['gender'] = gender
                shadow_data_dict[data_key][weight_name] = gradients[weight_idx]
                shadow_data_dict[data_key][bias_name] = gradients[bias_idx]
                for layer_name in leak_layer_dict[args.leak_layer]:
                    weight_sum[layer_name] += gradients[leak_layer_idx_dict[layer_name]]
                    weight_sum_square[layer_name] += gradients[leak_layer_idx_dict[layer_name]]**2
            
    # 1.2 calculate std and mean
    for key in leak_layer_dict[args.leak_layer]:
        weight_norm_mean_dict[key] = weight_sum[key] / shadow_training_sample_size
        tmp_data = weight_sum_square[key] / shadow_training_sample_size - (weight_sum[key] / shadow_training_sample_size)**2
        weight_norm_std_dict[key] = np.sqrt(tmp_data)
    
    # 2. train model to infer gender
    # 2.1 define model
    train_key_list, validate_key_list = train_test_split(list(shadow_data_dict.keys()), test_size=0.2, random_state=0)
    model = attack_model(args.leak_layer, args.feature_type)
    model = model.to(device)

    # 2.2 define data loader
    dataset_train = WeightDataGenerator(train_key_list, shadow_data_dict)
    dataset_valid = WeightDataGenerator(validate_key_list, shadow_data_dict)
    data_module = AttackDataModule(dataset_train, dataset_valid)

    # 2.3 initialize the early_stopping object
    early_stopping = EarlyStopping(monitor="val_loss", mode='min', patience=5,
                                   stopping_threshold=1e-4, check_finite=True)

    # 2.4 log saving path
    attack_model_result_path = Path.cwd().parents[0].joinpath('results', 'attack', args.leak_layer, args.model_type, args.feature_type, model_setting_str)
    log_path = Path.joinpath(attack_model_result_path, 'log_private_' + str(args.dataset))
    if log_path.exists(): shutil.rmtree(log_path)
    Path.mkdir(log_path, parents=True, exist_ok=True)
    mlf_logger = MLFlowLogger(experiment_name="ser", save_dir=str(log_path))

    checkpoint_callback = ModelCheckpoint(monitor="val_acc_epoch", mode="max", 
                                          dirpath=str(attack_model_result_path),
                                          filename='private_' + str(args.dataset) + '_model')
    # 2.5 training using pytorch lighting framework
    trainer = pl.Trainer(logger=mlf_logger, gpus=1, callbacks=[checkpoint_callback, early_stopping], max_epochs=50)
    trainer.fit(model, data_module)
    
    # 3. we evaluate the attacker performance on service provider training
    save_result_df = pd.DataFrame()
    # 3.1 we perform 5 fold evaluation, since we also train the private data 5 times
    for fold_idx in range(5):
        test_data_dict = {}
        for epoch in range(int(args.num_epochs)):
            torch.cuda.empty_cache()
            row_df = pd.DataFrame(index=['fold'+str(int(fold_idx+1))])
            
            # Model related
            federated_model_result_path = root_path.joinpath('federated_model_params', args.model_type, args.pred, args.feature_type, args.dataset, model_setting_str, 'fold'+str(int(fold_idx+1)))
            weight_file_str = str(federated_model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl'))

            with open(weight_file_str, 'rb') as f:
                test_gradient_dict = pickle.load(f)
            for speaker_id in test_gradient_dict:
                data_key = str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id
                gradients = test_gradient_dict[speaker_id]['gradient']
                test_data_dict[data_key] = {}
                test_data_dict[data_key]['gender'] = test_gradient_dict[speaker_id]['gender']
                test_data_dict[data_key][weight_name] = gradients[weight_idx]
                test_data_dict[data_key][bias_name] = gradients[bias_idx]

        dataset_test = WeightDataGenerator(list(test_data_dict.keys()), test_data_dict)
        dataloader_test = DataLoader(dataset_test, batch_size=20, num_workers=1, shuffle=False)
        
        # model.freeze()
        # trainer.test(test_dataloaders=data_module.train_dataloader())
        result_dict = trainer.test(dataloaders=dataloader_test, ckpt_path='best')
        pdb.set_trace()
        row_df['acc'], row_df['uar'] = result_dict[0]['test_acc_epoch'], result_dict[0]['test_uar_epoch']
        save_result_df = pd.concat([save_result_df, row_df])

        del dataset_test, dataloader_test
        
    row_df = pd.DataFrame(index=['average'])
    row_df['acc'], row_df['uar'] = np.mean(save_result_df['acc']), np.mean(save_result_df['uar'])
    save_result_df = pd.concat([save_result_df, row_df])
    save_result_df.to_csv(str(attack_model_result_path.joinpath('private_' + str(args.dataset) + '_result.csv')))

