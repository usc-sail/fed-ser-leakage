from moviepy.tools import verbose_print
import torch
import torch.nn as nn
import argparse, logging
import torch.multiprocessing
from torch.utils.data import DataLoader

import numpy as np
from pathlib import Path
import pandas as pd
import copy, time, pickle, shutil, sys, os, pdb
from copy import deepcopy

sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'model'))
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), 'model'))

from baseline_models import dnn_classifier

from update import average_weights, average_gradients
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
import pytorch_lightning as pl

pl.utilities.distributed.log.setLevel(logging.ERROR)

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
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        data = self.dataset[self.idxs[item]]['data']
        label = emo_dict[self.dataset[self.idxs[item]]['label']]
        dataset_str = self.dataset[self.idxs[item]]['dataset']

        return data, label, dataset_str


def read_data_dict_by_client(dataset_list, fold_idx):
    
    return_train_dict, return_test_dict = {}, {}
    # prepare the data for the training
    for dataset in dataset_list:
        with open(preprocess_path.joinpath(dataset, 'fold'+str(int(fold_idx+1)), 'training_'+args.norm+'.pkl'), 'rb') as f:
            train_dict = pickle.load(f)
        with open(preprocess_path.joinpath(dataset, 'fold'+str(int(fold_idx+1)), 'test_'+args.norm+'.pkl'), 'rb') as f:
            test_dict = pickle.load(f)
        
        for tmp_dict in [train_dict, test_dict]:
            for key in tmp_dict: tmp_dict[key]['dataset'] = dataset

        # test set will be the same
        for key in test_dict: return_test_dict[key] = test_dict[key].copy()
        
        # we remake the data dict per speaker for the ease of local training
        speaker_data_dict = {}
        for key in train_dict:
            speaker_id = str(train_dict[key]['speaker_id'])
            if speaker_id not in speaker_data_dict: 
                speaker_data_dict[speaker_id] = []
            speaker_data_dict[speaker_id].append(key)
        
        # in federated setting that the norm validation dict will be based on local client data
        # so we combine train_dict and validate_dict in centralized setting
        # then choose certain amount data per client as local validation set
        if dataset == 'crema-d':
            for key in train_dict:
                speaker_id = str(train_dict[key]['speaker_id'])
                if speaker_id not in return_train_dict: return_train_dict[speaker_id] = {}
                return_train_dict[speaker_id][key] = train_dict[key].copy()
        else:
            # we want to divide speaker data if the dataset is iemocap or msp-improv to increase client size
            for speaker_id in speaker_data_dict:
                # in iemocap and msp-improv
                # we spilit each speaker data into 10 parts in order to create more clients
                idx_array = np.random.permutation(len(speaker_data_dict[speaker_id]))
                key_list = speaker_data_dict[speaker_id]
                split_array = np.array_split(idx_array, 10)
                for split_idx in range(len(split_array)):
                    # we randomly pick 10% of data
                    idxs_train = split_array[split_idx]
                    for idx in idxs_train:
                        key = key_list[idx]
                        if speaker_id+'_'+str(split_idx) not in return_train_dict: return_train_dict[speaker_id+'_'+str(split_idx)] = {}
                        return_train_dict[speaker_id+'_'+str(split_idx)][key] = train_dict[key].copy()
    return return_train_dict, return_test_dict

if __name__ == '__main__':

    # torch.cuda.empty_cache() 
    torch.multiprocessing.set_sharing_strategy('file_system')

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='iemocap')
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
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--save_dir', default='/media/data/projects/speech-privacy')
    args = parser.parse_args()

    preprocess_path = Path(args.save_dir).joinpath('federated_learning', args.feature_type, args.pred)
    
    # set seeds
    seed_everything(8, workers=True)
    save_result_df = pd.DataFrame()
    dataset_list = args.dataset.split('_')

    # find device
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    # We perform 5 fold experiments
    for fold_idx in range(5):
        # save folder details
        save_row_str = 'fold'+str(int(fold_idx+1))
        row_df = pd.DataFrame(index=[save_row_str])
        
        model_setting_str = 'local_epoch_'+str(args.local_epochs) if args.model_type == 'fed_avg' else 'local_epoch_1'
        model_setting_str += '_dropout_' + str(args.dropout).replace('.', '')
        model_setting_str += '_lr_' + str(args.learning_rate)[2:]
        
        # Read the data per speaker
        train_speaker_dict, test_speaker_dict = read_data_dict_by_client(dataset_list, fold_idx)
        num_of_speakers = len(train_speaker_dict)
        speaker_list = list(set(train_speaker_dict.keys()))

        # We save the utterance keys used for training and validation per speaker (client)
        train_val_idx_dict = {}
        for speaker_id in train_speaker_dict:
            idx_array = np.random.permutation(len(train_speaker_dict[speaker_id]))
            train_val_idx_dict[speaker_id], tmp_keys = {}, list(train_speaker_dict[speaker_id].keys())
            train_val_idx_dict[speaker_id]['train'], train_val_idx_dict[speaker_id]['val'] = [], []
            for idx in idx_array[:int(0.8*len(idx_array))]: train_val_idx_dict[speaker_id]['train'].append(tmp_keys[idx])
            for idx in idx_array[int(0.8*len(idx_array)):]: train_val_idx_dict[speaker_id]['val'].append(tmp_keys[idx])
        
        # Define the model
        global_model = dnn_classifier(pred='emotion', input_spec=feature_len_dict[args.feature_type], dropout=float(args.dropout), args=args)
        global_model = global_model.to(device)
        global_weights = global_model.state_dict()
        
        # log saving path
        model_result_path = Path(args.save_dir).joinpath('federated_model_params', args.model_type, args.pred, args.feature_type, args.dataset, model_setting_str, save_row_str)
        model_result_csv_path = Path(os.path.realpath(__file__)).parents[1].joinpath('results', args.pred, args.model_type, args.feature_type, model_setting_str)
        Path.mkdir(model_result_path, parents=True, exist_ok=True)
        Path.mkdir(model_result_csv_path, parents=True, exist_ok=True)

        log_path = Path.joinpath(model_result_path, 'log')
        if log_path.exists(): shutil.rmtree(log_path)
        Path.mkdir(log_path, parents=True, exist_ok=True)
        mlf_logger = MLFlowLogger(experiment_name="ser", save_dir=str(log_path))
        
        # trainer
        if args.model_type == 'fed_avg':
            trainer = pl.Trainer(logger=mlf_logger, gpus=1, weights_summary=None, 
                                 progress_bar_refresh_rate=0, max_epochs=1)
        else:
            trainer = pl.Trainer(logger=mlf_logger, gpus=1, weights_summary=None, 
                                 progress_bar_refresh_rate=0, max_epochs=1, limit_train_batches=1)
        
        # test loader
        dataset_test = DatasetGenerator(test_speaker_dict, list(test_speaker_dict.keys()))
        test_dataloaders = DataLoader(dataset_test, batch_size=20, num_workers=0, shuffle=False)

        # Training steps
        result_dict, best_score = {}, 0
        for epoch in range(int(args.num_epochs)):
            # we choose 10% of clients in training
            idxs_speakers = np.random.choice(range(num_of_speakers), int(0.1 * num_of_speakers), replace=False)
            
            # define list varibles that saves the weights, loss, num_sample, etc.
            local_updates, local_losses, local_num_sampels = [], [], []
            gradient_hist_dict = {}

            # 1. Local training, return weights in fed_avg, return gradients in fed_sgd
            for idx in idxs_speakers:
                speaker_id = speaker_list[idx]
                print('speaker id %s' % (speaker_id))
                
                # 1.1 Local training
                local_model = copy.deepcopy(global_model)
                dataset_train = DatasetGenerator(train_speaker_dict[speaker_id], train_val_idx_dict[speaker_id]['train'])
                train_dataloaders = DataLoader(dataset_train, batch_size=20, num_workers=0, shuffle=True)
                trainer.fit(local_model, train_dataloaders=train_dataloaders)
                
                # read params to save
                train_sample_size = int(trainer.callback_metrics['train_size'])
                local_losses.append(float(trainer.callback_metrics['train_loss']))
                local_num_sampels.append(train_sample_size)
                # read shared updates: parameters in fed_avg and gradients for fed_sgd
                if args.model_type == 'fed_avg':
                    local_update = copy.deepcopy(local_model.state_dict())
                    local_updates.append(local_update)
                else:
                    local_update = []
                    for param in local_model.gradient: local_update.append(param)
                    local_updates.append(local_update)
                
                # 1.2 calculate and save the raw gradients or pseudo gradients
                gradients = []
                if args.model_type == 'fed_avg':
                    # 'fake' gradients saving code
                    # iterate all layers in the classifier model
                    original_model = copy.deepcopy(global_model).state_dict()

                    # calculate how many updates per local epoch 
                    local_update_per_epoch = int(train_sample_size / int(args.batch_size)) + 1
                        
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
                tmp_key = list(train_speaker_dict[speaker_id].keys())[0]
                gradient_hist_dict[speaker_id] = {}
                gradient_hist_dict[speaker_id]['gradient'] = gradients
                gradient_hist_dict[speaker_id]['gender'] = train_speaker_dict[speaker_id][tmp_key]['gender']
                del local_model

            # 1.4 dump the gradients for the later usage
            f = open(str(model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl')), "wb")
            pickle.dump(gradient_hist_dict, f)
            f.close()
            
            # 2. global model updates
            total_num_samples = np.sum(local_num_sampels)
            if args.model_type == 'fed_avg':
                # 2.1 average global weights
                global_weights = average_weights(local_updates, local_num_sampels)
            else:
                # 2.1 average global gradients
                global_gradients = average_gradients(local_updates, local_num_sampels)
                # 2.2 update global weights
                global_weights = copy.deepcopy(global_model.state_dict())
                global_weights_keys = list(global_weights.keys())
                
                for key_idx in range(len(global_weights_keys)):
                    key = global_weights_keys[key_idx]
                    global_weights[key] -= float(args.learning_rate)*global_gradients[key_idx].to(device)
                del global_gradients
            
            # 2.3 load new global weights
            global_model.load_state_dict(global_weights)

            # 3. Calculate avg validation accuracy/uar over all selected users at every epoch
            validation_acc, validation_uar, validation_loss, local_num_sampels = [], [], [], []
            # 3.1 Iterate each client at the current global round, calculate the performance
            for idx in idxs_speakers:
                speaker_id = speaker_list[idx]

                dataset_train = DatasetGenerator(train_speaker_dict[speaker_id], train_val_idx_dict[speaker_id]['val'])
                val_dataloaders = DataLoader(dataset_train, batch_size=20, num_workers=0, shuffle=False)
                val_result_dict = trainer.validate(copy.deepcopy(global_model), dataloaders=val_dataloaders, verbose=False)
                
                # save validation accuracy, uar, and loss
                local_num_sampels.append(int(val_result_dict[0]['val_size']))
                validation_acc.append(float(val_result_dict[0]['val_acc']))
                validation_uar.append(float(val_result_dict[0]['val_uar']))
                validation_loss.append(float(val_result_dict[0]['val_loss']))
                del val_dataloaders
            
            # 3.2 Re-Calculate weigted performance scores
            validate_result = {}
            weighted_acc, weighted_rec = 0, 0
            total_num_samples = np.sum(local_num_sampels)
            for acc_idx in range(len(validation_acc)):
                weighted_acc += validation_acc[acc_idx] * (local_num_sampels[acc_idx] / total_num_samples)
                weighted_rec += validation_uar[acc_idx] * (local_num_sampels[acc_idx] / total_num_samples)
            validate_result['val_acc'], validate_result['val_uar'] = weighted_acc, weighted_rec
            validate_result['loss'] = np.mean(validation_loss)

            print('| Global Round validation : {} | \tacc: {:.2f}% | \tuar: {:.2f}% | \tLoss: {:.6f}\n'.format(
                        epoch, weighted_acc*100, weighted_rec*100, np.mean(validation_loss)))
            
            # 4. Perform the test on holdout set
            test_result_dict = trainer.test(copy.deepcopy(global_model), dataloaders=test_dataloaders, verbose=False)
            test_result_dict[0]['conf'] = trainer.model.test_conf

            # 5. Save the results for later
            result_dict[epoch] = {}
            result_dict[epoch]['train'] = {}
            result_dict[epoch]['train']['loss'] = sum(local_losses) / len(local_losses)
            result_dict[epoch]['validate'], result_dict[epoch]['test'] = validate_result, test_result_dict[0]
            
            if validate_result['val_uar'] > best_score and epoch > 100:
                best_val_acc, best_score = validate_result['val_acc'], validate_result['val_uar']
                final_acc, final_recall = test_result_dict[0]['test_acc'], test_result_dict[0]['test_uar']
                final_confusion = test_result_dict[0]['conf']
                best_epoch, best_dict = epoch, test_result_dict[0].copy()
                best_model = deepcopy(global_model.state_dict())
            
            if epoch > 100:
                # log results
                print('best epoch %d, best final acc %.2f, best val acc %.2f' % (best_epoch, final_acc*100, best_val_acc*100))
                print('best epoch %d, best final rec %.2f, best val rec %.2f' % (best_epoch, final_recall*100, best_score*100))
                print(best_dict['conf'])
        
        # Performance save code
        row_df = save_result(save_row_str, best_dict['test_acc'], best_dict['test_uar'], best_epoch, args.dataset)
        save_result_df = pd.concat([save_result_df, row_df])
        save_result_df.to_csv(str(model_result_csv_path.joinpath('private_'+ str(args.dataset) + '.csv')))
        
        # Save best model and training history
        torch.save(best_model, str(model_result_path.joinpath('model.pt')))
        f = open(str(model_result_path.joinpath('results.pkl')), "wb")
        pickle.dump(result_dict, f)
        f.close()

    # Calculate the average of the 5-fold experiments
    tmp_df = save_result_df.loc[save_result_df['dataset'] == args.dataset]
    row_df = save_result('average', np.mean(tmp_df['acc']), np.mean(tmp_df['uar']), best_epoch, args.dataset)
    save_result_df = pd.concat([save_result_df, row_df])
    save_result_df.to_csv(str(model_result_csv_path.joinpath('private_'+ str(args.dataset) + '.csv')))
