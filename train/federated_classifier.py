import torch
import torch.nn as nn
import argparse
import torch.multiprocessing
from copy import deepcopy
from torch.utils.data import DataLoader, dataset
from re import L

import numpy as np
import torch
import pickle
from pathlib import Path
import pandas as pd
import copy

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
                       
# define feature len mapping
feature_len_dict = {'emobase': 988, 'ComParE': 6373, 'wav2vec': 9216, 
                    'apc': 512, 'distilhubert': 768, 'tera': 768, 'wav2vec2': 768,
                    'decoar2': 768, 'cpc': 256, 'audio_albert': 768, 
                    'mockingjay': 768, 'npc': 512, 'vq_apc': 512, 'vq_wav2vec': 512}

def save_result_df(save_index, acc, rec, best_epoch, dataset):
    row_df = pd.DataFrame(index=[save_index])
    row_df['acc'] = acc
    row_df['uar'] = rec
    row_df['epoch'] = best_epoch
    row_df['dataset'] = dataset
    return row_df


def test(model, device, data_loader, loss, epoch, args, pred='emotion'):
    model.eval()
    predict_dict, truth_dict = {}, {}
    
    predict_dict[args.dataset], truth_dict[args.dataset] = [], []
    for dataset in dataset_list:
        predict_dict[dataset], truth_dict[dataset] = [], []

    for batch_idx, (features, labels, dataset) in enumerate(data_loader):
        features, labels_arr = features.to(device), labels.to(device)
        features = features.float()
        
        preds = model(features)
        m = nn.Softmax(dim=1)
        preds = m(preds)
        
        # Save predictions
        prediction = np.argmax(preds.detach().cpu().numpy()[0])
        predict_dict[args.dataset].append(prediction)
        predict_dict[dataset[0]].append(prediction)

        # Save truth labels
        truth_dict[args.dataset].append(labels_arr.detach().cpu().numpy()[0])
        truth_dict[dataset[0]].append(labels_arr.detach().cpu().numpy()[0])
    
    # Calculate accuracy, uar, and loss using ReturnResultDict
    result_dict = {}
    for dataset in dataset_list:
        tmp_result_dict = ReturnResultDict(truth_dict, predict_dict, dataset, args.pred, mode='test', loss=None, epoch=epoch)
        result_dict[dataset] = tmp_result_dict[dataset].copy()
        
    tmp_result_dict = ReturnResultDict(truth_dict, predict_dict, args.dataset, args.pred, mode='test', loss=None, epoch=epoch)
    result_dict[args.dataset] = tmp_result_dict[args.dataset].copy()
    
    return result_dict


if __name__ == '__main__':

    torch.cuda.empty_cache() 
    torch.multiprocessing.set_sharing_strategy('file_system')

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--feature_type', default='mel_spec')
    parser.add_argument('--input_channel', default=1)
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--learning_rate', default=0.05)
    parser.add_argument('--input_spec_size', default=128)
    parser.add_argument('--batch_size', default=20)
    parser.add_argument('--aug', default=None)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--num_epochs', default=150)
    parser.add_argument('--local_epochs', default=1)
    parser.add_argument('--norm', default='min_max')
    parser.add_argument('--win_len', default=200)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--shift', default=1)
    parser.add_argument('--model_type', default='fed_sgd')
    parser.add_argument('--pred', default='emotion')

    args = parser.parse_args()
    shift = 'shift' if int(args.shift) == 1 else 'without_shift'
    setup_seed(8)

    root_path = Path('/media/data/projects/speech-privacy')
    pred = 'affect' if args.pred == 'arousal' or args.pred == 'valence' else 'emotion'
    preprocess_path = root_path.joinpath('federated_learning', args.feature_type, pred)
    
    save_result_df = pd.DataFrame()
    dataset_list = args.dataset.split('_')
    
    # We perform 5 fold experiments
    for fold_idx in range(0, 5):
        torch.cuda.empty_cache()
        torch.manual_seed(8)
        
        # save folder details
        save_row_str = 'fold'+str(int(fold_idx+1))
        row_df = pd.DataFrame(index=[save_row_str])
        
        model_setting_str = 'local_epoch_'+str(args.local_epochs) if args.model_type == 'fed_avg' else 'local_epoch_1'
        model_setting_str += '_dropout_' + str(args.dropout).replace('.', '')
        model_setting_str += '_lr_' + str(args.learning_rate)[2:]
        
        model_result_path = root_path.joinpath('federated_model_params', args.model_type, args.pred, args.feature_type, args.dataset, model_setting_str, save_row_str)
        Path.mkdir(model_result_path, parents=True, exist_ok=True)
        
        train_speaker_dict = {}
        train_speaker_key_dict = {}
        final_test_dict = {}

        # prepare the data for the training
        for dataset in dataset_list:
            with open(preprocess_path.joinpath(dataset, save_row_str, 'training_'+args.norm+'.pkl'), 'rb') as f:
                train_dict = pickle.load(f)
            with open(preprocess_path.joinpath(dataset, save_row_str, 'validation_'+args.norm+'.pkl'), 'rb') as f:
                validate_dict = pickle.load(f)
            with open(preprocess_path.joinpath(dataset, save_row_str, 'test_'+args.norm+'.pkl'), 'rb') as f:
                test_dict = pickle.load(f)
            
            for tmp_dict in [train_dict, validate_dict, test_dict]:
                for key in tmp_dict:
                    tmp_dict[key]['dataset'] = dataset

            # we remake the data dict per speaker for the ease of local training
            tmp_train_speaker_key_dict = {}
            for tmp_dict in [train_dict, validate_dict]:
                for key in tmp_dict:
                    speaker_id = str(tmp_dict[key]['speaker_id'])
                    if speaker_id not in train_speaker_key_dict: 
                        train_speaker_key_dict[speaker_id] = []
                        tmp_train_speaker_key_dict[speaker_id] = []
                    train_speaker_key_dict[speaker_id].append(key)
                    tmp_train_speaker_key_dict[speaker_id].append(key)
            # test set will be the same
            for key in test_dict:
                final_test_dict[key] = test_dict[key].copy()

            # in federated setting that the norm validation dict will be based on local client data
            # so we combine train_dict and validate_dict in centralized setting
            # then choose certain amount data per client as local validation set
            if dataset == 'crema-d':
                for tmp_dict in [train_dict, validate_dict]:
                    for key in tmp_dict:
                        speaker_id = str(tmp_dict[key]['speaker_id'])
                        if speaker_id not in train_speaker_dict: train_speaker_dict[speaker_id] = {}
                        train_speaker_dict[speaker_id][key] = tmp_dict[key].copy()
            else:
                # we want to divide speaker data if the dataset is iemocap or msp-improv to increase client size
                for speaker_id in tmp_train_speaker_key_dict:
                    # in iemocap and msp-improv
                    # we spilit each speaker data into 10 parts in order to create more clients
                    idx_array = np.random.permutation(len(tmp_train_speaker_key_dict[speaker_id]))
                    key_list = tmp_train_speaker_key_dict[speaker_id]
                    for i in range(10):
                        # we randomly pick 10% of data
                        idxs_train = idx_array[int(0.1*i*len(idx_array)):int(0.1*(i+1)*len(idx_array))]
                        for idx in idxs_train:
                            key = key_list[idx]
                            if speaker_id+'_'+str(i) not in train_speaker_dict: train_speaker_dict[speaker_id+'_'+str(i)] = {}
                            if key in train_dict:
                                train_speaker_dict[speaker_id+'_'+str(i)][key] = train_dict[key].copy()
                            else:
                                train_speaker_dict[speaker_id+'_'+str(i)][key] = validate_dict[key].copy()
        
        num_of_speakers = len(train_speaker_dict)
        speaker_list = list(set(train_speaker_dict.keys()))
        
        # Model related
        device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available(): print('GPU available, use GPU')

        global_model = dnn_classifier(pred='emotion', input_spec=feature_len_dict[args.feature_type], dropout=float(args.dropout))
        global_model = global_model.to(device)

        # define test data loader
        test_keys = list(final_test_dict.keys())
        dataset_test = DatasetSplit(final_test_dict, test_keys)
        dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=0, shuffle=False)

        # copy weights
        global_weights = global_model.state_dict()
        criterion = nn.CrossEntropyLoss().to(device)

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=10, verbose=True)
        
        model_parameters = filter(lambda p: p.requires_grad, global_model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(params)
        
        best_val_recall, final_recall, best_epoch, final_confusion = 0, 0, 0, 0
        best_val_acc, final_acc = 0, 0
        result_dict = {}

        train_loss, validation_accuracy = [], []

        # We save the utterance keys used for training and validation per speaker (client)
        train_validation_idx_dict = {}
        for speaker_id in train_speaker_dict:
            idx_array = np.random.permutation(len(train_speaker_dict[speaker_id]))
            train_validation_idx_dict[speaker_id] = {}
            train_validation_idx_dict[speaker_id]['train'] = idx_array[:int(0.8*len(idx_array))]
            train_validation_idx_dict[speaker_id]['val'] = idx_array[int(0.8*len(idx_array)):]
        
        # Training steps
        for epoch in range(int(args.num_epochs)):
            
            torch.cuda.empty_cache()
            
            # we choose 10% of clients in training
            m = max(int(0.1 * num_of_speakers), 1)
            idxs_speakers = np.random.choice(range(num_of_speakers), m, replace=False)
            
            # define list varibles that saves the weights, loss, num_sample, etc.
            local_weights, local_gradients, local_losses, local_num_sampels = [], [], [], []
            gradient_hist_dict = {}

            # 1. Local training, return weights in fed_avg, return gradients in fed_sgd
            for idx in idxs_speakers:
                speaker_id = speaker_list[idx]
                # args, dataset, device, criterion
                print('speaker id %s' % speaker_id)

                # 1.1 Local training
                local_model = LocalUpdate(args=args, dataset=train_speaker_dict[speaker_id], 
                                          idxs=list(train_speaker_dict[speaker_id].keys()), device=device, 
                                          criterion=criterion, model_type=args.model_type,
                                          train_validation_idx_dict=train_validation_idx_dict[speaker_id])
                if args.model_type == 'fed_avg':
                    w, loss, num_sampels = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                    local_weights.append(copy.deepcopy(w))
                else:
                    g, loss, num_sampels = local_model.update_gradients(model=copy.deepcopy(global_model), global_round=epoch)
                    local_gradients.append(copy.deepcopy(g))
                local_losses.append(copy.deepcopy(loss))
                local_num_sampels.append(num_sampels)

                del local_model
                
                # 1.2 calculate and save the raw gradients or pseudo gradients
                if args.model_type == 'fed_avg':
                    # 'fake' gradients saving code
                    # iterate all layers in the classifier model
                    gradients = []
                    for key in copy.deepcopy(global_model).state_dict():
                        original_params = copy.deepcopy(global_model).state_dict()[key].detach().clone()
                        update_params = copy.deepcopy(w)[key].detach().clone()
                        
                        # calculate how many updates per local epoch 
                        local_update_per_epoch = int(len(train_speaker_dict[speaker_id]) * 0.8 / int(args.batch_size)) + 1
                        
                        # calculate 'fake' gradients
                        tmp_gradients = (original_params - update_params)/(float(args.learning_rate)*local_update_per_epoch*int(args.local_epochs))
                        tmp_gradients = tmp_gradients.cpu().numpy()
                        gradients.append(tmp_gradients)
                        del tmp_gradients, original_params, update_params
                else:
                    gradients = copy.deepcopy(g)
                    for i in range(len(gradients)):
                        gradients[i] = gradients[i].cpu().numpy()
                
                # 1.3 save the attack features
                tmp_key = list(train_speaker_dict[speaker_id].keys())[0]
                gradient_hist_dict[speaker_id] = {}
                gradient_hist_dict[speaker_id]['gradient'] = gradients
                gradient_hist_dict[speaker_id]['gender'] = train_speaker_dict[speaker_id][tmp_key]['gender']
            # 1.4 dump the gradients for the later usage
            f = open(str(model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl')), "wb")
            pickle.dump(gradient_hist_dict, f)
            f.close()

            # 2. global model updates
            total_num_samples = np.sum(local_num_sampels)
            if args.model_type == 'fed_avg':
                # 2.1 average global weights
                global_weights = average_weights(local_weights, local_num_sampels)
            else:
                # 2.1 average global gradients
                global_gradients = average_gradients(local_gradients, local_num_sampels)
                # 2.2 update global weights
                global_weights = copy.deepcopy(global_model.state_dict())
                global_weights_keys = list(global_weights.keys())
                
                for key_idx in range(len(global_weights_keys)):
                    key = global_weights_keys[key_idx]
                    global_weights[key] -= float(args.learning_rate)*global_gradients[key_idx]
                del global_gradients
            
            # 2.3 load new global weights
            global_model.load_state_dict(global_weights)
            
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # 3. Calculate avg validation accuracy/uar over all selected users at every epoch
            validation_acc, validation_uar, validation_loss, local_num_sampels = [], [], [], []
            validate_result = {}
            global_model.eval()

            # 3.1 Iterate each client at the current global round, calculate the performance
            for idx in idxs_speakers:
                speaker_id = speaker_list[idx]
                local_model = LocalUpdate(args=args, dataset=train_speaker_dict[speaker_id], 
                                          idxs=list(train_speaker_dict[speaker_id].keys()), device=device, 
                                          criterion=criterion, model_type=args.model_type,
                                          train_validation_idx_dict=train_validation_idx_dict[speaker_id])
                acc_score, rec_score, loss, num_sampels = local_model.inference(model=global_model)
                
                # save validation accuracy, uar, and loss
                local_num_sampels.append(num_sampels)
                validation_acc.append(acc_score)
                validation_uar.append(rec_score)
                validation_loss.append(loss)

                del local_model
            
            # 3.2 Re-Calculate weigted performance scores
            weighted_acc, weighted_rec = 0, 0
            total_num_samples = np.sum(local_num_sampels)
            for acc_idx in range(len(validation_acc)):
                weighted_acc += validation_acc[acc_idx] * (local_num_sampels[acc_idx] / total_num_samples)
                weighted_rec += validation_uar[acc_idx] * (local_num_sampels[acc_idx] / total_num_samples)
            validate_result['acc'] = weighted_acc
            validate_result['rec'] = weighted_rec
            validate_result['loss'] = np.mean(validation_loss)
            
            # 4. Perform the test on holdout set
            test_result_dict = test(global_model, device, dataloader_test, criterion, epoch, args)

            # 5. Save the results for later
            result_dict[epoch] = {}
            result_dict[epoch]['train'] = {}
            result_dict[epoch]['train']['loss'] = {}
            result_dict[epoch]['validate'] = validate_result
            result_dict[epoch]['test'] = test_result_dict
            
            if validate_result['rec'] > best_val_recall:
                best_val_acc, best_val_recall = validate_result['acc'], validate_result['rec']
                final_acc = test_result_dict[args.dataset]['acc'][args.pred]
                final_recall = test_result_dict[args.dataset]['rec'][args.pred]
                final_confusion = test_result_dict[args.dataset]['conf'][args.pred]
                best_epoch, best_dict = epoch, test_result_dict.copy()
                best_model = deepcopy(global_model.state_dict())
            
            # Some print out
            print('best epoch %d, best final acc %.2f, best val acc %.2f' % (best_epoch, final_acc*100, best_val_acc*100))
            print('best epoch %d, best final rec %.2f, best val rec %.2f' % (best_epoch, final_recall*100, best_val_recall*100))
            print(best_dict[args.dataset]['conf'][args.pred])

        # Performance save code
        row_df = save_result_df(save_row_str, best_dict[args.dataset]['acc'][args.pred], best_dict[args.dataset]['rec'][args.pred], best_epoch, args.dataset)
        save_result_df = pd.concat([save_result_df, row_df])
        if len(dataset_list) > 1:
            for dataset in dataset_list:
                row_df = save_result_df(save_row_str, best_dict[dataset]['acc'][args.pred], best_dict[dataset]['rec'][args.pred], best_epoch, dataset)
                save_result_df = pd.concat([save_result_df, row_df])

        model_result_csv_path = Path.cwd().parents[0].joinpath('results', args.pred, args.model_type, args.feature_type, model_setting_str)
        Path.mkdir(model_result_csv_path, parents=True, exist_ok=True)
        save_result_df.to_csv(str(model_result_csv_path.joinpath('private_'+ str(args.dataset) + '.csv')))
        
        # Save best model and training history
        torch.save(best_model, str(model_result_path.joinpath('model.pt')))
        f = open(str(model_result_path.joinpath('results.pkl')), "wb")
        pickle.dump(result_dict, f)
        f.close()

    # Calculate the average of the 5-fold experiments
    tmp_df = save_result_df.loc[save_result_df['dataset'] == args.dataset]
    row_df = save_result_df('average', np.mean(tmp_df['acc']), np.mean(tmp_df['uar']), best_epoch, args.dataset)
    save_result_df = pd.concat([save_result_df, row_df])
    
    if len(dataset_list) > 1:
        for dataset in dataset_list:
            tmp_df = save_result_df.loc[save_result_df['dataset'] == dataset]
            row_df = save_result_df('average', np.mean(tmp_df['acc']), np.mean(tmp_df['uar']), best_epoch, dataset)
            save_result_df = pd.concat([save_result_df, row_df])
    save_result_df.to_csv(str(model_result_csv_path.joinpath('private_'+ str(args.dataset) + '.csv')))
