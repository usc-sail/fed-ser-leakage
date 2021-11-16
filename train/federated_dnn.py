import torch
import torch.nn as nn
import argparse
import torch.multiprocessing
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
from collections import Counter
from torch.utils.data import DataLoader, dataset
from torchvision import transforms
from re import L

import numpy as np
import torch
import pickle
from pathlib import Path
import pandas as pd
import copy
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'model'))
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'utils'))

from training_tools import EarlyStopping, SpeechDataGenerator, ReturnResultDict
from training_tools import speech_collate, setup_seed
from baseline_models import dnn_classifier
from update import LocalUpdate, average_weights, DatasetSplit
from sklearn.metrics.pairwise import cosine_similarity

import pdb

emo_dict = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
affect_dict = {'low': 0, 'med': 1, 'high': 2}

gender_dict = {'F': 0, 'M': 1}
speaker_id_arr_dict = {'msp-improv': np.arange(0, 12, 1), 
                       'crema-d': np.arange(1001, 1092, 1),
                       'iemocap': np.arange(0, 10, 1)}
feature_len_dict = {'emobase': 988, 'ComParE': 6373, 'wav2vec': 9216}

def create_folder(folder):
    if Path.exists(folder) is False: Path.mkdir(folder)


def test(model, device, data_loader, loss, epoch, args, pred='emotion'):
    model.eval()
    predict_dict, truth_dict = {}, {}
    
    predict_dict[args.dataset] = []
    truth_dict[args.dataset] = []
    for dataset in dataset_list:
        predict_dict[dataset] = []
        truth_dict[dataset] = []
    result_dict = {}

    for batch_idx, (features, labels, dataset) in enumerate(data_loader):
        features, labels = features.to(device), labels.to(device)
        features = features.float()

        labels_emo = labels.to(device)
        labels_arr = labels_emo
        
        preds = model(features)

        m = nn.Softmax(dim=1)
        preds = m(preds)
        prediction = np.argmax(preds.detach().cpu().numpy()[0])
        predict_dict[args.dataset].append(prediction)
        predict_dict[dataset[0]].append(prediction)

        truth_dict[args.dataset].append(labels_arr.detach().cpu().numpy()[0])
        truth_dict[dataset[0]].append(labels_arr.detach().cpu().numpy()[0])
    
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
    parser.add_argument('--learning_rate', default=0.0001)
    parser.add_argument('--input_spec_size', default=128)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--aug', default=None)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--num_epochs', default=100)
    parser.add_argument('--local_epochs', default=1)
    parser.add_argument('--norm', default='min_max')
    parser.add_argument('--win_len', default=200)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--shift', default=1)
    parser.add_argument('--model_type', default='dnn')
    parser.add_argument('--pred', default='emotion')

    args = parser.parse_args()
    shift = 'shift' if int(args.shift) == 1 else 'without_shift'
    
    setup_seed(8)

    root_path = Path('/media/data/projects/speech-privacy')
    pred = 'affect' if args.pred == 'arousal' or args.pred == 'valence' else 'emotion'
    preprocess_path = root_path.joinpath('federated_learning', args.feature_type, pred)
    
    save_result_df = pd.DataFrame()
    dataset_list = args.dataset.split('_')
    
    for fold_idx in range(0, 5):
        torch.cuda.empty_cache()
        torch.manual_seed(8)
        
        # save folder details
        save_row_str = 'fold'+str(int(fold_idx+1))
        row_df = pd.DataFrame(index=[save_row_str])

        model_setting_str = 'local_epoch_'+str(args.local_epochs)
        model_setting_str += '_dropout_' + str(args.dropout).replace('.', '')
        
        model_result_path = root_path.joinpath('federated_model_params', args.model_type, args.pred, args.feature_type, args.dataset, model_setting_str, save_row_str)
        Path.mkdir(model_result_path, parents=True, exist_ok=True)
        
        train_speaker_dict = {}
        train_speaker_key_dict = {}
        final_test_dict = {}
        
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

            # we remake the data dict per speaker
            tmp_train_speaker_key_dict = {}
            for tmp_dict in [train_dict, validate_dict]:
                for key in tmp_dict:
                    speaker_id = str(tmp_dict[key]['speaker_id'])
                    if speaker_id not in train_speaker_key_dict: 
                        train_speaker_key_dict[speaker_id] = []
                        tmp_train_speaker_key_dict[speaker_id] = []
                    train_speaker_key_dict[speaker_id].append(key)
                    tmp_train_speaker_key_dict[speaker_id].append(key)
            
            for key in test_dict:
                final_test_dict[key] = test_dict[key].copy()

            if dataset == 'crema-d':
                for tmp_dict in [train_dict, validate_dict]:
                    for key in tmp_dict:
                        speaker_id = str(tmp_dict[key]['speaker_id'])
                        if speaker_id not in train_speaker_dict: train_speaker_dict[speaker_id] = {}
                        train_speaker_dict[speaker_id][key] = tmp_dict[key].copy()
            else:
                # we want to divide speaker data
                for speaker_id in tmp_train_speaker_key_dict:
                    idx_array = np.random.permutation(len(tmp_train_speaker_key_dict[speaker_id]))
                    key_list = tmp_train_speaker_key_dict[speaker_id]
                    for i in range(10):
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
        device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available(): print('GPU available, use GPU')
        
        # so if it is trained using adv dataset or service provider dataset
        global_model = dnn_classifier(pred='emotion', input_spec=feature_len_dict[args.feature_type])
        global_model = global_model.to(device)

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
        train_validation_idx_dict = {}
        for speaker_id in train_speaker_dict:
            train_validation_idx_dict[speaker_id] = {}

            idx_array = np.random.permutation(len(train_speaker_dict[speaker_id]))
            idxs_train = idx_array[:int(0.8*len(idx_array))]
            idxs_val = idx_array[int(0.8*len(idx_array)):]
            train_validation_idx_dict[speaker_id]['train'] = idxs_train
            train_validation_idx_dict[speaker_id]['val'] = idxs_val
        
        
        best_dict = {}
        for epoch in range(args.num_epochs):
            frac = 0.1
            m = max(int(frac * num_of_speakers), 1)
            idxs_speakers = np.random.choice(range(num_of_speakers), m, replace=False)
            torch.cuda.empty_cache()

            local_weights, local_losses = [], []

            weight_hist_dict = {}
            for idx in idxs_speakers:
                speaker_id = speaker_list[idx]
                # args, dataset, device, criterion
                print('speaker id %s' % speaker_id)
                local_model = LocalUpdate(args=args, dataset=train_speaker_dict[speaker_id], 
                                          idxs=list(train_speaker_dict[speaker_id].keys()), device=device, 
                                          criterion=criterion, model_type=args.model_type,
                                          train_validation_idx_dict=train_validation_idx_dict[speaker_id])
                w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                del local_model
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
                
                # pdb.set_trace()
                gradients = []
                for key in copy.deepcopy(global_model).state_dict():
                    original_params = copy.deepcopy(global_model).state_dict()[key].detach().clone()
                    update_params = copy.deepcopy(w)[key].detach().clone()
                    local_update_per_batch = int(len(train_speaker_dict[speaker_id]) * 0.8 / int(args.batch_size)) + 1
                    tmp_gradients = (original_params - update_params)/(float(args.learning_rate)*local_update_per_batch*int(args.local_epochs))
                    tmp_gradients = tmp_gradients.cpu().numpy()
                    gradients.append(tmp_gradients)
                    del tmp_gradients, original_params, update_params
                
                tmp_key = list(train_speaker_dict[speaker_id].keys())[0]
                
                weight_hist_dict[speaker_id] = {}
                weight_hist_dict[speaker_id]['gradient'] = gradients
                weight_hist_dict[speaker_id]['gender'] = train_speaker_dict[speaker_id][tmp_key]['gender']

            # average global weights
            global_weights = average_weights(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_rec, list_loss = [], [], []
            validate_result = {}
            global_model.eval()
            for idx in idxs_speakers:
                speaker_id = speaker_list[idx]
                local_model = LocalUpdate(args=args, dataset=train_speaker_dict[speaker_id], 
                                          idxs=list(train_speaker_dict[speaker_id].keys()), device=device, 
                                          criterion=criterion, model_type=args.model_type,
                                          train_validation_idx_dict=train_validation_idx_dict[speaker_id])
                acc_score, rec_score, loss = local_model.inference(model=global_model)
                del local_model
                
                list_acc.append(acc_score)
                list_rec.append(rec_score)
                list_loss.append(loss)
            
            validate_result['acc'] = np.mean(list_acc)
            validate_result['rec'] = np.mean(list_rec)
            validate_result['loss'] = np.mean(list_loss)
            
            # perform the training, validate, and test
            # validate_result = train(global_cnn_model, device, dataloader_val, criterion, epoch, args, mode='validate')
            test_result_dict = test(global_model, device, dataloader_test, criterion, epoch, args)

            # save the results for later
            result_dict[epoch] = {}
            result_dict[epoch]['validate'] = validate_result
            result_dict[epoch]['test'] = test_result_dict
            
            if validate_result['acc'] > best_val_acc:
                best_val_acc = validate_result['acc']
                best_val_recall = validate_result['rec']
                final_acc = test_result_dict[args.dataset]['acc'][args.pred]
                final_recall = test_result_dict[args.dataset]['rec'][args.pred]
                final_confusion = test_result_dict[args.dataset]['conf'][args.pred]
                best_epoch = epoch
                best_model = deepcopy(global_model.state_dict())
                best_dict = test_result_dict.copy()
            
            print('best epoch %d, best final acc %.2f, best val acc %.2f' % (best_epoch, final_acc*100, best_val_acc*100))
            print('best epoch %d, best final rec %.2f, best val rec %.2f' % (best_epoch, final_recall*100, best_val_recall*100))
            # pdb.set_trace()
            print(test_result_dict[args.dataset]['conf'][args.pred])

            f = open(str(model_result_path.joinpath('weights_hist_'+str(epoch)+'.pkl')), "wb")
            pickle.dump(weight_hist_dict, f)
            f.close()

        row_df = pd.DataFrame(index=[save_row_str])
        row_df['acc'] = best_dict[args.dataset]['acc'][args.pred]
        row_df['rec'] = best_dict[args.dataset]['rec'][args.pred]
        row_df['epoch'] = best_epoch
        row_df['dataset'] = args.dataset
        save_result_df = pd.concat([save_result_df, row_df])
        
        for dataset in dataset_list:
            row_df = pd.DataFrame(index=[save_row_str])
            row_df['acc'] = best_dict[dataset]['acc'][args.pred]
            row_df['rec'] = best_dict[dataset]['rec'][args.pred]
            row_df['epoch'] = best_epoch
            row_df['dataset'] = dataset
            save_result_df = pd.concat([save_result_df, row_df])
        
        torch.save(best_model, str(model_result_path.joinpath('model.pt')))
        f = open(str(model_result_path.joinpath('results.pkl')), "wb")
        pickle.dump(result_dict, f)
        f.close()

        # pdb.set_trace()
        save_result_df.to_csv(str(root_path.joinpath('federated_model_params', args.model_type, args.pred, args.feature_type, args.dataset, model_setting_str, 'result.csv')))

