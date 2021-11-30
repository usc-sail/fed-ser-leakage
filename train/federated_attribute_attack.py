import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import torch.multiprocessing
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy

import numpy as np
import torch
import pickle
from pathlib import Path
import pandas as pd

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'model'))
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'utils'))

from training_tools import ReturnResultDict, EarlyStopping
from training_tools import setup_seed
from baseline_models import attack_model

from sklearn.model_selection import train_test_split

import pdb

emo_dict = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
affect_dict = {'low': 0, 'med': 1, 'high': 2}

gender_dict = {'F': 0, 'M': 1}
speaker_id_arr_dict = {'msp-improv': np.arange(0, 12, 1), 
                       'crema-d': np.arange(1001, 1092, 1),
                       'iemocap': np.arange(0, 10, 1)}
leak_layer_dict = {'full': ['w0', 'b0', 'w1', 'b1', 'w2', 'b2'],
                   'first': ['w0', 'b0'], 'second': ['w1', 'b1'], 'last': ['w2', 'b2']}

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
        if args.leak_layer == 'last':
            tmp_data = (self.data_dict[data_file_str]['w2'] - weight_norm_mean_dict['w2']) / (weight_norm_std_dict['w2'] + 0.00001)
            w2 = torch.from_numpy(np.ascontiguousarray(tmp_data))
            tmp_data = (self.data_dict[data_file_str]['b2'] - weight_norm_mean_dict['b2']) / (weight_norm_std_dict['b2'] + 0.00001)
            b2 = torch.from_numpy(np.ascontiguousarray(tmp_data))
        elif args.leak_layer == 'second':
            tmp_data = (self.data_dict[data_file_str]['w1'] - weight_norm_mean_dict['w1']) / (weight_norm_std_dict['w1'] + 0.00001)
            w1 = torch.from_numpy(np.ascontiguousarray(tmp_data))
            tmp_data = (self.data_dict[data_file_str]['b1'] - weight_norm_mean_dict['b1']) / (weight_norm_std_dict['b1'] + 0.00001)
            b1 = torch.from_numpy(np.ascontiguousarray(tmp_data))
        else:
            tmp_data = (self.data_dict[data_file_str]['w0'] - weight_norm_mean_dict['w0']) / (weight_norm_std_dict['w0'] + 0.00001)
            w0 = torch.from_numpy(np.ascontiguousarray(tmp_data))
            tmp_data = (self.data_dict[data_file_str]['b0'] - weight_norm_mean_dict['b0']) / (weight_norm_std_dict['b0'] + 0.00001)
            b0 = torch.from_numpy(np.ascontiguousarray(tmp_data))
        gender = gender_dict[self.data_dict[data_file_str]['gender']]
        return w0, w1, w2, b0, b1, b2, gender


def test(model, device, data_loader, epoch):
    model.eval()
    predict_dict, truth_dict = {}, {}
    
    predict_dict[args.dataset] = []
    truth_dict[args.dataset] = []
        
    loss_list = []
    for batch_idx, (w0, w1, w2, b0, b1, b2, labels) in enumerate(data_loader):
    
        if args.leak_layer == 'first':
            w0 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w0]))
            b0 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in b0]))
            w0, b0 = w0.to(device), b0.to(device)
            w0, b0 = w0.float(), b0.float()
            w0 = w0.unsqueeze(dim=1)
        elif args.leak_layer == 'second':
            w1 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w1]))
            b1 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in b1]))
            w1, b1 = w1.to(device), b1.to(device)
            w1, b1 = w1.float(), b1.float()
            w1 = w1.unsqueeze(dim=1)
        else:
            w2 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w2]))
            b2 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in b2]))
            w2, b2 = w2.to(device), b2.to(device)
            w2, b2 = w2.float(), b2.float()
            w2 = w2.unsqueeze(dim=1)
        
        labels = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in labels]))
        labels = labels.to(device)

        # Inference
        preds = model(w0, w1, w2, b0, b1, b2)
        _, predictions = torch.max(preds, 1)

        # Prediction
        # pdb.set_trace()
        predictions = predictions.view(-1)
        for pred_idx in range(len(preds)):
            # predict_dict[args.dataset].append(predictions.detach().cpu().numpy()[pred_idx])
            predict_dict[args.dataset].append(predictions.detach().cpu().numpy()[pred_idx])
            truth_dict[args.dataset].append(labels.detach().cpu().numpy()[pred_idx])
        
        del w0, w1, w2, b0, b1, b2
    
    tmp_result_dict = ReturnResultDict(truth_dict, predict_dict, args.dataset, args.pred, mode='test', loss=None, epoch=epoch)
    return tmp_result_dict


def train(model, device, data_loader, epoch, mode='train'):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    predict_dict, truth_dict = {}, {}
    
    predict_dict[args.adv_dataset] = []
    truth_dict[args.adv_dataset] = []
        
    loss_list = []
    for batch_idx, (w0, w1, w2, b0, b1, b2, labels) in enumerate(data_loader):
        if args.leak_layer == 'first':
            w0 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w0]))
            b0 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in b0]))
            w0, b0 = w0.to(device), b0.to(device)
            w0, b0 = w0.float(), b0.float()
            w0 = w0.unsqueeze(dim=1)
        elif args.leak_layer == 'second':
            w1 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w1]))
            b1 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in b1]))
            w1, b1 = w1.to(device), b1.to(device)
            w1, b1 = w1.float(), b1.float()
            w1 = w1.unsqueeze(dim=1)
        else:
            w2 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w2]))
            b2 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in b2]))
            w2, b2 = w2.to(device), b2.to(device)
            w2, b2 = w2.float(), b2.float()
            w2 = w2.unsqueeze(dim=1)
        
        labels = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in labels]))
        labels = labels.to(device)

        # Inference
        preds = model(w0, w1, w2, b0, b1, b2)
        batch_loss = criterion(preds, labels.squeeze())
        
        if mode == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        loss_list.append(batch_loss.item())

        # Prediction
        _, predictions = torch.max(preds, 1)
        predictions = predictions.view(-1)
        for pred_idx in range(len(preds)):
            predict_dict[args.adv_dataset].append(predictions.detach().cpu().numpy()[pred_idx])
            truth_dict[args.adv_dataset].append(labels.detach().cpu().numpy()[pred_idx])
        if batch_idx % 50 == 0:
            print('%s epoch: %d, %.2f' % (mode, epoch, batch_idx / len(data_loader) * 100))
        del w0, w1, w2, b0, b1, b2, preds
        torch.cuda.empty_cache()

    if mode != 'train':   
        if args.optimizer == 'adam': scheduler.step(np.mean(loss_list))
    tmp_result_dict = ReturnResultDict(truth_dict, predict_dict, args.adv_dataset, args.pred, mode=mode, loss=np.mean(loss_list), epoch=epoch)
    return tmp_result_dict


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
    parser.add_argument('--shift', default=1)
    parser.add_argument('--model_type', default='fed_sgd')
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--leak_layer', default='full')
    parser.add_argument('--dropout', default=0.2)

    args = parser.parse_args()
    shift = 'shift' if int(args.shift) == 1 else 'without_shift'
    
    setup_seed(8)
    torch.manual_seed(8)

    root_path = Path('/media/data/projects/speech-privacy')
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
        weight_norm_mean_dict[key] = []
        weight_norm_std_dict[key] = []
        weight_sum[key] = 0
        weight_sum_square[key] = 0

    shadow_training_sample_size = 0
    shadow_data_dict = {}
    for shadow_idx in range(0, 5):
        for epoch in range(int(args.num_epochs)):
            adv_federated_model_result_path = root_path.joinpath('federated_model_params', args.model_type, args.pred, args.feature_type, args.adv_dataset, model_setting_str, 'fold'+str(int(shadow_idx+1)))
            weight_file_str = str(adv_federated_model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl'))
            weight_file_list.append(weight_file_str)
            # if shadow_idx == 0 and epoch < 10:
            if epoch % 20 == 0:
                print('reading shadow model %d, epoch %d' % (shadow_idx, epoch))
            with open(weight_file_str, 'rb') as f:
                adv_fed_weight_hist_dict = pickle.load(f)
            for speaker_id in adv_fed_weight_hist_dict:
                gradients = adv_fed_weight_hist_dict[speaker_id]['gradient']
                shadow_training_sample_size += 1
                
                # calculate running stats for computing std and mean
                if args.leak_layer == 'first':
                    weight_sum['w0'] += gradients[0]
                    weight_sum['b0'] += gradients[1]
                    weight_sum_square['w0'] += gradients[0]**2
                    weight_sum_square['b0'] += gradients[1]**2
                    shadow_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id] = {}
                    shadow_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['w0'] = gradients[0]
                    shadow_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['b0'] = gradients[1]
                    shadow_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['gender'] = adv_fed_weight_hist_dict[speaker_id]['gender']
                elif args.leak_layer == 'second':
                    weight_sum['w1'] += gradients[2]
                    weight_sum['b1'] += gradients[3]
                    weight_sum_square['w1'] += gradients[2]**2
                    weight_sum_square['b1'] += gradients[3]**2
                    shadow_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id] = {}
                    shadow_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['w1'] = gradients[2]
                    shadow_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['b1'] = gradients[3]
                    shadow_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['gender'] = adv_fed_weight_hist_dict[speaker_id]['gender']
                else:
                    weight_sum['w2'] += gradients[4]
                    weight_sum['b2'] += gradients[5]
                    weight_sum_square['w2'] += gradients[4]**2
                    weight_sum_square['b2'] += gradients[5]**2
                
                    shadow_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id] = {}
                    shadow_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['w2'] = gradients[4]
                    shadow_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['b2'] = gradients[5]
                    shadow_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['gender'] = adv_fed_weight_hist_dict[speaker_id]['gender']

    for key in leak_layer_dict[args.leak_layer]:
        weight_norm_mean_dict[key] = weight_sum[key] / shadow_training_sample_size
        tmp_data = weight_sum_square[key] / shadow_training_sample_size - (weight_sum[key] / shadow_training_sample_size)**2
        weight_norm_std_dict[key] = np.sqrt(tmp_data)
    
    # train model to infer gender
    train_key_list, validate_key_list = train_test_split(list(shadow_data_dict.keys()), test_size=0.2, random_state=0)

    # so if it is trained using adv dataset or service provider dataset
    model = attack_model(args.leak_layer, args.feature_type)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.model_learning_rate), weight_decay=1e-04, betas=(0.9, 0.98), eps=1e-9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    # define data loader
    dataset_train = WeightDataGenerator(train_key_list, shadow_data_dict)
    dataset_valid = WeightDataGenerator(validate_key_list, shadow_data_dict)

    dataloader_train = DataLoader(dataset_train, batch_size=20, num_workers=0, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=20, num_workers=0, shuffle=False)

    best_val_recall, best_val_acc, best_epoch = 0, 0, 0
    best_conf_array = None

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in range(50):
        print('feature type: %s' % (args.feature_type))
        train_result = train(model, device, dataloader_train, epoch, mode='train')
        validate_result = train(model, device, dataloader_valid, epoch, mode='validate')
        
        if validate_result[args.adv_dataset]['acc'][args.pred] > best_val_acc:
            best_val_acc = validate_result[args.adv_dataset]['acc'][args.pred]
            best_val_recall = validate_result[args.adv_dataset]['rec'][args.pred]
            best_epoch = epoch
            best_model = deepcopy(model.state_dict())
            best_conf_array = validate_result[args.adv_dataset]['conf'][args.pred]
        print('best epoch %d, best val acc %.2f, best val rec %.2f' % (best_epoch, best_val_acc*100, best_val_recall*100))
        print(best_conf_array)

        early_stopping(validate_result[args.adv_dataset]['loss'][args.pred], best_model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    del model
    
    attack_model_result_csv_path = Path.cwd().parents[0].joinpath('results', 'attack', args.leak_layer, args.model_type, args.feature_type, model_setting_str)
    Path.mkdir(attack_model_result_csv_path, parents=True, exist_ok=True)
    torch.save(best_model, str(attack_model_result_csv_path.joinpath('private_' + str(args.dataset) + '_model.pt')))

    # load the evaluation model
    eval_model = attack_model(args.leak_layer, args.feature_type)
    eval_model = eval_model.to(device)
    eval_model.load_state_dict(best_model)

    # we evaluate the attacker performance on service provider training
    for fold_idx in range(0, 5):
        test_list = []
        test_data_dict = {}
        for epoch in range(int(args.num_epochs)):
            torch.cuda.empty_cache()
            save_row_str = 'fold'+str(int(fold_idx+1))
            row_df = pd.DataFrame(index=[save_row_str])
            
            # Model related
            federated_model_result_path = root_path.joinpath('federated_model_params', args.model_type, args.pred, args.feature_type, args.dataset, model_setting_str, 'fold'+str(int(fold_idx+1)))
            weight_file_str = str(federated_model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl'))
            test_list.append(weight_file_str)

            with open(weight_file_str, 'rb') as f:
                test_fed_weight_hist_dict = pickle.load(f)
            for speaker_id in test_fed_weight_hist_dict:
                gradients = test_fed_weight_hist_dict[speaker_id]['gradient']

                if args.leak_layer == 'first':
                    test_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id] = {}
                    test_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['w0'] = gradients[0]
                    test_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['b0'] = gradients[1]
                    test_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['gender'] = test_fed_weight_hist_dict[speaker_id]['gender']

                elif args.leak_layer == 'second':
                    test_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id] = {}
                    test_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['w1'] = gradients[2]
                    test_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['b1'] = gradients[3]
                    test_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['gender'] = test_fed_weight_hist_dict[speaker_id]['gender']
                else:
                    test_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id] = {}
                    test_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['w2'] = gradients[4]
                    test_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['b2'] = gradients[5]
                    test_data_dict[str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id]['gender'] = test_fed_weight_hist_dict[speaker_id]['gender']
        
        dataset_test = WeightDataGenerator(list(test_data_dict.keys()), test_data_dict)
        dataloader_test = DataLoader(dataset_test, batch_size=20, num_workers=0, shuffle=False)

        test_results_dict = test(eval_model, device, dataloader_test, epoch=0)
        row_df['acc'] = test_results_dict[args.dataset]['acc'][args.pred]
        row_df['uar'] = test_results_dict[args.dataset]['rec'][args.pred]
        save_result_df = pd.concat([save_result_df, row_df])

        del dataset_test, dataloader_test
        
    row_df = pd.DataFrame(index=['average'])
    row_df['acc'] = np.mean(save_result_df['acc'])
    row_df['uar'] = np.mean(save_result_df['uar'])
    save_result_df = pd.concat([save_result_df, row_df])

    save_result_df.to_csv(str(attack_model_result_csv_path.joinpath('private_' + str(args.dataset) + '_result.csv')))

