from re import L
from numpy.core.fromnumeric import mean
import torch
from torch.utils.data import DataLoader, dataset
from torchvision import transforms
import torch.nn as nn
import argparse
from torch import optim
import torch.multiprocessing
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

from training_tools import ReturnResultDict
from training_tools import setup_seed
from baseline_models import attack_model
from update import LocalUpdate, average_weights

from sklearn.model_selection import train_test_split

import pdb

emo_dict = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
affect_dict = {'low': 0, 'med': 1, 'high': 2}

gender_dict = {'F': 0, 'M': 1}
speaker_id_arr_dict = {'msp-improv': np.arange(0, 12, 1), 
                       'crema-d': np.arange(1001, 1092, 1),
                       'iemocap': np.arange(0, 10, 1)}

class WeightDataGenerator():
    def __init__(self, dict_keys):
        self.dict_keys = dict_keys

    def __len__(self):
        return len(self.dict_keys)

    def __getitem__(self, idx):
        data_file_str = self.dict_keys[idx]
        with open(data_file_str, 'rb') as f:
            adv_fed_weight_hist_dict = pickle.load(f)

        w0 = []
        b0 = []
        w1 = []
        b1 = []
        w2 = []
        b2 = []
        gender = []

        for speaker_id in adv_fed_weight_hist_dict:
            label = gender_dict[adv_fed_weight_hist_dict[speaker_id]['gender']]
            gradients = adv_fed_weight_hist_dict[speaker_id]['gradient']
            
            w0.append(torch.from_numpy(np.ascontiguousarray(gradients[0])))
            w1.append(torch.from_numpy(np.ascontiguousarray(gradients[2])))
            w2.append(torch.from_numpy(np.ascontiguousarray(gradients[4])))
            b0.append(torch.from_numpy(np.ascontiguousarray(gradients[1])))
            b1.append(torch.from_numpy(np.ascontiguousarray(gradients[3])))
            b2.append(torch.from_numpy(np.ascontiguousarray(gradients[5])))
            gender.append(label)

        return w0, w1, w2, b0, b1, b2, gender


def create_folder(folder):
    if Path.exists(folder) is False:
        Path.mkdir(folder)


def test(model, device, data_loader, epoch):
    model.eval()
    predict_dict, truth_dict = {}, {}
    
    predict_dict[args.dataset] = []
    truth_dict[args.dataset] = []
        
    loss_list = []
    for batch_idx, (w0, w1, w2, b0, b1, b2, labels) in enumerate(data_loader):
        if args.model_type == 'dnn':
            w0 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w0]))
            w1 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w1]))
            w2 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w2]))
        else:
            w0 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w0])).unsqueeze(dim=1)
            w1 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w1])).unsqueeze(dim=1)
            w2 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w2])).unsqueeze(dim=1)
        b0 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in b0]))
        b1 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in b1]))
        b2 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in b2]))
        labels = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in labels]))

        w0, w1, w2 = w0.to(device), w1.to(device), w2.to(device)
        b0, b1, b2 = b0.to(device), b1.to(device), b2.to(device)
        labels = labels.to(device)
        
        w0, w1, w2 = w0.float(), w1.float(), w2.float()
        b0, b1, b2 = b0.float(), b1.float(), b2.float()
        
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
        
        if args.model_type == 'dnn':
            w0 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w0]))
            w1 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w1]))
            w2 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w2]))
        else:
            w0 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w0])).unsqueeze(dim=1)
            w1 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w1])).unsqueeze(dim=1)
            w2 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w2])).unsqueeze(dim=1)
        
        b0 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in b0]))
        b1 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in b1]))
        b2 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in b2]))
        labels = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in labels]))

        w0, w1, w2 = w0.to(device), w1.to(device), w2.to(device)
        b0, b1, b2 = b0.to(device), b1.to(device), b2.to(device)
        labels = labels.to(device)

        w0, w1, w2 = w0.float(), w1.float(), w2.float()
        b0, b1, b2 = b0.float(), b1.float(), b2.float()
        
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
        if (batch_idx / len(data_loader) * 100) % 20 == 0:
            print('%s epoch: %d, %.2f' % (mode, epoch, batch_idx / len(data_loader) * 100))
        del w0, w1, w2, b0, b1, b2, preds
        torch.cuda.empty_cache()

    if mode != 'train':   
        if args.optimizer == 'adam': scheduler.step(np.mean(loss_list))
    tmp_result_dict = ReturnResultDict(truth_dict, predict_dict, args.adv_dataset, args.pred, mode=mode, loss=None, epoch=epoch)
    return tmp_result_dict


if __name__ == '__main__':

    torch.cuda.empty_cache() 
    torch.multiprocessing.set_sharing_strategy('file_system')

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--adv_dataset', default='iemocap')
    parser.add_argument('--feature_type', default='mel_spec')
    parser.add_argument('--input_channel', default=1)
    parser.add_argument('--learning_rate', default=0.0001)
    parser.add_argument('--input_spec_size', default=128)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--aug', default=None)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--num_epochs', default=100)
    parser.add_argument('--local_epochs', default=5)
    parser.add_argument('--norm', default='znorm')
    parser.add_argument('--win_len', default=200)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--shift', default=1)
    parser.add_argument('--model_type', default='dnn')
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--leak_layer', default='full')
    parser.add_argument('--dropout', default=0.5)

    args = parser.parse_args()
    shift = 'shift' if int(args.shift) == 1 else 'without_shift'
    
    setup_seed(8)
    torch.manual_seed(8)

    root_path = Path('/media/data/projects/speech-privacy')
    pred = 'affect' if args.pred == 'arousal' or args.pred == 'valence' else 'emotion'
    
    save_result_df = pd.DataFrame()
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    model_setting_str = 'local_epoch_'+str(args.local_epochs)
    model_setting_str += '_dropout_' + str(args.dropout).replace('.', '')
    model_setting_str += '_lr_' + str(args.learning_rate)[2:]
    
    weight_file_list = []
    for shadow_idx in range(0, 5):
        for epoch in range(args.num_epochs):
        # for epoch in range(10):
            adv_federated_model_result_path = root_path.joinpath('federated_model_params', args.model_type, args.pred, args.feature_type, args.adv_dataset, model_setting_str, 'fold'+str(int(shadow_idx+1)))
            weight_file_str = str(adv_federated_model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl'))
            weight_file_list.append(weight_file_str)

    # train model to infer gender
    gradient_layer = 4
    train_key_list, validate_key_list = train_test_split(weight_file_list, test_size=0.2, random_state=0)

    # so if it is trained using adv dataset or service provider dataset
    model = attack_model(args.leak_layer, args.feature_type)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-04, betas=(0.9, 0.98), eps=1e-9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

    # define data loader
    dataset_train = WeightDataGenerator(train_key_list)
    dataloader_train = DataLoader(dataset_train, batch_size=1, num_workers=0, shuffle=True)
    
    dataset_valid = WeightDataGenerator(validate_key_list)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, num_workers=0, shuffle=False)

    best_val_recall, best_val_acc, best_epoch = 0, 0, 0
    best_conf_array = None
    for epoch in range(50):
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
    del model
    
    attack_model_result_csv_path = Path.cwd().parents[0].joinpath('results', 'attack', args.leak_layer, args.model_type, args.feature_type, model_setting_str)
    Path.mkdir(attack_model_result_csv_path, parents=True, exist_ok=True)
    torch.save(best_model, str(attack_model_result_csv_path.joinpath('model.pt')))

    # load the evaluation model
    eval_model = attack_model(args.leak_layer, args.feature_type)
    eval_model = eval_model.to(device)
    eval_model.load_state_dict(best_model)

    # we evaluate the attacker performance on service provider training
    for fold_idx in range(0, 5):
        test_list = []
        for epoch in range(args.num_epochs):
            torch.cuda.empty_cache()
            save_row_str = 'fold'+str(int(fold_idx+1))
            row_df = pd.DataFrame(index=[save_row_str])
            
            # Model related
            federated_model_result_path = root_path.joinpath('federated_model_params', args.model_type, args.pred, args.feature_type, args.dataset, model_setting_str, 'fold'+str(int(fold_idx+1)))
            weight_file_str = str(federated_model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl'))
            test_list.append(weight_file_str)
            
        dataset_test = WeightDataGenerator(test_list)
        dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=0, shuffle=False)

        test_results_dict = test(eval_model, device, dataloader_test, epoch=0)
        row_df['acc'] = test_results_dict[args.dataset]['acc'][args.pred]
        row_df['uar'] = test_results_dict[args.dataset]['rec'][args.pred]
        save_result_df = pd.concat([save_result_df, row_df])

        del dataset_test, dataloader_test
        
    row_df = pd.DataFrame(index=['average'])
    row_df['acc'] = np.mean(save_result_df['acc'])
    row_df['uar'] = np.mean(save_result_df['uar'])
    save_result_df = pd.concat([save_result_df, row_df])

    save_result_df.to_csv(str(attack_model_result_csv_path.joinpath('private_'+ str(args.dataset) + '_local_' + str(args.local_epochs) + '_result.csv')))

