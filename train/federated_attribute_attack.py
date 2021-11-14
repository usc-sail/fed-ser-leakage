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
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', '..', 'utils'))

from training_tools import EarlyStopping, SpeechDataGenerator, ReturnResultDict
from training_tools import speech_collate, setup_seed, seed_worker, get_class_weight
from baseline_models import two_d_cnn, dnn_classifier, attack_model, attack_conv_model, attack_conv_fusion_model
from sampling import ser_iid
from update import LocalUpdate, average_weights

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

import pdb

emo_dict = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
affect_dict = {'low': 0, 'med': 1, 'high': 2}

gender_dict = {'F': 0, 'M': 1}
speaker_id_arr_dict = {'msp-improv': np.arange(0, 12, 1), 
                       'crema-d': np.arange(1001, 1092, 1),
                       'iemocap': np.arange(0, 10, 1)}

shift_len = 50


class DataGenerator():
    def __init__(self, data_dict, dict_keys):
        """
        Read the textfile and get the paths
        """
        self.data_dict = data_dict
        self.dict_keys = dict_keys

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict[self.dict_keys[idx]]
        
        w0 = data[0]
        b0 = data[1]
        w1 = data[2]
        b1 = data[3]
        w2 = data[4]
        b2 = data[5]
        label = data['gender']
        
        sample = {'w0': torch.from_numpy(np.ascontiguousarray(w0)),
                  'w1': torch.from_numpy(np.ascontiguousarray(w1)),
                  'w2': torch.from_numpy(np.ascontiguousarray(w2)),
                  'b0': torch.from_numpy(np.ascontiguousarray(b0)),
                  'b1': torch.from_numpy(np.ascontiguousarray(b1)),
                  'b2': torch.from_numpy(np.ascontiguousarray(b2)),
                  'label': torch.from_numpy(np.ascontiguousarray(label))}
        return sample

def data_collate(batch):
    w0 = []
    w1 = []
    w2 = []
    b0 = []
    b1 = []
    b2 = []
    label = []
    for sample in batch:
        w0.append(sample['w0'])
        w1.append((sample['w1']))
        w2.append(sample['w2'])
        b0.append(sample['b0'])
        b1.append((sample['b1']))
        b2.append(sample['b2'])
        label.append(sample['label'])
    return w0, w1, w2, b0, b1, b2, label



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
        model.train()
        if args.model_type == 'dnn':
            w0 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w0])).unsqueeze(dim=1)
            w1 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w1])).unsqueeze(dim=1)
            w2 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w2])).unsqueeze(dim=1)
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
        loss_list.append(batch_loss.item())

        # Prediction
        _, predictions = torch.max(preds, 1)
        predictions = predictions.view(-1)
        for pred_idx in range(len(preds)):
            predict_dict[args.dataset].append(predictions.detach().cpu().numpy()[pred_idx])
            truth_dict[args.dataset].append(labels.detach().cpu().numpy()[pred_idx])
    
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
            w0 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w0])).unsqueeze(dim=1)
            w1 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w1])).unsqueeze(dim=1)
            w2 = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in w2])).unsqueeze(dim=1)
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
    
    if mode != 'train':   
        if args.optimizer == 'adam':
            scheduler.step(np.mean(loss_list))
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

    args = parser.parse_args()
    shift = 'shift' if int(args.shift) == 1 else 'without_shift'
    
    setup_seed(8)
    torch.manual_seed(8)

    root_path = Path('/media/data/projects/speech-privacy')
    pred = 'affect' if args.pred == 'arousal' or args.pred == 'valence' else 'emotion'
    preprocess_path = root_path.joinpath('federated_learning', shift, args.feature_type, args.input_spec_size, pred)
    
    save_result_df = pd.DataFrame()
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    gradients_layer_dict = {}
    gender_list = []
    for shadow_idx in range(0, 5):
        # adv_federated_model_result_path = root_path.joinpath('federated_shadow', exp_result_str, str(args.local_epochs), args.pred, args.feature_type, args.adv_dataset, args.input_spec_size, 'shadow_'+str(shadow_idx+1))
        if args.feature_type == 'wav2vec':
            adv_federated_model_result_path = root_path.joinpath('federated', args.model_type, str(args.local_epochs), args.pred, args.feature_type, args.adv_dataset, 'fold'+str(int(shadow_idx+1)))
        else:
            adv_federated_model_result_path = root_path.joinpath('federated', args.model_type, str(args.local_epochs), args.pred, args.feature_type, args.adv_dataset, shift, args.input_spec_size, 'fold'+str(int(shadow_idx+1)))
        
        with open(str(adv_federated_model_result_path.joinpath('weights_hist.pkl')), 'rb') as f:
            adv_fed_weight_hist_dict = pickle.load(f)
        
        accuracy_dict = {}
        for epoch in range(args.num_epochs):
        # for epoch in range(10):
            accuracy_dict[epoch] = []
            for speaker_id in adv_fed_weight_hist_dict[epoch]:
                key = str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id
                label = gender_dict[adv_fed_weight_hist_dict[epoch][speaker_id]['gender']]
                gradients = adv_fed_weight_hist_dict[epoch][speaker_id]['gradient']
                gradients_layer_dict[key] = {}

                for layer_idx in range(len(gradients)):
                    # if it is a filter
                    if len(gradients[layer_idx].shape) == 4:
                        filter_size = gradients[layer_idx].shape
                        # gradients_layer_dict[key][layer_idx] = gradients[layer_idx].detach().clone().cpu().numpy().reshape([filter_size[0], filter_size[1], filter_size[2]*filter_size[3]])
                        gradients_layer_dict[key][layer_idx] = gradients[layer_idx]
                    else:
                        gradients_layer_dict[key][layer_idx] = gradients[layer_idx]
                    
                gradients_layer_dict[key]['gender'] = label
        del adv_fed_weight_hist_dict
    
    # train model to infer gender
    gradient_layer = 4
    train_key, validate_key = train_test_split(list(gradients_layer_dict.keys()), test_size=0.2, random_state=0)

    # so if it is trained using adv dataset or service provider dataset
    if args.model_type == 'dnn':
        model = attack_model(args.leak_layer)
    else:
        # model = attack_conv_model(args.leak_layer)
        # model = attack_conv_fusion_model(args.leak_layer)
        model = attack_model(args.leak_layer)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-04, betas=(0.9, 0.98), eps=1e-9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

    train_dict, validate_dict = {}, {}
    for key in train_key:
        train_dict[key] = gradients_layer_dict[key].copy()
    for key in validate_key:
        validate_dict[key] = gradients_layer_dict[key].copy()
    
    norm_dict = {}
    for layer_idx in range(len(gradients)):
        tmp_data = []
        norm_dict[layer_idx] = {}
        for key in train_key:
            tmp_data.append(train_dict[key][layer_idx])
        norm_dict[layer_idx]['mean'] = np.mean(np.array(tmp_data), axis=0)
        norm_dict[layer_idx]['std'] = np.std(np.array(tmp_data), axis=0)
        
        for key in train_key:
            train_dict[key][layer_idx] = (train_dict[key][layer_idx] - norm_dict[layer_idx]['mean']) / norm_dict[layer_idx]['std']
        for key in validate_key:
            validate_dict[key][layer_idx] = (validate_dict[key][layer_idx] - norm_dict[layer_idx]['mean']) / norm_dict[layer_idx]['std']
        
    dataset_train = DataGenerator(train_dict, list(train_dict.keys()))
    dataloader_train = DataLoader(dataset_train, batch_size=10, num_workers=0, shuffle=True, collate_fn=data_collate)

    dataset_valid = DataGenerator(validate_dict, list(validate_dict.keys()))
    dataloader_valid = DataLoader(dataset_valid, batch_size=10, num_workers=0, shuffle=False, collate_fn=data_collate)

    loss, total, correct = 0.0, 0.0, 0.0
    best_val_recall, final_recall, best_epoch, final_confusion = 0, 0, 0, 0
    best_val_acc, final_acc = 0, 0

    del gradients_layer_dict

    for epoch in range(30):
        train_result = train(model, device, dataloader_train, epoch, mode='train')
        validate_result = train(model, device, dataloader_valid, epoch, mode='validate')
        
        if validate_result[args.adv_dataset]['acc'][args.pred] > best_val_acc:
            best_val_acc = validate_result[args.adv_dataset]['acc'][args.pred]
            best_val_recall = validate_result[args.adv_dataset]['rec'][args.pred]
            best_epoch = epoch
            best_model = deepcopy(model.state_dict())
            print('best epoch %d, best val acc %.2f, best val rec %.2f' % (best_epoch, best_val_acc*100, best_val_recall*100))

    if args.model_type == 'dnn':
        eval_model = attack_model(args.leak_layer)
    else:
        # eval_model = attack_conv_model(args.leak_layer)
        # eval_model = attack_conv_fusion_model(args.leak_layer)
        eval_model = attack_model(args.leak_layer)
    eval_model = eval_model.to(device)
    eval_model.load_state_dict(best_model)
    
    epoch_info_list = [[0, args.num_epochs], [0, 25], [25, 50], [50, 75], [75, 100]]
    for epoch_info in epoch_info_list:
        for fold_idx in range(0, 5):
            torch.cuda.empty_cache()

            save_row_str = 'fold'+str(int(fold_idx+1))
            row_df = pd.DataFrame(index=[save_row_str])
            
            # Model related
            exp_result_str = args.model_type
            if args.feature_type == 'wav2vec':
                federated_model_result_path = root_path.joinpath('federated', exp_result_str, str(args.local_epochs), args.pred, args.feature_type, args.dataset, save_row_str)
            else:
                federated_model_result_path = root_path.joinpath('federated', exp_result_str, str(args.local_epochs), args.pred, args.feature_type, args.dataset, shift, args.input_spec_size, save_row_str)
            
            with open(str(federated_model_result_path.joinpath('weights_hist.pkl')), 'rb') as f:
                fed_weight_hist_dict = pickle.load(f)
            
            accuracy_dict = {}
            gradients_layer_dict = {}
            gender_list = []
            for epoch in range(epoch_info[0], epoch_info[1]):
                accuracy_dict[epoch] = []
                for speaker_id in fed_weight_hist_dict[epoch]:
                    key = str(fold_idx)+'_'+str(epoch)+'_'+speaker_id
                    label = gender_dict[fed_weight_hist_dict[epoch][speaker_id]['gender']]
                    gradients = fed_weight_hist_dict[epoch][speaker_id]['gradient']
                    gradients_layer_dict[key] = {}
                    for layer_idx in range(len(gradients)):
                        if len(gradients[layer_idx].shape) == 4:
                            filter_size = gradients[layer_idx]
                            # gradients_layer_dict[key][layer_idx] = gradients[layer_idx].detach().clone().cpu().numpy().reshape([filter_size[0], filter_size[1], filter_size[2]*filter_size[3]])
                            gradients_layer_dict[key][layer_idx] = gradients[layer_idx]
                        else:
                            gradients_layer_dict[key][layer_idx] = gradients[layer_idx]
                        
                    gradients_layer_dict[key]['gender'] = label

            test_dict = {}
            for key in list(gradients_layer_dict.keys()):
                test_dict[key] = gradients_layer_dict[key].copy()
                for layer_idx in range(len(gradients)):
                    test_dict[key][layer_idx] = (test_dict[key][layer_idx] - norm_dict[layer_idx]['mean']) / norm_dict[layer_idx]['std']
            
            dataset_test = DataGenerator(test_dict, list(test_dict.keys()))
            dataloader_test = DataLoader(dataset_test, batch_size=10, num_workers=0, shuffle=False, collate_fn=data_collate)

            test_results_dict = test(eval_model, device, dataloader_test, epoch=0)
            row_df['acc'] = test_results_dict[args.dataset]['acc'][args.pred]
            row_df['rec'] = test_results_dict[args.dataset]['rec'][args.pred]
            row_df['start'] = epoch_info[0]
            row_df['end'] = epoch_info[1]
            save_result_df = pd.concat([save_result_df, row_df])
        
        row_df = pd.DataFrame(index=['average'])
        row_df['acc'] = np.mean(save_result_df['acc'])
        row_df['rec'] = np.mean(save_result_df['rec'])
        row_df['start'] = epoch_info[0]
        row_df['end'] = epoch_info[1]
        save_result_df = pd.concat([save_result_df, row_df])
    
    create_folder(Path.cwd().parents[0].joinpath('results'))
    create_folder(Path.cwd().parents[0].joinpath('results', args.leak_layer))
    create_folder(Path.cwd().parents[0].joinpath('results', args.leak_layer, args.model_type))
    create_folder(Path.cwd().parents[0].joinpath('results', args.leak_layer, args.model_type, args.feature_type))
    create_folder(Path.cwd().parents[0].joinpath('results', args.leak_layer, args.model_type, args.feature_type, shift))
    create_folder(Path.cwd().parents[0].joinpath('results', args.leak_layer, args.model_type, args.feature_type, shift, args.input_spec_size))

    save_result_df.to_csv(str(Path.cwd().parents[0].joinpath('results', args.leak_layer, args.model_type, args.feature_type, shift, args.input_spec_size, 'private_'+ str(args.dataset) + '_local_' + str(args.local_epochs) + '_result.csv')))

