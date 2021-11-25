import pandas as pd
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import numpy as np
import pdb

emo_dict = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
affect_dict = {'low': 0, 'med': 1, 'high': 2}

gender_dict = {'F': 0, 'M': 1}

from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # data, label, global_data = self.dataset[self.idxs[item]]['data'][:, :400, :], emo_dict[self.dataset[self.idxs[item]]['label']], self.dataset[self.idxs[item]]['global_data'][0]
        data = self.dataset[self.idxs[item]]['data']
        label = emo_dict[self.dataset[self.idxs[item]]['label']]
        dataset_str = self.dataset[self.idxs[item]]['dataset']

        return torch.tensor(data), torch.tensor(label), dataset_str


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, device, criterion, model_type, train_validation_idx_dict):
        self.args = args
        self.device = device
        self.criterion = criterion
        self.model_type = model_type
        self.train_validation_idx_dict = train_validation_idx_dict
        self.trainloader, self.validloader = self.train_val_test(dataset, list(idxs))
        
    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 20)
        dataset_keys = list(dataset.keys())
        idxs_train = [dataset_keys[idx] for idx in self.train_validation_idx_dict['train']]
        idxs_val = [dataset_keys[idx] for idx in self.train_validation_idx_dict['val']]
        
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=int(self.args.batch_size), shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=1, shuffle=False)
        return trainloader, validloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        
        lr = float(self.args.learning_rate)

        # initialize the optimizer
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2) 
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-04, betas=(0.9, 0.98), eps=1e-9)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

        for iter in range(int(self.args.local_epochs)):
            train_loss_list = []
            for batch_idx, (features, labels, dataset) in enumerate(self.trainloader):
                features, labels = features.to(self.device), labels.to(self.device)
                features = features.float()
                
                model.zero_grad()
                optimizer.zero_grad()
                preds = model(features)
                loss = self.criterion(preds, labels)

                loss.backward()
                optimizer.step()
                
                train_loss_list.append(loss.item())

                if (batch_idx % 10 == 0) and iter == 4:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                train_loss_list.append(loss.item())
            epoch_loss.append(sum(train_loss_list)/len(train_loss_list))
        
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), len(self.train_validation_idx_dict['train'])

    def update_gradients(self, model, global_round):
        # Set mode to train model
        model.train()
        lr = float(self.args.learning_rate)

        model.zero_grad()
        for batch_idx, (features, labels, dataset) in enumerate(self.trainloader):
            if batch_idx == 0:
                features, labels = features.to(self.device), labels.to(self.device)
                features = features.float()
                preds = model(features)
                loss = self.criterion(preds, labels)
                loss.backward()

        grads = []
        for param in model.parameters():
            grads.append(param.grad.detach().clone())
        # num_samples = len(features)
        num_samples = len(self.train_validation_idx_dict['train'])
            
        return grads, loss.item(), num_samples

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        predict_list, truth_list = [], []
        loss_list = []

        for batch_idx, (features, labels, dataset) in enumerate(self.validloader):
            features, labels = features.to(self.device), labels.to(self.device)
            features = features.float()
            
            # Inference
            preds = model(features)
            batch_loss = self.criterion(preds, labels)
            loss_list.append(batch_loss.item())

            # Prediction
            _, predictions = torch.max(preds, 1)
            predictions = predictions.view(-1)
            for pred_idx in range(len(preds)):
                # pdb.set_trace()
                predict_list.append(predictions.detach().cpu().numpy()[pred_idx])
                truth_list.append(labels.detach().cpu().numpy()[pred_idx])
        loss = np.nanmean(loss_list)

        acc_score = accuracy_score(truth_list, predict_list)
        rec_score = recall_score(truth_list, predict_list, average='macro')
        
        return acc_score, rec_score, loss, len(self.train_validation_idx_dict['val'])


def average_weights(w, num_samples_list):
    """
    Returns the average of the weights.
    """
    total_num_samples = np.sum(num_samples_list)
    w_avg = copy.deepcopy(w[0])

    for key in w_avg.keys():
        w_avg[key] = w[0][key]*(num_samples_list[0]/total_num_samples)
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += torch.div(w[i][key]*num_samples_list[i], total_num_samples)
    return w_avg


def average_gradients(g, num_samples_list):
    """
    Returns the average of the gradients.
    """
    total_num_samples = np.sum(num_samples_list)
    g_avg = copy.deepcopy(g[0])
    
    for layer_idx in range(len(g[0])):
        g_avg[layer_idx] = g[0][layer_idx] # *(num_samples_list[0]/total_num_samples)
    for layer_idx in range(len(g[0])):
        for client_idx in range(1, len(g)):
            g_avg[layer_idx] += torch.div(g[client_idx][layer_idx]*num_samples_list[client_idx], total_num_samples)
            # g_avg[layer_idx] += g[client_idx][layer_idx]
    return g_avg