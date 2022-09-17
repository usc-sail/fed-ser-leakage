import pandas as pd
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import copy, pdb, time, warnings, torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore') 


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
        g_avg[layer_idx] = g[0][layer_idx] * (num_samples_list[0]/total_num_samples)
    for layer_idx in range(len(g[0])):
        for client_idx in range(1, len(g)):
            g_avg[layer_idx] += torch.div(g[client_idx][layer_idx]*num_samples_list[client_idx], total_num_samples)
    return g_avg


def result_summary(step_outputs):
    loss_list, y_true, y_pred = [], [], []
    for step in range(len(step_outputs)):
        for idx in range(len(step_outputs[step]['pred'])):
            y_true.append(step_outputs[step]['truth'][idx])
            y_pred.append(step_outputs[step]['pred'][idx])
        loss_list.append(step_outputs[step]['loss'])

    result_dict = {}
    acc_score = accuracy_score(y_true, y_pred)
    rec_score = recall_score(y_true, y_pred, average='macro')
    confusion_matrix_arr = np.round(confusion_matrix(y_true, y_pred, normalize='true')*100, decimals=2)
    
    result_dict['acc'] = acc_score
    result_dict['uar'] = rec_score
    result_dict['conf'] = confusion_matrix_arr
    result_dict['loss'] = np.mean(loss_list)
    result_dict['num_samples'] = len(y_pred)
    return result_dict


class local_trainer(object):
    def __init__(self, args, device, criterion, model_type, dataloader):
        self.args = args
        self.device = device
        self.criterion = criterion
        self.model_type = model_type
        self.dataloader = dataloader
        
    def update_weights(self, model, clip=None):
        # Set mode to train model
        model.train()
        
        step_outputs = []
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.args.learning_rate), weight_decay=1e-04, betas=(0.9, 0.98), eps=1e-9)
        
        for iter in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                x, y, dataset = batch_data
                x, y = x.to(self.device), y.to(self.device)
                
                model.zero_grad()
                optimizer.zero_grad()
                logits = model(x.float())
                loss = self.criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
                pred_list = [predictions[pred_idx] for pred_idx in range(len(predictions))]
                truth_list = [y.detach().cpu().numpy()[pred_idx] for pred_idx in range(len(predictions))]
                step_outputs.append({'loss': loss.item(), 'pred': pred_list, 'truth': truth_list})
        
        result_dict = result_summary(step_outputs)
        return model.state_dict(), result_dict

    def update_gradients(self, model):
        # Set mode to train model
        model.train()
        step_outputs = []
        for batch_idx, batch_data in enumerate(self.dataloader):
            if batch_idx == 0:
                x, y, dataset = batch_data
                x, y = x.to(self.device), y.to(self.device)

                model.zero_grad()
                logits = model(x.float())
                loss = self.criterion(logits, y)
                loss.backward()
                grads = [param.grad.detach().clone() for param in model.parameters()]

                predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
                pred_list = [predictions[pred_idx] for pred_idx in range(len(predictions))]
                truth_list = [y.detach().cpu().numpy()[pred_idx] for pred_idx in range(len(predictions))]
                step_outputs.append({'loss': loss.item(), 'pred': pred_list, 'truth': truth_list})
        result_dict = result_summary(step_outputs)
        return grads, result_dict

    def inference(self, model):
        model.eval()
        step_outputs = []
        
        for batch_idx, batch_data in enumerate(self.dataloader):
            x, y, dataset = batch_data
            x, y = x.to(self.device), y.to(self.device)

            logits = model(x.float())
            loss = self.criterion(logits, y)
            
            predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
            pred_list = [predictions[pred_idx] for pred_idx in range(len(predictions))]
            truth_list = [y.detach().cpu().numpy()[pred_idx] for pred_idx in range(len(predictions))]
            step_outputs.append({'loss': loss.item(), 'pred': pred_list, 'truth': truth_list})
        result_dict = result_summary(step_outputs)
        return result_dict

def noise_add(noise_scale, w, device):
    w_noise = copy.deepcopy(w)
    for i in w.keys():
        noise = np.random.normal(0, noise_scale, w[i].size())
        noise = torch.from_numpy(noise).float().to(device)
        w_noise[i] = w_noise[i] + noise
    return w_noise
