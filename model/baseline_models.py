#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tiantian
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import pdb
from torch.nn.modules import dropout
import itertools
from pytorch_lightning.core.lightning import LightningModule
import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau


class dnn_classifier(LightningModule):
    def __init__(self, pred, input_spec, dropout, args):

        super(dnn_classifier, self).__init__()
        self.dropout_p = dropout
        self.num_emo_classes = 4
        self.num_affect_classes = 3
        self.num_gender_class = 2
        self.lr = float(args.learning_rate)
        self.fed_alg = args.model_type
        self.updates = None
        self.test_conf = None
        self.gradient = None

        self.dropout = nn.Dropout(p=self.dropout_p)
        
        self.dense_relu1 = nn.ReLU()
        self.dense_relu2 = nn.ReLU()

        self.dense1 = nn.Linear(input_spec, 256)
        self.dense2 = nn.Linear(256, 128)

        self.pred = pred
        if self.pred == 'emotion':
            self.pred_layer = nn.Linear(128, self.num_emo_classes)
        elif self.pred == 'affect':
            self.pred_layer = nn.Linear(128, self.num_affect_classes)
        else:
            self.pred_layer = nn.Linear(128, self.num_gender_class)
        
        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, input_var):

        x = input_var.float()
        
        x = self.dense1(x)
        x = self.dense_relu1(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.dense_relu2(x)
        x = nn.Dropout(p=0.2)(x)

        preds = self.pred_layer(x)
        preds = torch.log_softmax(preds, dim=1)
        
        return preds

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        
        data, label, dataset_str = train_batch
        logits = self.forward(data)
        loss = self.cross_entropy_loss(logits, label)

        predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
        pred_list, truth_list = [], []
        for pred_idx in range(len(predictions)):
            pred_list.append(predictions[pred_idx])
            truth_list.append(label.detach().cpu().numpy()[pred_idx])
        
        if self.fed_alg == 'fed_sgd':
            loss.backward(retain_graph=True)
            grads = []
            for param in self.parameters():
                grads.append(param.grad.detach().clone())
            self.gradient = grads

        return {'loss': loss, 'pred': pred_list, 'truth': truth_list}
    
    def training_epoch_end(self, train_step_outputs):
        result_dict = self.result_summary(train_step_outputs, self.current_epoch, mode='train')
        self.log('train_loss', result_dict['loss'], on_epoch=True)
        self.log('train_acc_epoch', result_dict['acc'], on_epoch=True)
        self.log('train_uar_epoch', result_dict['uar'], on_epoch=True)
        self.log('train_size', result_dict['size'], on_epoch=True)

    def validation_step(self, val_batch, batch_idx):
        data, label, dataset_str = val_batch
        logits = self.forward(data)
        loss = self.cross_entropy_loss(logits, label)
        predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
        pred_list, truth_list = [], []
        for pred_idx in range(len(predictions)):
            pred_list.append(predictions[pred_idx])
            truth_list.append(label.detach().cpu().numpy()[pred_idx])
        
        return {'loss': loss, 'pred': pred_list, 'truth': truth_list}

    def validation_epoch_end(self, val_step_outputs):
        result_dict = self.result_summary(val_step_outputs, self.current_epoch, mode='validation')
        self.log('val_loss', result_dict['loss'], on_epoch=True)
        self.log('val_acc', result_dict['acc'], on_epoch=True)
        self.log('val_uar', result_dict['uar'], on_epoch=True)
        self.log('val_size', result_dict['size'], on_epoch=True)

    def test_step(self, test_batch, batch_nb):
        data, label, dataset_str = test_batch
        logits = self.forward(data)
        predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
        pred_list, truth_list = [], []
        for pred_idx in range(len(predictions)):
            pred_list.append(predictions[pred_idx])
            truth_list.append(label.detach().cpu().numpy()[pred_idx])
        return {'pred': pred_list, 'truth': truth_list}

    def test_epoch_end(self, test_step_outputs):
        result_dict = self.result_summary(test_step_outputs, 0, mode='test')
        self.log('test_acc', result_dict['acc'], on_epoch=True)
        self.log('test_uar', result_dict['uar'], on_epoch=True)
        self.test_conf = result_dict['conf']
        # self.log('test_conf', result_dict['conf'], on_epoch=True)
        # self.log_dict({'test_conf': result_dict['conf']})
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def result_summary(self, step_outputs, epoch, mode='train'):

        loss_list, y_true, y_pred = [], [], []
        for step in range(len(step_outputs)):
            for idx in range(len(step_outputs[step]['pred'])):
                y_true.append(step_outputs[step]['truth'][idx])
                y_pred.append(step_outputs[step]['pred'][idx])
            if mode != 'test': loss_list.append(step_outputs[step]['loss'].item())

        result_dict = {}
        acc_score = accuracy_score(y_true, y_pred)
        rec_score = recall_score(y_true, y_pred, average='macro')
        confusion_matrix_arr = np.round(confusion_matrix(y_true, y_pred, normalize='true')*100, decimals=2)
        
        result_dict['acc'] = acc_score
        result_dict['uar'] = rec_score
        result_dict['conf'] = confusion_matrix_arr
        result_dict['size'] = len(y_true)
        if mode != 'test':
            result_dict['loss'] = np.mean(loss_list)

        '''
        print()
        if mode != 'test':
            result_dict['loss'] = np.mean(loss_list)
            print('%s accuracy %.3f / recall %.3f / loss %.3f after %d' % (mode, acc_score, rec_score, np.mean(loss_list), epoch))
            print()
            print(confusion_matrix_arr)
        else:
            print('%s accuracy %.3f / recall %.3f' % (mode, acc_score, rec_score))
            print()
        '''
        
        return result_dict

