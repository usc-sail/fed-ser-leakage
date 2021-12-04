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
from pytorch_lightning.core.lightning import LightningModule
import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau


class attack_model(LightningModule):
    def __init__(self, leak_layer, feature_type):

        super(attack_model, self).__init__()
        self.dropout_p = 0.2
        self.num_emo_classes = 4
        self.num_gender_class = 2
        self.num_affect_classes = 3
        self.leak_layer = leak_layer
        self.lr = 0.0001

        if feature_type == 'wav2vec':
            first_layer_feat_size = 1408
        elif feature_type == 'emobase':
            first_layer_feat_size = 3840+256
            stride_len = 8
        elif feature_type == 'ComParE':
            first_layer_feat_size = 1152
        elif feature_type == 'cpc':
            first_layer_feat_size = 4352
            stride_len = 4
        elif feature_type == 'apc' or feature_type == 'npc' or feature_type == 'vq_apc':
            first_layer_feat_size = 3200+256
            stride_len = 6
        else:
            first_layer_feat_size = 3072+256
            stride_len = 8
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=(4, 4)),
            nn.Dropout2d(self.dropout_p),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=stride_len, stride=(stride_len, stride_len)),
            nn.Dropout2d(self.dropout_p),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=(4, 4)),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=(4, 4)),
            nn.Dropout2d(self.dropout_p)
        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(1, 4)),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(1, 4)),
            nn.Dropout2d(self.dropout_p),
        )

        if self.leak_layer == 'full':
            self.dense1 = nn.Linear(first_layer_feat_size+768+516, 256)
        elif self.leak_layer == 'first':
            self.dense1 = nn.Linear(first_layer_feat_size, 256)
        elif self.leak_layer == 'second':
            self.dense1 = nn.Linear(2176, 256)
        else:
            self.dense1 = nn.Linear(1028, 256)

        self.dense2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=self.dropout_p)
        
        self.dense_relu1 = nn.ReLU()
        self.dense_relu2 = nn.ReLU()
        self.pred_layer = nn.Linear(128, self.num_gender_class)
        self.test_results = None
        self.init_weight()

        self.test_result = {}

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, weights, bias):
        if self.leak_layer == 'first':
            weights = self.conv0(weights.float())
        elif self.leak_layer == 'second':
            weights = self.conv1(weights.float())
        else:
            weights = self.conv2(weights.float())
            
        weights_size = weights.size()
        weights = weights.reshape(-1, weights_size[1]*weights_size[2]*weights_size[3])
        z = torch.cat((weights, bias), 1)
        
        z = self.dense1(z)
        z = self.dense_relu1(z)
        z = self.dropout(z)

        z = self.dense2(z)
        z = self.dense_relu2(z)
        z = self.dropout(z)

        preds = self.pred_layer(z)
        preds = torch.log_softmax(preds, dim=1)
        return preds

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        weights, bias, gender = train_batch
        logits = self.forward(weights.unsqueeze(dim=1), bias)
        loss = self.cross_entropy_loss(logits, gender)

        predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
        pred_list, truth_list = [], []
        for pred_idx in range(len(predictions)):
            pred_list.append(predictions[pred_idx])
            truth_list.append(gender.detach().cpu().numpy()[pred_idx])

        return {'loss': loss, 'pred': pred_list, 'truth': truth_list}

    def training_epoch_end(self, train_step_outputs):
        result_dict = self.result_summary(train_step_outputs, self.current_epoch, mode='train')
        self.log('train_loss', result_dict['loss'], on_epoch=True)
        self.log('train_acc_epoch', result_dict['acc'], on_epoch=True)
        self.log('train_uar_epoch', result_dict['uar'], on_epoch=True)

    def validation_step(self, val_batch, batch_idx):
        weights, bias, gender = val_batch
        logits = self.forward(weights.unsqueeze(dim=1), bias)
        loss = self.cross_entropy_loss(logits, gender)
        predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
        pred_list, truth_list = [], []
        for pred_idx in range(len(predictions)):
            pred_list.append(predictions[pred_idx])
            truth_list.append(gender.detach().cpu().numpy()[pred_idx])
        
        return {'loss': loss, 'pred': pred_list, 'truth': truth_list}

    def validation_epoch_end(self, val_step_outputs):
        result_dict = self.result_summary(val_step_outputs, self.current_epoch, mode='validation')
        self.log('val_loss', result_dict['loss'], on_epoch=True)
        self.log('val_acc_epoch', result_dict['acc'], on_epoch=True)
        self.log('val_uar_epoch', result_dict['uar'], on_epoch=True)

    def test_step(self, test_batch, batch_nb):
        weights, bias, gender = test_batch
        logits = self.forward(weights.unsqueeze(dim=1), bias)
        predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
        pred_list, truth_list = [], []
        for pred_idx in range(len(predictions)):
            pred_list.append(predictions[pred_idx])
            truth_list.append(gender.detach().cpu().numpy()[pred_idx])
        return {'pred': pred_list, 'truth': truth_list}

    def test_epoch_end(self, test_step_outputs):
        result_dict = self.result_summary(test_step_outputs, 0, mode='test')
        self.log('test_acc_epoch', result_dict['acc'], on_epoch=True)
        self.log('test_uar_epoch', result_dict['uar'], on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
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

        print()
        if mode != 'test':
            result_dict['loss'] = np.mean(loss_list)
            print('%s accuracy %.3f / recall %.3f / loss %.3f after %d' % (mode, acc_score, rec_score, np.mean(loss_list), epoch))
            print()
        else:
            print('%s accuracy %.3f / recall %.3f' % (mode, acc_score, rec_score))
            print()
        print(confusion_matrix_arr)
        return result_dict
