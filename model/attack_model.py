#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tiantian
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb
import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau


class attack_model(nn.Module):
    def __init__(self, leak_layer, feature_type):

        super(attack_model, self).__init__()
        self.dropout_p = 0.2
        self.num_gender_class = 2
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