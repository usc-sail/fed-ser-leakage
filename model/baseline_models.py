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

class dnn_classifier(nn.Module):
    def __init__(self, pred, input_spec, dropout):

        super(dnn_classifier, self).__init__()
        self.dropout_p = dropout
        self.num_emo_classes = 4
        self.num_affect_classes = 3
        self.num_gender_class = 2
        
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
        # we set the dropout 
        x = nn.Dropout(p=0.2)(x)

        preds = self.pred_layer(x)
        
        return preds


class attack_model(nn.Module):
    def __init__(self, leak_layer, feature_type):

        super(attack_model, self).__init__()
        self.dropout_p = 0.2
        self.num_emo_classes = 4
        self.num_gender_class = 2
        self.num_affect_classes = 3
        self.leak_layer = leak_layer

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
        
        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, w0, w1, w2, b0, b1, b2):

        # pdb.set_trace()
        if self.leak_layer == 'full':
            w0 = self.conv0(w0.float())
            w1 = self.conv1(w1.float())
            w2 = self.conv2(w2.float())

            w0_size = w0.size()
            w1_size = w1.size()
            w2_size = w2.size()

            w0 = w0.reshape(-1, w0_size[1]*w0_size[2]*w0_size[3])
            w1 = w1.reshape(-1, w1_size[1]*w1_size[2]*w1_size[3])
            w2 = w2.reshape(-1, w2_size[1]*w2_size[2]*w2_size[3])

            z = torch.cat((w0, w1), 1)
            z = torch.cat((z, w2), 1)
            z = torch.cat((z, b0.squeeze(dim=1)), 1)
            z = torch.cat((z, b1.squeeze(dim=1)), 1)
            z = torch.cat((z, b2.squeeze(dim=1)), 1)

            # pdb.set_trace()
        elif self.leak_layer == 'first':
            
            w0 = self.conv0(w0.float())
            w0_size = w0.size()
            w0 = w0.reshape(-1, w0_size[1]*w0_size[2]*w0_size[3])
            z = torch.cat((w0, b0.squeeze(dim=1)), 1)
            # pdb.set_trace()

        elif self.leak_layer == 'second':
            w1 = self.conv1(w1.float())
            w1_size = w1.size()
            w1 = w1.reshape(-1, w1_size[1]*w1_size[2]*w1_size[3])
            z = torch.cat((w1, b1.squeeze(dim=1)), 1)

            # pdb.set_trace()
        else:
            w2 = self.conv2(w2.float())
            w2_size = w2.size()
            w2 = w2.reshape(-1, w2_size[1]*w2_size[2]*w2_size[3])
            z = torch.cat((w2, b2), 1)
            
        # pdb.set_trace()

        z = self.dense1(z)
        z = self.dense_relu1(z)
        z = self.dropout(z)

        z = self.dense2(z)
        z = self.dense_relu2(z)
        z = self.dropout(z)

        preds = self.pred_layer(z)
        
        return preds

