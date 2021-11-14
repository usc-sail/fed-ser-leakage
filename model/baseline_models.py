#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tiantian
"""
from bdb import set_trace
from re import T
import pandas as pd
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.nn import functional as F
import pdb

from torch.nn.modules import dropout
import itertools

class two_d_cnn(nn.Module):
    def __init__(self, input_spec_size=128, pred='emotion',
                attention_size=256, global_feature=1):

        super(two_d_cnn, self).__init__()
        self.dropout_p = 0.25
        self.num_emo_classes = 4
        self.num_gender_class = 2
        self.num_affect_classes = 3
        self.pred = pred

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=3, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 10), stride=(2, 10)),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            # nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

        )

        self.dense1 = nn.Linear(1024, 256)
        self.dense2 = nn.Linear(256, 128)

        self.dropout = nn.Dropout(p=self.dropout_p)
        
        self.dense_relu1 = nn.ReLU()
        self.dense_relu2 = nn.ReLU()

        if pred == 'emotion':
            self.pred_layer = nn.Linear(128, self.num_emo_classes) 
        elif pred == 'gender':
            self.pred_layer = nn.Linear(128, self.num_gender_class) 
        else:
            self.pred_layer = nn.Linear(128, self.num_affect_classes) 
        
        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, input_var, global_feature=None):

        x = input_var.float()
        x = x.transpose(2, 3)
        x = self.conv(x.float())

        # pdb.set_trace()
        x_size = x.size()
        x = x.view(x_size[0], x_size[1] * x_size[2] * x_size[3])
        
        z = self.dense1(x)
        z = self.dense_relu1(z)
        z = self.dropout(z)

        z = self.dense2(z)
        z = self.dense_relu2(z)
        z = self.dropout(z)

        preds = self.pred_layer(z)
        
        return preds


class two_d_cnn_lstm(nn.Module):
    def __init__(self, input_spec_size=128, pred='emotion', 
                 hidden_size=64, global_feature=1):

        super(two_d_cnn_lstm, self).__init__()
        self.dropout_p = 0.2
        self.num_emo_classes = 4
        self.num_gender_class = 2
        self.num_affect_classes = 3
        self.pred = pred
        self.rnn_input_size = int(64 * input_spec_size / 8)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(32, 48, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(48, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 2), stride=(5, 2)),
            nn.Dropout2d(self.dropout_p),
        )

        self.rnn_cell = nn.GRU
        self.rnn1 = self.rnn_cell(input_size=self.rnn_input_size, hidden_size=hidden_size,
                                 num_layers=4, batch_first=True,
                                 dropout=self.dropout_p, bidirectional=True)
        
        d_att, n_att = 256, 8
        self.att_pool = nn.Tanh()
        
        self.att_mat1 = torch.nn.Parameter(torch.rand(d_att, int(hidden_size*2)), requires_grad=True)
        self.att_mat2 = torch.nn.Parameter(torch.rand(n_att, d_att), requires_grad=True)
        
        self.dense1 = nn.Linear(int(hidden_size*2)*10, 256)
        self.dense2 = nn.Linear(256, 128)

        self.dropout = nn.Dropout(p=self.dropout_p)
        
        self.dense_relu1 = nn.ReLU()
        self.dense_relu2 = nn.ReLU()

        if pred == 'emotion':
            self.pred_layer = nn.Linear(128, self.num_emo_classes) 
        elif pred == 'gender':
            self.pred_layer = nn.Linear(128, self.num_gender_class) 
        else:
            self.pred_layer = nn.Linear(128, self.num_affect_classes) 
        
        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, input_var, global_feature=None):

        x = input_var.float()
        x = self.conv(x.float())
        x = x.transpose(1, 2).contiguous()
        
        x_size = x.size()
        x = x.reshape(-1, x_size[1], x_size[2]*x_size[3])

        # pdb.set_trace()
        
        self.rnn1.flatten_parameters()
        x, h_state = self.rnn1(x)

        x_size = x.size()
        z = x.reshape(-1, x_size[1]*x_size[2])
        
        z = self.dense1(z)
        z = self.dense_relu1(z)
        z = self.dropout(z)

        z = self.dense2(z)
        z = self.dense_relu2(z)
        z = self.dropout(z)

        preds = self.pred_layer(z)
        
        return preds


class dnn_classifier(nn.Module):
    def __init__(self, pred, input_spec):

        super(dnn_classifier, self).__init__()
        self.dropout_p = 0.5
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
        x = self.dropout(x)

        preds = self.pred_layer(x)
        
        return preds


class attack_model(nn.Module):
    def __init__(self, leak_layer):

        super(attack_model, self).__init__()
        self.dropout_p = 0.25
        self.num_emo_classes = 4
        self.num_gender_class = 2
        self.num_affect_classes = 3
        self.leak_layer = leak_layer
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(4, 4)),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(4, 4)),
            nn.Dropout2d(self.dropout_p),
            
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(4, 8)),
            nn.Dropout2d(self.dropout_p),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(4, 4)),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(4, 4)),
            nn.Dropout2d(self.dropout_p)
        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 4)),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 4)),
            nn.Dropout2d(self.dropout_p),
        )

        if self.leak_layer == 'full':
            # self.dense1 = nn.Linear(2692, 128)
            self.dense1 = nn.Linear(2436, 256)
        elif self.leak_layer == 'first':
            # self.dense1 = nn.Linear(1280, 128)
            # self.dense1 = nn.Linear(1536, 128)
            # self.dense1 = nn.Linear(1280, 128)
            self.dense1 = nn.Linear(1024, 256)
        elif self.leak_layer == 'second':
            self.dense1 = nn.Linear(576, 256)
        else:
            self.dense1 = nn.Linear(260, 256)

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

            # pdb.set_trace()

            z = torch.cat((w0, w1), 1)
            z = torch.cat((z, w2), 1)
            z = torch.cat((z, b0), 1)
            z = torch.cat((z, b1), 1)
            z = torch.cat((z, b2), 1)
        elif self.leak_layer == 'first':
            
            w0 = self.conv0(w0.float())
            w0_size = w0.size()
            w0 = w0.reshape(-1, w0_size[1]*w0_size[2]*w0_size[3])
            z = torch.cat((w0, b0), 1)

            # pdb.set_trace()
        elif self.leak_layer == 'second':
            w1 = self.conv1(w1.float())
            w1_size = w1.size()
            w1 = w1.reshape(-1, w1_size[1]*w1_size[2]*w1_size[3])
            z = torch.cat((w1, b1), 1)

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

