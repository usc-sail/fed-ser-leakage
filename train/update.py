import pandas as pd
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import numpy as np
import time
import warnings
import pdb
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