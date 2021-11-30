import numpy as np
import torch
import random
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
import math


emo_dict = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
affect_dict = {'low': 0, 'med': 1, 'high': 2}
gender_dict = {'F': 0, 'M': 1}

from collections import namedtuple


PrenetConfig = namedtuple(
  'PrenetConfig', ['input_size', 'hidden_size', 'num_layers', 'dropout'])

RNNConfig = namedtuple(
  'RNNConfig',
  ['input_size', 'hidden_size', 'num_layers', 'dropout', 'residual'])

class SpeechDataGenerator():
    """Speech dataset."""

    def __init__(self, data_dict, dict_keys, mode='train', input_channel=1):
        """
        Read the textfile and get the paths
        """
        self.data_dict = data_dict
        self.dict_keys = dict_keys
        self.input_channel = input_channel
        self.mode = mode

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict[self.dict_keys[idx]]
        
        import pdb
        pdb.set_trace()

        if self.input_channel == 1:
            specgram = np.expand_dims(data['data'][0], axis=0)
        else:
            specgram = data['data']
        lens = specgram.shape[1]
        
        global_data = data['global_data'][0]
        emo_id = emo_dict[data['label']]
        gen_id = gender_dict[data['gender']]
        data_set = data['dataset']
        speaker_id = data['speaker_id']
        pitch = data['pitch_data']
        
        sample = {'spec': torch.from_numpy(np.ascontiguousarray(specgram)),
                  'labels_emo': torch.from_numpy(np.ascontiguousarray(emo_id)),
                  'labels_gen': torch.from_numpy(np.ascontiguousarray(gen_id)),
                  'lengths': torch.from_numpy(np.ascontiguousarray(lens)),
                  'global': torch.from_numpy(np.ascontiguousarray(global_data)),
                  'dataset': data_set,
                  'speaker_id': speaker_id,
                  'key': self.dict_keys[idx],
                  'pitch': torch.from_numpy(np.ascontiguousarray(pitch))}
        return sample

def speech_collate(batch):
    gender = []
    emotion=[]
    specs = []
    lengths = []
    global_data = []
    data_set = []
    key = []
    speaker_id = []
    pitch = []
    for sample in batch:
        specs.append(sample['spec'])
        emotion.append((sample['labels_emo']))
        gender.append(sample['labels_gen'])
        lengths.append(sample['lengths'])
        global_data.append(sample['global'])
        data_set.append(sample['dataset'])
        key.append(sample['key'])
        speaker_id.append(sample['speaker_id'])
        pitch.append(sample['pitch'])
    return specs, emotion, gender, lengths, global_data, data_set, key, speaker_id, pitch


class AffectDataGenerator():
    """Speech Affect dataset."""

    def __init__(self, data_dict, dict_keys, mode='train', input_channel=1):
        """
        Read the textfile and get the paths
        """
        self.data_dict = data_dict
        self.dict_keys = dict_keys
        self.input_channel = input_channel
        self.mode = mode

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict[self.dict_keys[idx]]
        # import pdb
        # pdb.set_trace()

        if self.input_channel == 1:
            specgram = np.expand_dims(data['data'][0], axis=0)
        else:
            specgram = data['data']
        lens = specgram.shape[1]
        
        global_data = data['global_data'][0]
        arousal_id = affect_dict[data['arousal_label']]
        valence_id = affect_dict[data['valence_label']]
        gen_id = gender_dict[data['gender']]
        speaker_id = data['speaker_id']
        pitch = data['pitch_data']
        
        sample = {'spec': torch.from_numpy(np.ascontiguousarray(specgram)),
                  'labels_arousal': torch.from_numpy(np.ascontiguousarray(arousal_id)),
                  'labels_valence': torch.from_numpy(np.ascontiguousarray(valence_id)),
                  'labels_gen': torch.from_numpy(np.ascontiguousarray(gen_id)),
                  'lengths': torch.from_numpy(np.ascontiguousarray(lens)),
                  'global': torch.from_numpy(np.ascontiguousarray(global_data)),
                  'speaker_id': speaker_id,
                  'key': self.dict_keys[idx],
                  'pitch': torch.from_numpy(np.ascontiguousarray(pitch))}
        return sample

def affect_collate(batch):
    gender = []
    arousal=[]
    valence=[]
    specs = []
    lengths = []
    global_data = []
    key = []
    speaker_id = []
    pitch = []
    for sample in batch:
        specs.append(sample['spec'])
        arousal.append((sample['labels_arousal']))
        valence.append((sample['labels_valence']))
        gender.append(sample['labels_gen'])
        lengths.append(sample['lengths'])
        global_data.append(sample['global'])
        key.append(sample['key'])
        speaker_id.append(sample['speaker_id'])
        pitch.append(sample['pitch'])
    return specs, arousal, valence, gender, lengths, global_data, key, speaker_id, pitch

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def ReturnResultDict(truth_dict, predict_dict, dataset, pred, mode='test', loss=None, epoch=None):
    result_dict = {}
    result_dict[dataset] = {}
    result_dict[dataset]['acc'] = {}
    result_dict[dataset]['rec'] = {}
    result_dict[dataset]['loss'] = {}
    result_dict[dataset]['conf'] = {}
    
    acc_score = accuracy_score(truth_dict[dataset], predict_dict[dataset])
    rec_score = recall_score(truth_dict[dataset], predict_dict[dataset], average='macro')
    confusion_matrix_arr = np.round(confusion_matrix(truth_dict[dataset], predict_dict[dataset], normalize='true')*100, decimals=2)

    print('Total %s, %s accuracy %.3f / recall %.3f after %d' % (dataset, mode, acc_score, rec_score, epoch))
    print(confusion_matrix_arr)

    result_dict[dataset]['acc'][pred] = acc_score
    result_dict[dataset]['rec'][pred] = rec_score
    result_dict[dataset]['conf'][pred] = confusion_matrix_arr
    result_dict[dataset]['loss'][pred] = loss

    if 'combine' in dataset:
        tmp_list = ['iemocap', 'crema-d', 'msp-improv'] if dataset == 'combine' else ['iemocap', 'crema-d']
        for tmp_str in tmp_list:
            result_dict[tmp_str] = {}
            result_dict[tmp_str]['acc'] = {}
            result_dict[tmp_str]['rec'] = {}
            result_dict[tmp_str]['loss'] = {}
            result_dict[tmp_str]['conf'] = {}

            acc_score = accuracy_score(truth_dict[tmp_str], predict_dict[tmp_str])
            rec_score = recall_score(truth_dict[tmp_str], predict_dict[tmp_str], average='macro')
            confusion_matrix_arr = np.round(confusion_matrix(truth_dict[tmp_str], predict_dict[tmp_str], normalize='true')*100, decimals=2)

            print('%s: total %s accuracy %.3f / recall %.3f after %d' % (tmp_str, mode, acc_score, rec_score, epoch))
            print(confusion_matrix_arr)

            result_dict[tmp_str]['acc'][pred] = acc_score
            result_dict[tmp_str]['rec'][pred] = rec_score
            result_dict[tmp_str]['conf'][pred] = confusion_matrix_arr
    
    return result_dict


def get_class_weight(labels_dict):
    """Calculate the weights of different categories

    >>> get_class_weight({0: 633, 1: 898, 2: 641, 3: 699, 4: 799})
    {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    >>> get_class_weight({0: 5, 1: 78, 2: 2814, 3: 7914})
    {0: 7.366950709511269, 1: 4.619679795255778, 2: 1.034026384271035, 3: 1.0}
    """
    total = sum(labels_dict.values())
    max_num = max(labels_dict.values())
    mu = 1.0 / (total / max_num)
    class_weight = dict()
    for key, value in labels_dict.items():
        score = math.log(mu * total / float(value))
        # score = total / (float(value) * len(labels_dict))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight
            