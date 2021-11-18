from pathlib import Path
from numpy.lib.npyio import save
import pandas as pd
import re
import pickle
import numpy as np
import argparse
import pdb
from torch.nn.modules.module import T
from tqdm import tqdm


def create_folder(folder):
    if Path.exists(folder) is False:
        Path.mkdir(folder)


def write_data_dict(tmp_dict, data, label, gender, speaker_id):
    tmp_dict['label'] = label
    tmp_dict['gender'] = gender
    tmp_dict['speaker_id'] = speaker_id

    if args.dataset != 'crema-d':
        tmp_dict['arousal'] = arousal
        tmp_dict['valence'] = valence
        tmp_dict['arousal_label'] = arousal_label
        tmp_dict['valence_label'] = valence_label

    # save for normalization later
    if speaker_id not in training_norm_dict:
        training_norm_dict[speaker_id] = []
    training_norm_dict[speaker_id].append(data.copy())
    tmp_dict['data'] = data.copy()
    

def save_data_dict(save_data, label, gender, speaker_id):

    if speaker_id in test_speaker_id_arr:
        test_dict[sentence_file] = {}
        write_data_dict(test_dict[sentence_file], save_data, label, gender, speaker_id)
    elif speaker_id in validation_speaker_id_arr:
        valid_dict[sentence_file] = {}
        write_data_dict(valid_dict[sentence_file], save_data, label, gender, speaker_id)
    elif speaker_id in train_speaker_id_arr:
        training_dict[sentence_file] = {}
        write_data_dict(training_dict[sentence_file], save_data, label, gender, speaker_id)


def return_affect_label(score, low_threshold, high_threshold):
    if score <= low_threshold:
        label = 'low'
    elif low_threshold < score <= high_threshold:
        label = 'med'
    else:
        label = 'high'
    return label


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--pred', default='affect')
    parser.add_argument('--norm', default='znorm')
    parser.add_argument('--feature_type', default='mel_spec')
    parser.add_argument('--test_fold',  default='fold1')
    parser.add_argument('--test_id',  default=0)
    parser.add_argument('--train_arr', nargs='*', type=int, default=None)
    parser.add_argument('--validation_arr', nargs='*', type=int, default=None)
    parser.add_argument('--test_arr', nargs='*', type=int, default=None)
    
    args = parser.parse_args()

    # read args
    test_fold = args.test_fold
    feature_type = args.feature_type
    
    train_arr, validation_arr, test_arr = args.train_arr, args.validation_arr, args.test_arr
    
    # save preprocess file
    root_path = Path('/media/data/projects/speech-privacy')
    create_folder(root_path.joinpath('federated_learning'))
    create_folder(root_path.joinpath('federated_learning', feature_type))
    create_folder(root_path.joinpath('federated_learning', feature_type, args.pred))

    preprocess_path = root_path.joinpath('federated_learning', feature_type, args.pred)

    # feature folder
    feature_path = root_path.joinpath('federated_feature', feature_type)
    training_norm_dict = {}

    for data_set_str in [args.dataset]:

        if data_set_str in ['iemocap', 'crema-d', 'msp-improv']:
            with open(feature_path.joinpath(data_set_str, 'data.pkl'), 'rb') as f:
                data_dict = pickle.load(f)
        
        training_dict, valid_dict, test_dict = {}, {}, {}
        
        if data_set_str == 'msp-improv':
            # data root folder
            sentence_file_list = list(data_dict.keys())
            sentence_file_list.sort()

            speaker_id_arr = ['M01', 'F01', 'M02', 'F02', 'M03', 'F03', 'M04', 'F04', 'M05', 'F05', 'M06', 'F06']

            train_speaker_id_arr = [speaker_id_arr[tmp_idx] for tmp_idx in train_arr]
            validation_speaker_id_arr = [speaker_id_arr[tmp_idx] for tmp_idx in validation_arr]
            test_speaker_id_arr = [speaker_id_arr[tmp_idx] for tmp_idx in test_arr]
            
            # data root folder
            data_root_path = Path('/media/data').joinpath('sail-data')
            data_str = 'MSP-IMPROV'

            evaluation_path = data_root_path.joinpath(data_str, data_str, 'Evalution.txt')
            with open(str(evaluation_path)) as f:
                evaluation_lines = f.readlines()

            label_dict, arousal_dict, valence_dict = {}, {}, {}
            for evaluation_line in evaluation_lines:
                if 'UTD-' in evaluation_line:
                    file_name = evaluation_line.split('.avi')[0]
                    emotion = evaluation_line.split('; ')[1][0]
                    arousal = float(evaluation_line.split('; ')[2][2:])
                    valence = float(evaluation_line.split('; ')[3][2:])
                    label_dict['MSP-'+file_name[4:]] = emotion
                    arousal_dict['MSP-'+file_name[4:]] = arousal
                    valence_dict['MSP-'+file_name[4:]] = valence
                    
            for sentence_file in tqdm(sentence_file_list, ncols=100, miniters=100):
                sentence_part = sentence_file.split('-')
                recording_type = sentence_part[-2][-1:]
                gender = sentence_part[-3][:1]
                speaker_id = sentence_part[-3]
                emotion = label_dict[sentence_file]
                arousal = arousal_dict[sentence_file]
                valence = valence_dict[sentence_file]

                # we keep improv data only
                if recording_type == 'P':
                    continue
                if recording_type == 'R':
                    continue
                
                if emotion == 'N':
                    label = 'neu'
                elif emotion == 'S':
                    label = 'sad'
                elif emotion == 'H':
                    label = 'hap'
                elif emotion == 'A':
                    label = 'ang'
                else:
                    label = 'oth'
                
                arousal_label = return_affect_label(arousal, 2.75, 3.25)
                valence_label = return_affect_label(valence, 2.75, 3.25)

                data = data_dict[sentence_file]
                if args.feature_type == 'wav2vec':
                    save_data = np.array(data['data'])[:, 0, :].flatten()
                else:
                    save_data = np.array(data['data'])[0]

                if args.pred == 'emotion': 
                    if label != 'oth':
                        save_data_dict(save_data, label, gender, speaker_id)
                else:
                    save_data_dict(save_data, label, gender, speaker_id)

        elif data_set_str == 'crema-d':
            
            # speaker id for training, validation, and test
            train_speaker_id_arr = [tmp_idx for tmp_idx in train_arr]
            validation_speaker_id_arr = [tmp_idx for tmp_idx in validation_arr]
            test_speaker_id_arr = [tmp_idx for tmp_idx in test_arr]
            
            # data root folder
            data_root_path = Path('/media/data').joinpath('public-data', 'SER')
            demo_df = pd.read_csv(str(data_root_path.joinpath(data_set_str, 'VideoDemographics.csv')), index_col=0)
            rating_df = pd.read_csv(str(data_root_path.joinpath(data_set_str, 'summaryTable.csv')), index_col=1)
            sentence_file_list = list(data_root_path.joinpath(data_set_str).glob('*.wav'))
           
            sentence_file_list.sort()
            speaker_id_arr = np.arange(1001, 1092, 1)
            
            for sentence_file in tqdm(sentence_file_list, ncols=100, miniters=100):
                sentence_file = str(sentence_file).split('/')[-1].split('.wav')[0]
                sentence_part = sentence_file.split('_')
                
                speaker_id = int(sentence_part[0])
                emotion = rating_df.loc[sentence_file, 'MultiModalVote']
                
                if emotion == 'A' or emotion == 'N' or emotion == 'S' or emotion == 'H':
                    if sentence_file not in data_dict:
                        continue
                    
                    if emotion == 'N':
                        label = 'neu'
                    elif emotion == 'S':
                        label = 'sad'
                    elif emotion == 'H':
                        label = 'hap'
                    elif emotion == 'A':
                        label = 'ang'
                    
                    data = data_dict[sentence_file]
                    if args.feature_type == 'wav2vec':
                        save_data = np.array(data['data'])[:, 0, :].flatten()
                    else:
                        save_data = np.array(data['data'])[0]

                    session_id = int(sentence_part[0])
                    gender = 'M' if demo_df.loc[int(session_id), 'Sex'] == 'Male' else 'F'
                    speaker_id = int(sentence_file.split('_')[0])
                    save_data_dict(save_data, label, gender, speaker_id)

        elif data_set_str == 'iemocap':

            speaker_id_arr = ['Ses01F', 'Ses01M', 'Ses02F', 'Ses02M', 'Ses03F', 'Ses03M', 'Ses04F', 'Ses04M', 'Ses05F', 'Ses05M']
            
            # speaker id for training, validation, and test
            train_speaker_id_arr = [speaker_id_arr[tmp_idx] for tmp_idx in train_arr]
            validation_speaker_id_arr = [speaker_id_arr[tmp_idx] for tmp_idx in validation_arr]
            test_speaker_id_arr = [speaker_id_arr[tmp_idx] for tmp_idx in test_arr]
          
            # data root folder
            data_root_path = Path('/media/data').joinpath('sail-data')
            for session_id in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
                ground_truth_path_list = list(data_root_path.joinpath(data_set_str, session_id, 'dialog', 'EmoEvaluation').glob('*.txt'))
                for ground_truth_path in tqdm(ground_truth_path_list, ncols=100, miniters=100):
                    with open(str(ground_truth_path)) as f:
                        file_content = f.read()
                        useful_regex = re.compile(r'\[.+\]\n', re.IGNORECASE)
                        label_lines = re.findall(useful_regex, file_content)
                        for line in label_lines:
                            if 'Ses' in line:
                                sentence_file = line.split('\t')[-3]
                                label = line.split('\t')[-2]

                                arousal = float(line.split('\t')[-1].split(',')[0][1:])
                                valence = float(line.split('\t')[-1].split(',')[1][1:])
                                
                                arousal_label = return_affect_label(arousal, 2.75, 3.25)
                                valence_label = return_affect_label(valence, 2.75, 3.25)
                            
                                data = data_dict[sentence_file]
                                if args.feature_type == 'wav2vec':
                                    save_data = np.array(data['data'])[:, 0, :].flatten()
                                else:
                                    save_data = np.array(data['data'])[0].flatten()
                                
                                gender = sentence_file.split('_')[-1][0]
                                speaker_id = sentence_file.split('_')[0][:-1] + gender
                                
                                if 'impro' not in line: continue
                                if args.pred == 'emotion':
                                    if label == 'ang' or label == 'neu' or label == 'sad' or label == 'hap' or label == 'exc':
                                        if label == 'exc': 
                                            label = 'hap'
                                        save_data_dict(save_data, label, gender, speaker_id)
                                else:
                                    save_data_dict(save_data, label, gender, speaker_id)

        # if we are not trying to combine the dataset, we should do the normalization or augmentation
        speaker_norm_dict = {}
        for speaker_id in training_norm_dict:
            norm_data_list = training_norm_dict[speaker_id]
            speaker_norm_dict[speaker_id] = {}
            speaker_norm_dict[speaker_id]['mean'] = np.nanmean(np.array(norm_data_list), axis=0)
            speaker_norm_dict[speaker_id]['std'] = np.nanstd(np.array(norm_data_list), axis=0)
            speaker_norm_dict[speaker_id]['min'] = np.nanmin(np.array(norm_data_list), axis=0)
            speaker_norm_dict[speaker_id]['max'] = np.nanmax(np.array(norm_data_list), axis=0)

        for tmp_dict in [training_dict, valid_dict, test_dict]:
            for file_name in tmp_dict:
                speaker_id = tmp_dict[file_name]['speaker_id']
                if args.norm == 'znorm':
                    tmp_data = (tmp_dict[file_name]['data'].copy() - speaker_norm_dict[speaker_id]['mean']) / (speaker_norm_dict[speaker_id]['std']+1e-5)
                elif args.norm == 'min_max':
                    tmp_data = (tmp_dict[file_name]['data'].copy() - speaker_norm_dict[speaker_id]['min']) / (speaker_norm_dict[speaker_id]['max'] - speaker_norm_dict[speaker_id]['min'])
                    tmp_data = tmp_data * 2 - 1
                tmp_dict[file_name]['data'] = tmp_data.copy()


        create_folder(preprocess_path.joinpath(data_set_str))
        create_folder(preprocess_path.joinpath(data_set_str, test_fold))

        f = open(str(preprocess_path.joinpath(data_set_str, test_fold, 'training_'+args.norm+'.pkl')), "wb")
        pickle.dump(training_dict, f)
        f.close()

        f = open(str(preprocess_path.joinpath(data_set_str, test_fold, 'validation_'+args.norm+'.pkl')), "wb")
        pickle.dump(valid_dict, f)
        f.close()

        f = open(str(preprocess_path.joinpath(data_set_str, test_fold, 'test_'+args.norm+'.pkl')), "wb")
        pickle.dump(test_dict, f)
        f.close()
        