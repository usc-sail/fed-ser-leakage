from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle, argparse, re, pdb
from sklearn.model_selection import KFold

emo_map_dict = {'N': 'neu', 'S': 'sad', 'H': 'hap', 'A': 'ang'}

speaker_id_arr_dict = {'msp-improv': np.arange(0, 12, 1), 
                       'crema-d': np.arange(1001, 1092, 1),
                       'iemocap': np.arange(0, 10, 1)}

def write_data_dict(tmp_dict, data, label, gender, speaker_id):
    tmp_dict['label'], tmp_dict['gender'], tmp_dict['speaker_id']  = label, gender, speaker_id
    # save for normalization later
    if speaker_id not in training_norm_dict: training_norm_dict[speaker_id] = []
    training_norm_dict[speaker_id].append(data.copy())
    tmp_dict['data'] = data.copy()
    
def save_data_dict(save_data, label, gender, speaker_id):
    if speaker_id in test_speaker_id_arr:
        test_dict[sentence_file] = {}
        write_data_dict(test_dict[sentence_file], save_data, label, gender, speaker_id)
    elif speaker_id in train_speaker_id_arr:
        training_dict[sentence_file] = {}
        write_data_dict(training_dict[sentence_file], save_data, label, gender, speaker_id)

if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--norm', default='znorm')
    parser.add_argument('--feature_type', default='emobase')
    parser.add_argument('--data_dir', default='/media/data/public-data/SER')
    parser.add_argument('--save_dir', default='/media/data/projects/speech-privacy')
    args = parser.parse_args()

    # get the 5 different test folds
    speaker_id_arr = speaker_id_arr_dict[args.dataset]    
    train_array, test_array = [], []
    
    # read args
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    fold_idx, feature_type, data_set_str = 1, args.feature_type, args.dataset

    for train_index, test_index in kf.split(speaker_id_arr):
        
        # 80% are training (80% of data on a client is for training, rest validation), and 20% are test
        train_arr, test_arr = speaker_id_arr[train_index], speaker_id_arr[test_index]
        test_fold = 'fold'+str(fold_idx)
        print('Process %s training set with test %s' % (data_set_str, test_fold))
        
        # save preprocess file dir
        preprocess_path = Path(args.save_dir).joinpath('federated_learning', feature_type, args.pred)
        Path.mkdir(preprocess_path, parents=True, exist_ok=True)

        # feature folder
        feature_path = Path(args.save_dir).joinpath('federated_feature', feature_type)
        training_norm_dict = {}

        # read features
        with open(feature_path.joinpath(data_set_str, 'data.pkl'), 'rb') as f:
            data_dict = pickle.load(f)
        
        training_dict, test_dict = {}, {}
        if data_set_str == 'msp-improv':
            # data root folder
            sentence_file_list = list(data_dict.keys())
            sentence_file_list.sort()
            speaker_id_list = ['M01', 'F01', 'M02', 'F02', 'M03', 'F03', 'M04', 'F04', 'M05', 'F05', 'M06', 'F06']

            train_speaker_id_arr = [speaker_id_list[tmp_idx] for tmp_idx in train_arr]
            test_speaker_id_arr = [speaker_id_list[tmp_idx] for tmp_idx in test_arr]
            print('Train speaker:')
            print(train_speaker_id_arr)
            print('Test speaker:')
            print(test_speaker_id_arr)
            
            # data root folder
            evaluation_path = Path(args.data_dir).joinpath('Evalution.txt')
            with open(str(evaluation_path)) as f:
                evaluation_lines = f.readlines()

            label_dict = {}
            for evaluation_line in evaluation_lines:
                if 'UTD-' in evaluation_line:
                    file_name = 'MSP-'+evaluation_line.split('.avi')[0][4:]
                    label_dict[file_name] = evaluation_line.split('; ')[1][0]
                    
            for sentence_file in tqdm(sentence_file_list, ncols=100, miniters=100):
                sentence_part = sentence_file.split('-')
                recording_type = sentence_part[-2][-1:]
                gender, speaker_id, emotion = sentence_part[-3][:1], sentence_part[-3], label_dict[sentence_file]
                
                # we keep improv data only
                if recording_type == 'P' or recording_type == 'R': continue
                if emotion not in emo_map_dict: continue
                label, data = emo_map_dict[emotion], data_dict[sentence_file]
                save_data = np.array(data['data'])[0] if args.feature_type == 'emobase' else np.array(data['data'])[0, 0, :].flatten()
                save_data_dict(save_data, label, gender, speaker_id)

        elif data_set_str == 'crema-d':
            
            # speaker id for training and test
            train_speaker_id_arr, test_speaker_id_arr = [tmp_idx for tmp_idx in train_arr], [tmp_idx for tmp_idx in test_arr]
            print('Train speaker:')
            print(train_speaker_id_arr)
            print('Test speaker:')
            print(test_speaker_id_arr)

            # data root folder
            demo_df = pd.read_csv(str(Path(args.data_dir).joinpath('processedResults', 'VideoDemographics.csv')), index_col=0)
            rating_df = pd.read_csv(str(Path(args.data_dir).joinpath('processedResults', 'summaryTable.csv')), index_col=1)
            sentence_file_list = list(Path(args.data_dir).joinpath('AudioWAV').glob('*.wav'))
            sentence_file_list.sort()
            
            for sentence_file in tqdm(sentence_file_list, ncols=100, miniters=100):
                sentence_file = str(sentence_file).split('/')[-1].split('.wav')[0]
                sentence_part = sentence_file.split('_')
                speaker_id = int(sentence_part[0])
                emotion = rating_df.loc[sentence_file, 'MultiModalVote']
                
                if sentence_file not in data_dict: continue
                if emotion not in emo_map_dict: continue
                label, data = emo_map_dict[emotion], data_dict[sentence_file]
                save_data = np.array(data['data'])[0] if args.feature_type == 'emobase' else np.array(data['data'])[0, 0, :].flatten()
                gender = 'M' if demo_df.loc[int(sentence_part[0]), 'Sex'] == 'Male' else 'F'
                save_data_dict(save_data, label, gender, speaker_id)

        elif data_set_str == 'iemocap':
            # speaker id for training, validation, and test
            speaker_id_list = ['Ses01F', 'Ses01M', 'Ses02F', 'Ses02M', 'Ses03F', 'Ses03M', 'Ses04F', 'Ses04M', 'Ses05F', 'Ses05M']
            train_speaker_id_arr = [speaker_id_list[tmp_idx] for tmp_idx in train_arr]
            test_speaker_id_arr = [speaker_id_list[tmp_idx] for tmp_idx in test_arr]
            print('Train speaker:')
            print(train_speaker_id_arr)
            print('Test speaker:')
            print(test_speaker_id_arr)
        
            for session_id in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
                ground_truth_path_list = list(Path(args.data_dir).joinpath(session_id, 'dialog', 'EmoEvaluation').glob('*.txt'))
                for ground_truth_path in tqdm(ground_truth_path_list, ncols=100, miniters=100):
                    with open(str(ground_truth_path)) as f:
                        file_content = f.read()
                        useful_regex = re.compile(r'\[.+\]\n', re.IGNORECASE)
                        label_lines = re.findall(useful_regex, file_content)
                        for line in label_lines:
                            if 'Ses' in line:
                                sentence_file = line.split('\t')[-3]
                                gender = sentence_file.split('_')[-1][0]
                                speaker_id = sentence_file.split('_')[0][:-1] + gender
                                label, data = line.split('\t')[-2], data_dict[sentence_file]
                                save_data = np.array(data['data'])[0] if args.feature_type == 'emobase' else np.array(data['data'])[0, 0, :].flatten()
                                
                                if 'impro' not in line: continue
                                if label == 'ang' or label == 'neu' or label == 'sad' or label == 'hap' or label == 'exc':
                                    if label == 'exc': label = 'hap'
                                    save_data_dict(save_data, label, gender, speaker_id)
                                
        # if we are not trying to combine the dataset, we should do the normalization or augmentation
        speaker_norm_dict = {}
        for speaker_id in training_norm_dict:
            norm_data_list = training_norm_dict[speaker_id]
            speaker_norm_dict[speaker_id] = {}
            speaker_norm_dict[speaker_id]['mean'] = np.nanmean(np.array(norm_data_list), axis=0)
            speaker_norm_dict[speaker_id]['std'] = np.nanstd(np.array(norm_data_list), axis=0)

        for tmp_dict in [training_dict, test_dict]:
            for file_name in tmp_dict:
                speaker_id = tmp_dict[file_name]['speaker_id']
                if args.norm == 'znorm': 
                    tmp_data = (tmp_dict[file_name]['data'].copy() - speaker_norm_dict[speaker_id]['mean']) / (speaker_norm_dict[speaker_id]['std']+1e-5)
                tmp_dict[file_name]['data'] = tmp_data.copy()

        Path.mkdir(preprocess_path.joinpath(data_set_str, test_fold), parents=True, exist_ok=True)
        f = open(str(preprocess_path.joinpath(data_set_str, test_fold, 'training_'+args.norm+'.pkl')), "wb")
        pickle.dump(training_dict, f)
        f.close()

        f = open(str(preprocess_path.joinpath(data_set_str, test_fold, 'test_'+args.norm+'.pkl')), "wb")
        pickle.dump(test_dict, f)
        f.close()

        fold_idx += 1
        del training_dict, test_dict
