from pathlib import Path
import numpy as np
import pickle
import pandas as pd
import pdb
from moviepy.editor import *
import argparse
import torchaudio
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'model'))
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'utils'))


from apc_model import APCModel
from utils import RNNConfig


def create_folder(folder):
    if Path.exists(folder) is False: Path.mkdir(folder)


def pretrained_feature(audio):
    with torch.inference_mode():
        features, _ = model.extract_features(audio)
        
    save_feature = []
    for idx in range(len(features)): save_feature.append(np.mean(features[idx].detach().cpu().numpy(), axis=1))
    
    del features
    return save_feature


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--feature_type', default='wav2vec')
    args = parser.parse_args()

    # save feature file
    root_path = Path('/media/data/projects/speech-privacy')
    create_folder(root_path.joinpath('federated_feature'))
    create_folder(root_path.joinpath('federated_feature', args.feature_type))
    save_feat_path = root_path.joinpath('federated_feature', args.feature_type)
    
    audio_features = {}

    # Model related
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    if args.feature_type == 'wav2vec':
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model().to(device)
    elif args.feature_type == 'apc':
        rnn_config = RNNConfig(input_size=80, hidden_size=512, num_layers=3, dropout=0.)
        pretrained_apc = APCModel(mel_dim=80, prenet_config=None, rnn_config=rnn_config).cuda()
        pretrained_weights_path =  os.path.join(os.path.abspath(os.path.curdir), '..', 'model', 'bs32-rhl3-rhs512-rd0-adam-res-ts3.pt')
        pretrained_apc.load_state_dict(torch.load(pretrained_weights_path))

    # msp-podcast
    if args.dataset == 'msp-podcast':
        # data root folder
        data_root_path = Path('/media/data').joinpath('sail-data')
        data_str = 'MSP-podcast'
        label_df = pd.read_csv(data_root_path.joinpath(data_str, 'Labels', 'labels_concensus.csv'), index_col=0)
        
        audio_features['train'] = {}
        audio_features['validate'] = {}
        audio_features['test'] = {}
        
        for file_name in tqdm(list(label_df.index), ncols=100, miniters=100):
            file_path = data_root_path.joinpath(data_str, 'Audios', file_name)
            if Path.exists(file_path) is False:
                continue

            speaker_id = label_df.loc[file_name, 'SpkrID']
            gender = label_df.loc[file_name, 'Gender']

            if 'Test2' in label_df.loc[file_name, 'Split_Set']: continue
            if 'Unknown' in speaker_id or 'Unknown' in gender: continue

            audio, sample_rate = torchaudio.load(str(file_path))
            transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
            audio = transform_model(audio)
        
            audio_features[file_name] = {}
            
    # msp-improv
    elif args.dataset == 'msp-improv':
        # data root folder
        data_root_path = Path('/media/data').joinpath('sail-data')
        session_list = [x.parts[-1] for x in data_root_path.joinpath('MSP-IMPROV', 'MSP-IMPROV', 'Audio').iterdir() if 'session' in x.parts[-1]]
        session_list.sort()
        
        for session_id in session_list:
            file_path_list = list(data_root_path.joinpath('MSP-IMPROV', 'MSP-IMPROV', 'Audio', session_id).glob('**/**/*.wav'))
            for file_path in tqdm(file_path_list, ncols=50, miniters=100):
                file_name = file_path.parts[-1].split('.wav')[0].split('/')[-1]
                print("process %s %s" % (session_id, file_name))

                audio, sample_rate = torchaudio.load(str(file_path))
                transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
                audio = transform_model(audio)
                audio = audio.to(device)
            
                audio_features[file_name] = {}
                save_feature = pretrained_feature(audio)
                audio_features[file_name]['data'] = save_feature
                audio_features[file_name]['session'] = session_id
                del save_feature

    # crema-d
    elif args.dataset == 'crema-d':
        # data root folder
        data_root_path = Path('/media/data').joinpath('public-data', 'SER')
        file_list = [x for x in data_root_path.joinpath(args.dataset).iterdir() if '.wav' in x.parts[-1]]
        file_list.sort()

        for file_path in tqdm(file_list, ncols=100, miniters=100):
            print('process %s' % file_path)
            if '1076_MTI_SAD_XX.wav' in str(file_path):
                continue
            file_name = file_path.parts[-1].split('.wav')[0]
            audio, sample_rate = torchaudio.load(str(file_path))
            audio = audio.to(device)

            audio_features[file_name] = {}
            save_feature = pretrained_feature(audio)
            audio_features[file_name]['data'] = save_feature
            del save_feature
            
    # iemocap
    elif args.dataset == 'iemocap':
        # data root folder
        data_root_path = Path('/media/data').joinpath('sail-data')
        session_list = [x.parts[-1] for x in data_root_path.joinpath(args.dataset).iterdir() if  'Session' in x.parts[-1]]
        session_list.sort()
        for session_id in session_list:
            file_path_list = list(data_root_path.joinpath(args.dataset, session_id, 'sentences', 'wav').glob('**/*.wav'))
            for file_path in tqdm(file_path_list, ncols=100, miniters=100):
                file_name = file_path.parts[-1].split('.wav')[0].split('/')[-1]
                audio, sample_rate = torchaudio.load(str(file_path))
                audio = audio.to(device)

                audio_features[file_name] = {}
                save_feature = pretrained_feature(audio)
                audio_features[file_name]['data'] = save_feature
                del save_feature

    create_folder(save_feat_path.joinpath(args.dataset))
    save_path = str(save_feat_path.joinpath(args.dataset, 'data.pkl'))
    with open(save_path, 'wb') as handle:
        pickle.dump(audio_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
            

