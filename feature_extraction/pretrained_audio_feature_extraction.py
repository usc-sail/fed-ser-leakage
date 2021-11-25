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
import s3prl.hub as hub


sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'model'))
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'utils'))


from apc_model import APCModel
from training_tools import RNNConfig


def create_folder(folder):
    if Path.exists(folder) is False: Path.mkdir(folder)


def pretrained_feature(audio):
    
    save_feature = []
    if args.feature_type == 'wav2vec':
        with torch.inference_mode():
            features, _ = model.extract_features(audio)
        for idx in range(len(features)): save_feature.append(np.mean(features[idx].detach().cpu().numpy(), axis=1))
    elif args.feature_type == 'distilhubert' or args.feature_type == 'wav2vec2' or args.feature_type == 'vq_wav2vec':
        features = model([audio[0]])['last_hidden_state']
        save_feature.append(np.mean(features.detach().cpu().numpy(), axis=1))
        save_feature.append(np.std(features.detach().cpu().numpy(), axis=1))
        save_feature.append(np.max(features.detach().cpu().numpy(), axis=1))
    elif args.feature_type == 'cpc':
        features = model([audio[0]])['last_hidden_state']
        save_feature.append(np.mean(features.detach().cpu().numpy(), axis=1))
        save_feature.append(np.std(features.detach().cpu().numpy(), axis=1))
        save_feature.append(np.max(features.detach().cpu().numpy(), axis=1))
    else:
        # pdb.set_trace()
        features = model(audio)['last_hidden_state']
        save_feature.append(np.mean(features.detach().cpu().numpy(), axis=1))
        save_feature.append(np.std(features.detach().cpu().numpy(), axis=1))
        save_feature.append(np.max(features.detach().cpu().numpy(), axis=1))
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
    device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    if args.feature_type == 'wav2vec':
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model()
    if args.feature_type == 'wav2vec2':
        model = getattr(hub, 'wav2vec2')()
    elif args.feature_type == 'cpc':
        model = getattr(hub, 'modified_cpc')()
    elif args.feature_type == 'vq_apc':
        model = getattr(hub, 'vq_apc')()
    elif args.feature_type == 'vq_wav2vec':
        model = getattr(hub, 'vq_wav2vec')()
    elif args.feature_type == 'mockingjay':
        model = getattr(hub, 'mockingjay')()
    elif args.feature_type == 'apc':
        model = getattr(hub, 'apc')()
    elif args.feature_type == 'vq_apc':
        model = getattr(hub, 'vq_apc')()
    elif args.feature_type == 'decoar2':
        model = getattr(hub, 'decoar2')()
    elif args.feature_type == 'distilhubert':
        model = getattr(hub, 'distilhubert')()
    elif args.feature_type == 'audio_albert':
        model = getattr(hub, 'audio_albert')()
    elif args.feature_type == 'npc':
        model = getattr(hub, 'npc')()
    elif args.feature_type == 'tera':
        model = getattr(hub, 'tera')()
    model = model.to(device)

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

        
            

