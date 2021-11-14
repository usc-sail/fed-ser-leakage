from pathlib import Path
import numpy as np
import pickle
import pandas as pd
import python_speech_features as ps
import pdb
from moviepy.editor import *
import argparse
import torchaudio
import torch
from tqdm import tqdm
import opensmile


def feature_extraction(audio, n_fft=1024, feature_len=128):

    window_size = n_fft
    window_hop = 160
    n_mels = feature_len
    window_fn = torch.hann_window

    audio_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=int(window_size),
        hop_length=int(window_hop),
        window_fn=window_fn,
    )

    audio_amp_to_db = torchaudio.transforms.AmplitudeToDB()
    return audio_amp_to_db(audio_transform(audio).detach())


def create_folder(folder):
    if Path.exists(folder) is False:
        Path.mkdir(folder)


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--feature_type', default='mel_spec')
    args = parser.parse_args()

    feature_type = args.feature_type

    # save feature file
    root_path = Path('/media/data/projects/speech-privacy')
    create_folder(root_path.joinpath('feature'))
    create_folder(root_path.joinpath('feature', feature_type))
    save_feat_path = root_path.joinpath('feature', feature_type)
    
    audio_features = {}

    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)

    emobase_smile = opensmile.Smile(feature_set=opensmile.FeatureSet.emobase,
                            feature_level=opensmile.FeatureLevel.Functionals)

    # Model related
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)

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
            if feature_type == 'mfcc':
                audio_features[file_name]['mfcc'] = mfcc(audio)
            else:
                audio_features[file_name]['mel1'] = mel_spectrogram(audio, n_fft=800, feature_len=feature_len)
                audio_features[file_name]['mel2'] = mel_spectrogram(audio, n_fft=1600, feature_len=feature_len)
            audio_features[file_name]['gemaps'] = np.array(smile.process_file(str(file_path)))
            audio_features[file_name]['emobase'] = np.array(emobase_smile.process_file(str(file_path)))

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
                with torch.inference_mode():
                    features, _ = model.extract_features(audio)
                
                save_feature = []
                for idx in range(len(features)): save_feature.append(np.mean(features[idx].detach().cpu().numpy(), axis=1))
                audio_features[file_name][feature_type] = save_feature
                
                audio_features[file_name]['gemaps'] = np.array(smile.process_file(str(file_path)))
                audio_features[file_name]['emobase'] = np.array(emobase_smile.process_file(str(file_path)))
                audio_features[file_name]['session'] = session_id
                del features
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
            with torch.inference_mode():
                features, _ = model.extract_features(audio)
            
            save_feature = []
            for idx in range(len(features)): save_feature.append(np.mean(features[idx].detach().cpu().numpy(), axis=1))
            audio_features[file_name][feature_type] = save_feature

            audio_features[file_name]['gemaps'] = np.array(smile.process_file(str(file_path)))
            audio_features[file_name]['emobase'] = np.array(emobase_smile.process_file(str(file_path)))
            del features
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
                with torch.inference_mode():
                    features, _ = model.extract_features(audio)
                    
                save_feature = []
                for idx in range(len(features)): save_feature.append(np.mean(features[idx].detach().cpu().numpy(), axis=1))
                audio_features[file_name][feature_type] = save_feature

                audio_features[file_name]['gemaps'] = np.array(smile.process_file(str(file_path)))
                audio_features[file_name]['emobase'] = np.array(emobase_smile.process_file(str(file_path)))
                del features
                del save_feature

    create_folder(save_feat_path.joinpath(args.dataset))
    save_path = str(save_feat_path.joinpath(args.dataset, 'data.pkl'))
    with open(save_path, 'wb') as handle:
        pickle.dump(audio_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
            

