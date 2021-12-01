# This is the script for training data preprocess
# audio feature part
# three datasets are iemocap, crema-d, and msp-improv dataset
# cpc, apc, tera, decoar2, audio_albert, distilhubert
taskset 100 python3 federated_data_preprocess.py --dataset crema-d --feature_type vq_apc --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset iemocap --feature_type vq_apc --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset msp-improv --feature_type vq_apc --pred emotion --norm znorm

taskset 100 python3 federated_data_preprocess.py --dataset crema-d --feature_type vq_wav2vec --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset iemocap --feature_type vq_wav2vec --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset msp-improv --feature_type vq_wav2vec --pred emotion --norm znorm

taskset 100 python3 federated_data_preprocess.py --dataset crema-d --feature_type audio_albert --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset iemocap --feature_type audio_albert --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset msp-improv --feature_type audio_albert --pred emotion --norm znorm

taskset 100 python3 federated_data_preprocess.py --dataset crema-d --feature_type tera --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset iemocap --feature_type tera --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset msp-improv --feature_type tera --pred emotion --norm znorm

taskset 100 python3 federated_data_preprocess.py --dataset crema-d --feature_type apc --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset iemocap --feature_type apc --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset msp-improv --feature_type apc --pred emotion --norm znorm

taskset 100 python3 federated_data_preprocess.py --dataset crema-d --feature_type cpc --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset iemocap --feature_type cpc --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset msp-improv --feature_type cpc --pred emotion --norm znorm

taskset 100 python3 federated_data_preprocess.py --dataset crema-d --feature_type decoar2 --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset iemocap --feature_type decoar2 --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset msp-improv --feature_type decoar2 --pred emotion --norm znorm

taskset 100 python3 federated_data_preprocess.py --dataset crema-d --feature_type distilhubert --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset iemocap --feature_type distilhubert --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset msp-improv --feature_type distilhubert --pred emotion --norm znorm

taskset 100 python3 federated_data_preprocess.py --dataset crema-d --feature_type mockingjay --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset iemocap --feature_type mockingjay --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset msp-improv --feature_type mockingjay --pred emotion --norm znorm

taskset 100 python3 federated_data_preprocess.py --dataset crema-d --feature_type wav2vec2 --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset iemocap --feature_type wav2vec2 --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset msp-improv --feature_type wav2vec2 --pred emotion --norm znorm

taskset 100 python3 federated_data_preprocess.py --dataset crema-d --feature_type npc --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset iemocap --feature_type npc --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset msp-improv --feature_type npc --pred emotion --norm znorm
