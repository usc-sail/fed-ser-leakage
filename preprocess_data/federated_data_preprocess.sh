# This is the script for training data preprocess
# audio feature part
# three datasets are iemocap, crema-d, and msp-improv dataset

taskset 100 python3 federated_data_preprocess.py --dataset crema-d --feature_type wav2vec --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset iemocap --feature_type wav2vec --pred emotion --norm znorm
taskset 100 python3 federated_data_preprocess.py --dataset msp-improv --feature_type wav2vec --pred emotion --norm znorm