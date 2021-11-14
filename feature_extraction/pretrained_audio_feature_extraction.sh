# This is the script for opensmile feature extraction

# audio feature part
# three datasets are iemocap, crema-d, and msp-improv dataset
# features include wav2vec

taskset 100 python3 pretrained_audio_feature_extraction.py --dataset msp-improv --feature_type wav2vec
taskset 100 python3 pretrained_audio_feature_extraction.py --dataset iemocap --feature_type wav2vec
taskset 100 python3 pretrained_audio_feature_extraction.py --dataset crema-d --feature_type wav2vec
