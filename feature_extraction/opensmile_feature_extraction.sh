# This is the script for opensmile feature extraction

# audio feature part
# three datasets are iemocap, crema-d, and msp-improv dataset
# features include emobase, ComParE, and gemap opensmile features

taskset 100 python3 opensmile_feature_extraction.py --dataset msp-improv --feature_type pase_plus
taskset 100 python3 opensmile_feature_extraction.py --dataset iemocap --feature_type pase_plus
taskset 100 python3 opensmile_feature_extraction.py --dataset crema-d --feature_type pase_plus
