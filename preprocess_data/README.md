# Preprocess features
The scripts under this folder prepares the ready-to-go data (in pickle files) for training. 

### Quick example
The bash file federated_data_preprocess.sh provides an example of running the preprocess python file. e.g.:

```sh
taskset 100 python3 federated_data_preprocess.py --dataset crema-d --feature_type apc --pred emotion --norm znorm
```
The arg data set specifies the data set. The support data sets are IEMOCAP, MSP-Improv, and CREMA-D. 

The arg feature_type is the feature reprentation type. Please refer to README under feature extraction for more details.

The arg emotion is prediction label. Currently support SER only, the arousal and valence predictions are ongoing work.

The arg norm specifies the normalization method including z-normalization and min-max normalization. The normalization is implemented within a speaker.
