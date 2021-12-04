# Training scripts
The scripts under this folder trains the SER model in FL and attack model. 

### SER training in FL
The bash file federated_ser_classifier.sh provides an example of running the preprocess python file. e.g.:

```sh
python3 federated_ser_classifier.py --dataset iemocap --local_epochs 5 --learning_rate 0.05 --model_type fed_sgd \
                        --feature_type apc --pred emotion --norm znorm --optimizer adam --dropout 0.2 --num_epochs 200

```
- The arg `dataset` specifies the data set. The support data sets are IEMOCAP (iemocap), MSP-Improv (msp-improv), and CREMA-D (crema-d). 

- The arg `feature_type` is the feature reprentation type. Please refer to README under feature extraction for more details.

- The arg `pred` is prediction label. Currently support SER only, the arousal and valence predictions are ongoing work.

- The arg `norm` specifies the normalization method including z-normalization and min-max normalization. The normalization is implemented within a speaker.

- The arg `model_type` specifies the FL algorithm: fed_sgd and fed_avg.

- The arg `dropout` specifies the dropout value in the first dense layer.

- The arg `local_epochs` specifies the local epoch, ignored in fed_sgd.

- The arg `num_epochs` specifies the global epoch in FL.

- The arg `learning_rate` specifies the learning rate in FL.


