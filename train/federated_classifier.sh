python3 federated_dnn.py --dataset crema-d --local_epochs 5 --learning_rate 0.0005 \
                        --feature_type vq_wav2vec --pred emotion --norm znorm --optimizer adam

python3 federated_dnn.py --dataset iemocap --local_epochs 5 --learning_rate 0.0005 \
                        --feature_type vq_wav2vec --pred emotion --norm znorm --optimizer adam

python3 federated_dnn.py --dataset msp-improv --local_epochs 5 --learning_rate 0.0005 \
                        --feature_type vq_wav2vec --pred emotion --norm znorm --optimizer adam

python3 federated_dnn.py --dataset iemocap_crema-d --local_epochs 5 --learning_rate 0.0005 \
                        --feature_type vq_wav2vec --pred emotion --norm znorm --optimizer adam

python3 federated_dnn.py --dataset iemocap_msp-improv --local_epochs 5 --learning_rate 0.0005 \
                        --feature_type vq_wav2vec --pred emotion --norm znorm --optimizer adam

python3 federated_dnn.py --dataset msp-improv_crema-d --local_epochs 5 --learning_rate 0.0005 \
                        --feature_type vq_wav2vec --pred emotion --norm znorm --optimizer adam
