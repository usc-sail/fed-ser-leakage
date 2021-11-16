python3 federated_dnn.py --dataset iemocap_msp-improv --local_epochs 5 \
                        --feature_type wav2vec --pred emotion --norm znorm --optimizer adam

python3 federated_dnn.py --dataset iemocap_msp-improv --local_epochs 5 \
                        --feature_type ComParE --pred emotion --norm znorm --optimizer adam

python3 federated_dnn.py --dataset iemocap_msp-improv --local_epochs 5 \
                        --feature_type emobase --pred emotion --norm znorm --optimizer adam
                        

#python3 federated_dnn.py --dataset crema-d --local_epochs 5 \
#                        --feature_type wav2vec --pred emotion --norm znorm --optimizer adam

#python3 federated_dnn.py --dataset msp-improv --local_epochs 5 \
#                        --feature_type wav2vec --pred emotion --norm znorm --optimizer adam

#python3 federated_dnn.py --dataset iemocap_crema-d --local_epochs 5 \
#                        --feature_type wav2vec --pred emotion --norm znorm --optimizer adam

#python3 federated_dnn.py --dataset iemocap_msp-improv --local_epochs 5 \
#                        --feature_type wav2vec --pred emotion --norm znorm --optimizer adam

#python3 federated_dnn.py --dataset msp-improv_crema-d --local_epochs 5 \
#                        --feature_type wav2vec --pred emotion --norm znorm --optimizer adam