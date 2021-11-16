# python3 federated_dnn_mem_adv_infer_encoder.py --dataset iemocap_msp-improv \
#                        --adv_dataset crema-d --win_len 500 \
#                        --feature_type mfcc --shift 0 --pred emotion --local_epochs 5 \
#                        --input_spec_size 40 --norm znorm --optimizer adam

# python3 federated_dnn_mem_adv_infer_encoder.py --dataset iemocap_crema-d \
#                        --adv_dataset msp-improv --win_len 500 \
#                        --feature_type mfcc --shift 0 --pred emotion --local_epochs 5 \
#                        --input_spec_size 40 --norm znorm --optimizer adam

# python3 federated_dnn_mem_adv_infer_encoder.py --dataset msp-improv_crema-d \
#                        --adv_dataset iemocap --win_len 500 \
#                        --feature_type mfcc --shift 0 --pred emotion --local_epochs 5 \
#                        --input_spec_size 40 --norm znorm --optimizer adam

# python3 federated_dnn_mem_adv_infer_encoder.py --dataset msp-improv --leak_layer full \
#                        --adv_dataset iemocap_crema-d --win_len 200 --model_type 2d-cnn-lstm \
#                        --feature_type mel_spec --shift 1 --pred emotion --local_epochs 5 \
#                        --input_spec_size 128 --norm znorm --optimizer adam

python3 federated_attribute_attack.py --dataset iemocap --leak_layer first \
                        --adv_dataset msp-improv_crema-d --pred emotion --model_type dnn \
                        --feature_type wav2vec --local_epochs 5 --norm znorm --optimizer adam

python3 federated_attribute_attack.py --dataset crema-d --leak_layer first \
                        --adv_dataset iemocap_msp-improv --pred emotion --model_type dnn \
                        --feature_type wav2vec --local_epochs 5 --norm znorm --optimizer adam

python3 federated_attribute_attack.py --dataset msp-improv --leak_layer first \
                        --adv_dataset iemocap_crema-d --pred emotion --model_type dnn \
                        --feature_type wav2vec --local_epochs 5 --norm znorm --optimizer adam

# python3 federated_attribute_attack.py --dataset crema-d --leak_layer first \
#                        --adv_dataset iemocap_msp-improv --win_len 500 --model_type dnn \
#                        --feature_type wav2vec --shift 0 --pred emotion --local_epochs 5 \
#                        --input_spec_size 128 --norm znorm --optimizer adam

# python3 federated_dnn_mem_adv_infer_encoder.py --dataset msp-improv --leak_layer full \
#                        --adv_dataset iemocap_crema-d --win_len 200 --model_type 2d-cnn-lstm \
#                        --feature_type mel_spec --shift 1 --pred emotion --local_epochs 5 \
#                        --input_spec_size 128 --norm znorm --optimizer adam

# python3 federated_dnn_mem_adv_infer_encoder.py --dataset msp-improv --leak_layer first \
#                        --adv_dataset iemocap_crema-d --win_len 200 --model_type 2d-cnn-lstm \
#                        --feature_type mel_spec --shift 1 --pred emotion --local_epochs 5 \
#                        --input_spec_size 128 --norm znorm --optimizer adam

