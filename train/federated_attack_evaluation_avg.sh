taskset 100 python3 federated_attribute_attack.py --dataset iemocap --leak_layer full \
                        --learning_rate 0.0001 --optimizer adam --model_type fed_avg \
                        --adv_dataset msp-improv_crema-d --pred emotion \
                        --feature_type distilhubert --local_epochs 5 --norm znorm --num_epochs 200

taskset 100 python3 federated_attribute_attack.py --dataset crema-d --leak_layer full \
                        --learning_rate 0.0001 --optimizer adam --model_type fed_avg \
                        --adv_dataset iemocap_msp-improv --pred emotion \
                        --feature_type distilhubert --local_epochs 5 --norm znorm --num_epochs 200

taskset 100 python3 federated_attribute_attack.py --dataset msp-improv --leak_layer full \
                        --learning_rate 0.0001 --optimizer adam --model_type fed_avg \
                        --adv_dataset iemocap_crema-d --pred emotion \
                        --feature_type distilhubert --local_epochs 5 --norm znorm --num_epochs 200
