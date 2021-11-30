taskset 200 python3 federated_attribute_attack.py --dataset iemocap --leak_layer second \
                        --learning_rate 0.0005 --optimizer adam --model_type fed_avg \
                        --adv_dataset msp-improv_crema-d --pred emotion --device 0 \
                        --feature_type decoar2 --local_epochs 1 --norm znorm --num_epochs 200 \
                        --model_learning_rate 0.0001


taskset 200 python3 federated_attribute_attack.py --dataset crema-d --leak_layer second \
                        --learning_rate 0.0005 --optimizer adam --model_type fed_avg \
                        --adv_dataset iemocap_msp-improv --pred emotion --device 0 \
                        --feature_type decoar2 --local_epochs 1 --norm znorm --num_epochs 200 \
                        --model_learning_rate 0.0001


taskset 200 python3 federated_attribute_attack.py --dataset msp-improv --leak_layer second \
                        --learning_rate 0.0005 --optimizer adam --model_type fed_avg \
                        --adv_dataset iemocap_crema-d --pred emotion --device 0 \
                        --feature_type decoar2 --local_epochs 1 --norm znorm --num_epochs 200 \
                        --model_learning_rate 0.0001
