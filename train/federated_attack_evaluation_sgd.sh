taskset 200 python3 federated_attribute_attack.py --dataset iemocap --leak_layer last \
                        --learning_rate 0.05 --optimizer adam --model_type fed_sgd \
                        --adv_dataset msp-improv_crema-d --pred emotion --device 0 \
                        --feature_type vq_apc --local_epochs 5 --norm znorm --num_epochs 200 \
                        --model_learning_rate 0.0005


taskset 200 python3 federated_attribute_attack.py --dataset crema-d --leak_layer last \
                        --learning_rate 0.05 --optimizer adam --model_type fed_sgd \
                        --adv_dataset iemocap_msp-improv --pred emotion --device 0 \
                        --feature_type vq_apc --local_epochs 5 --norm znorm --num_epochs 200 \
                        --model_learning_rate 0.0005


taskset 200 python3 federated_attribute_attack.py --dataset msp-improv --leak_layer last \
                        --learning_rate 0.05 --optimizer adam --model_type fed_sgd \
                        --adv_dataset iemocap_crema-d --pred emotion --device 0 \
                        --feature_type vq_apc --local_epochs 5 --norm znorm --num_epochs 200 \
                        --model_learning_rate 0.0005
