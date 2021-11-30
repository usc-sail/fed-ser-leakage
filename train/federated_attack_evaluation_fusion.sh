taskset 200 python3 federated_attribute_attack_fusion.py --dataset iemocap \
                        --learning_rate 0.0001 --model_type fed_avg \
                        --adv_dataset msp-improv_crema-d --pred emotion --device 0 \
                        --feature_type vq_apc --local_epochs 5 --norm znorm --num_epochs 200


taskset 200 python3 federated_attribute_attack_fusion.py --dataset crema-d \
                        --learning_rate 0.0001 --model_type fed_avg \
                        --adv_dataset iemocap_msp-improv --pred emotion --device 0 \
                        --feature_type vq_apc --local_epochs 5 --norm znorm --num_epochs 200


taskset 200 python3 federated_attribute_attack_fusion.py --dataset msp-improv \
                        --learning_rate 0.0001 --model_type fed_avg \
                        --adv_dataset iemocap_crema-d --pred emotion --device 0 \
                        --feature_type vq_apc --local_epochs 5 --norm znorm --num_epochs 200
