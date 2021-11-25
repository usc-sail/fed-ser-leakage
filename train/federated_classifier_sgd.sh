python3 federated_classifier.py --dataset iemocap --local_epochs 5 --learning_rate 0.05 --model_type fed_sgd \
                        --feature_type distilhubert --pred emotion --norm znorm --optimizer adam --dropout 0.2 --num_epochs 200

python3 federated_classifier.py --dataset msp-improv_crema-d --local_epochs 5 --learning_rate 0.05 --model_type fed_sgd \
                        --feature_type distilhubert --pred emotion --norm znorm --optimizer adam --dropout 0.2 --num_epochs 200

python3 federated_classifier.py --dataset crema-d --local_epochs 5 --learning_rate 0.05 --model_type fed_sgd \
                        --feature_type distilhubert --pred emotion --norm znorm --optimizer adam --dropout 0.2 --num_epochs 200

python3 federated_classifier.py --dataset iemocap_msp-improv --local_epochs 5 --learning_rate 0.05 --model_type fed_sgd \
                        --feature_type distilhubert --pred emotion --norm znorm --optimizer adam --dropout 0.2 --num_epochs 200

python3 federated_classifier.py --dataset msp-improv --local_epochs 5 --learning_rate 0.05 --model_type fed_sgd \
                        --feature_type distilhubert --pred emotion --norm znorm --optimizer adam --dropout 0.2 --num_epochs 200

python3 federated_classifier.py --dataset iemocap_crema-d --local_epochs 5 --learning_rate 0.05 --model_type fed_sgd \
                        --feature_type distilhubert --pred emotion --norm znorm --optimizer adam --dropout 0.2 --num_epochs 200


