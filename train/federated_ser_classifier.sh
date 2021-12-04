python3 federated_classifier.py --dataset iemocap --local_epochs 5 --learning_rate 0.05 --model_type fed_sgd \
                        --feature_type apc --pred emotion --norm znorm --optimizer adam --dropout 0.2 --num_epochs 200

