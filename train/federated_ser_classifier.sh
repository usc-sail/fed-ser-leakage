python3 federated_ser_classifier.py --dataset msp-improv_crema-d --norm znorm \
                        --feature_type emobase --dropout 0.2 --num_epochs 200 --local_epochs 1 \
                        --optimizer adam --model_type fed_avg --learning_rate 0.0005 \
                        --save_dir /media/data/projects/speech-privacy