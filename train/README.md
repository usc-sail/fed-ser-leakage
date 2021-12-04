# Training scripts
The scripts under this folder trains the SER model in FL and attack model. 

### SER training in FL
The bash file federated_ser_classifier.sh provides an example of running the preprocess python file. e.g.:

```sh
python3 federated_ser_classifier.py --dataset iemocap --local_epochs 5 --learning_rate 0.05 --model_type fed_sgd \
                        --feature_type apc --pred emotion --norm znorm --optimizer adam --dropout 0.2 --num_epochs 200

```
- The arg `dataset` specifies the data set. The support data sets are IEMOCAP (iemocap), MSP-Improv (msp-improv), and CREMA-D (crema-d). 

- The arg `feature_type` is the feature reprentation type. Please refer to README under feature extraction for more details.

- The arg `pred` is prediction label. Currently support SER only, the arousal and valence predictions are ongoing work.

- The arg `norm` specifies the normalization method including z-normalization and min-max normalization. The normalization is implemented within a speaker.

- The arg `model_type` specifies the FL algorithm: fed_sgd and fed_avg.

- The arg `dropout` specifies the dropout value in the first dense layer.

- The arg `local_epochs` specifies the local epoch, ignored in fed_sgd.

- The arg `num_epochs` specifies the global epoch in FL.

- The arg `learning_rate` specifies the learning rate in FL.

### This the code that average the gradients in fed_sgd
```python
# 2.1 average global gradients
global_gradients = average_gradients(local_updates, local_num_sampels)
# 2.2 update global weights
global_weights = copy.deepcopy(global_model.state_dict())
global_weights_keys = list(global_weights.keys())

for key_idx in range(len(global_weights_keys)):
    key = global_weights_keys[key_idx]
    global_weights[key] -= float(args.learning_rate)*global_gradients[key_idx].to(device)
```
### This the code that average the weights in fed_avg

```python 
global_weights = average_weights(local_updates, local_num_sampels)
```

This is how we compute the psuedo gradient
```python
# 'fake' gradients saving code
# iterate all layers in the classifier model
original_model = copy.deepcopy(global_model).state_dict()

# calculate how many updates per local epoch 
local_update_per_epoch = int(train_sample_size / int(args.batch_size)) + 1
    
for key in original_model:
    original_params = original_model[key].detach().clone().cpu().numpy()
    update_params = local_update[key].detach().clone().cpu().numpy()
    
    # calculate 'fake' gradients
    tmp_gradients = (original_params - update_params)/(float(args.learning_rate)*local_update_per_epoch*int(args.local_epochs))
    gradients.append(tmp_gradients)
```
