import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy


from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np
import torch.nn as nn
import sys, os, shutil, pickle, argparse, pdb

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'utils'))

from training_tools import EarlyStopping, seed_worker, result_summary
from attack_model import attack_model

EarlyStopping

# some general mapping for this script
gender_dict = {'F': 0, 'M': 1}
leak_layer_dict = {'full': ['w0', 'b0', 'w1', 'b1', 'w2', 'b2'],
                   'first': ['w0', 'b0'], 'second': ['w1', 'b1'], 'last': ['w2', 'b2']}
leak_layer_idx_dict = {'w0': 0, 'w1': 2, 'w2': 4, 'b0': 1, 'b1': 3, 'b2': 5}

class WeightDataGenerator():
    def __init__(self, dict_keys, data_dict = None):
        self.dict_keys = dict_keys
        self.data_dict = data_dict

    def __len__(self):
        return len(self.dict_keys)

    def __getitem__(self, idx):
        data_file_str = self.dict_keys[idx]
        gender = gender_dict[self.data_dict[data_file_str]['gender']]
        tmp_data = (self.data_dict[data_file_str][weight_name] - weight_norm_mean_dict[weight_name]) / (weight_norm_std_dict[weight_name] + 0.00001)
        weights = torch.from_numpy(np.ascontiguousarray(tmp_data))
        tmp_data = (self.data_dict[data_file_str][bias_name] - weight_norm_mean_dict[bias_name]) / (weight_norm_std_dict[bias_name] + 0.00001)
        bias = torch.from_numpy(np.ascontiguousarray(tmp_data))
        return weights, bias, gender

def run_one_epoch(model, data_loader, optimizer, scheduler, loss_func, epoch, mode='train'):
    
    model.train() if mode == 'train' else model.eval()
    step_outputs = []
    
    for batch_idx, data_batch in enumerate(data_loader):
        weights, bias, y = data_batch
        weights, bias, y = weights.to(device), bias.to(device), y.to(device)
        logits = model(weights.float().unsqueeze(dim=1), bias.float())
        loss = loss_func(logits, y)

        predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
        pred_list = [predictions[pred_idx] for pred_idx in range(len(predictions))]
        truth_list = [y.detach().cpu().numpy()[pred_idx] for pred_idx in range(len(predictions))]
        step_outputs.append({'loss': loss.item(), 'pred': pred_list, 'truth': truth_list})

        # step the loss back
        if mode == 'train':
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        del data_batch, logits, loss
        torch.cuda.empty_cache()
    result_dict = result_summary(step_outputs, mode, epoch)

    # if validate mode, step the loss
    if mode == 'validate':
        mean_loss = np.mean(result_dict['loss'])
        scheduler.step(mean_loss)
    return result_dict

    
if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--adv_dataset', default='iemocap')
    parser.add_argument('--feature_type', default='apc')
    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--model_learning_rate', default=0.0005)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--num_epochs', default=200)
    parser.add_argument('--local_epochs', default=5)
    parser.add_argument('--norm', default='znorm')
    parser.add_argument('--device', default='0')
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--model_type', default='fed_sgd')
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--leak_layer', default='full')
    parser.add_argument('--dropout', default=0.2)
    parser.add_argument('--privacy_budget', default=None)
    parser.add_argument('--save_dir', default='/media/data/projects/speech-privacy')
    args = parser.parse_args()

    seed_worker(8)
    device = torch.device("cuda:"+str(args.device)) if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    model_setting_str = 'local_epoch_'+str(args.local_epochs) if args.model_type == 'fed_avg' else 'local_epoch_1'
    model_setting_str += '_dropout_' + str(args.dropout).replace('.', '')
    model_setting_str += '_lr_' + str(args.learning_rate)[2:]
    if args.privacy_budget is not None: model_setting_str += '_udp_' + str(args.privacy_budget)

    torch.cuda.empty_cache() 
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # 1. normalization tmp computations
    weight_norm_mean_dict, weight_norm_std_dict = {}, {}
    weight_sum, weight_sum_square = {}, {}
    for key in ['w0', 'w1', 'w2', 'b0', 'b1', 'b2']:
        weight_norm_mean_dict[key], weight_norm_std_dict[key] = 0, 0
        weight_sum[key], weight_sum_square[key] = 0, 0
    
    # the updates layer name and their idx in gradient file
    weight_name, bias_name = leak_layer_dict[args.leak_layer][0], leak_layer_dict[args.leak_layer][1]
    weight_idx, bias_idx = leak_layer_idx_dict[weight_name], leak_layer_idx_dict[bias_name]

    # 1.1 read all data and compute the tmp variables
    shadow_training_sample_size, shadow_data_dict = 0, {}
    print('reading file %s' % str(Path(args.save_dir).joinpath('tmp_model_params', args.model_type, args.pred, args.feature_type, args.adv_dataset, model_setting_str)))
    for shadow_idx in range(5):
        for epoch in range(int(args.num_epochs)):
            adv_federated_model_result_path = Path(args.save_dir).joinpath('tmp_model_params', args.model_type, args.pred, args.feature_type, args.adv_dataset, model_setting_str, 'fold'+str(int(shadow_idx+1)))
            file_str = str(adv_federated_model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl'))
            # if shadow_idx == 0 and epoch < 10:
            if epoch % 20 == 0: 
                print('reading shadow model %d, epoch %d' % (shadow_idx, epoch))
            with open(file_str, 'rb') as f:
                adv_gradient_dict = pickle.load(f)
            for speaker_id in adv_gradient_dict:
                data_key = str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id
                gradients = adv_gradient_dict[speaker_id]['gradient']
                shadow_training_sample_size += 1
                
                # calculate running stats for computing std and mean
                shadow_data_dict[data_key] = {}
                shadow_data_dict[data_key]['gender'] = adv_gradient_dict[speaker_id]['gender']
                shadow_data_dict[data_key][weight_name] = gradients[weight_idx]
                shadow_data_dict[data_key][bias_name] = gradients[bias_idx]
                for layer_name in leak_layer_dict[args.leak_layer]:
                    weight_sum[layer_name] += gradients[leak_layer_idx_dict[layer_name]]
                    weight_sum_square[layer_name] += gradients[leak_layer_idx_dict[layer_name]]**2
            
    # 1.2 calculate std and mean
    for key in leak_layer_dict[args.leak_layer]:
        weight_norm_mean_dict[key] = weight_sum[key] / shadow_training_sample_size
        tmp_data = weight_sum_square[key] / shadow_training_sample_size - (weight_sum[key] / shadow_training_sample_size)**2
        weight_norm_std_dict[key] = np.sqrt(tmp_data)
    
    # 2. train model to infer gender
    # 2.1 define model
    train_key_list, validate_key_list = train_test_split(list(shadow_data_dict.keys()), test_size=0.2, random_state=0)
    model = attack_model(args.leak_layer, args.feature_type)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(args.model_learning_rate), weight_decay=1e-04, betas=(0.9, 0.98), eps=1e-9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.2, verbose=True, min_lr=1e-6)

    # 2.2 define data loader
    dataset_train = WeightDataGenerator(train_key_list, shadow_data_dict)
    dataset_valid = WeightDataGenerator(validate_key_list, shadow_data_dict)
    train_loader = DataLoader(dataset_train, batch_size=20, num_workers=0, shuffle=True)
    validation_loader =DataLoader(dataset_valid, batch_size=20, num_workers=0, shuffle=False)
    
    # 2.3 initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10, verbose=True)
    loss = nn.NLLLoss().to(device)
    
    # 2.4 log saving path
    attack_model_result_path = Path(os.path.realpath(__file__)).parents[1].joinpath('results', 'attack', args.leak_layer, args.model_type, args.feature_type, model_setting_str)
    log_path = Path.joinpath(attack_model_result_path, 'log_private_' + str(args.dataset))
    if log_path.exists(): shutil.rmtree(log_path)
    Path.mkdir(log_path, parents=True, exist_ok=True)
    
    # 2.5 training attack model
    result_dict, best_val_dict = {}, {}
    for epoch in range(40):
        # perform the training, validate, and test
        train_result = run_one_epoch(model, train_loader, optimizer, scheduler, loss, epoch, mode='train')
        validate_result = run_one_epoch(model, validation_loader, optimizer, scheduler, loss, epoch, mode='validate')
        
        # save the results for later
        result_dict[epoch] = {}
        result_dict[epoch]['train'], result_dict[epoch]['validate'] = train_result, validate_result
        
        if len(best_val_dict) == 0: best_val_dict, best_epoch = validate_result, epoch
        if validate_result['uar'] > best_val_dict['uar'] and epoch > 30:
            best_val_dict, best_epoch = validate_result, epoch
            best_model = deepcopy(model.state_dict())
            torch.save(deepcopy(model.state_dict()), str(attack_model_result_path.joinpath('private_'+args.dataset+'.pt')))
            
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        if epoch > 30: early_stopping(validate_result['loss'], model)

        # print(final_acc, best_val_acc, best_epoch)
        print('best epoch %d, best final acc %.2f, best final uar %.2f' % (best_epoch, best_val_dict['acc']*100, best_val_dict['uar']*100))
        print(best_val_dict['conf'])
        
        if early_stopping.early_stop and epoch > 30:
            print("Early stopping")
            break

    # 3. we evaluate the attacker performance on service provider training
    save_result_df = pd.DataFrame()
    eval_model = attack_model(args.leak_layer, args.feature_type)
    eval_model.load_state_dict(torch.load(str(attack_model_result_path.joinpath('private_'+args.dataset+'.pt'))))
    eval_model = eval_model.to(device)
    # 3.1 we perform 5 fold evaluation, since we also train the private data 5 times
    for fold_idx in range(5):
        test_data_dict = {}
        for epoch in range(int(args.num_epochs)):
            row_df = pd.DataFrame(index=['fold'+str(int(fold_idx+1))])
            
            # Model related
            federated_model_result_path = Path(args.save_dir).joinpath('tmp_model_params', args.model_type, args.pred, args.feature_type, args.dataset, model_setting_str, 'fold'+str(int(fold_idx+1)))
            weight_file_str = str(federated_model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl'))

            with open(weight_file_str, 'rb') as f:
                test_gradient_dict = pickle.load(f)
            for speaker_id in test_gradient_dict:
                data_key = str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id
                gradients = test_gradient_dict[speaker_id]['gradient']
                test_data_dict[data_key] = {}
                test_data_dict[data_key]['gender'] = test_gradient_dict[speaker_id]['gender']
                test_data_dict[data_key][weight_name] = gradients[weight_idx]
                test_data_dict[data_key][bias_name] = gradients[bias_idx]

        dataset_test = WeightDataGenerator(list(test_data_dict.keys()), test_data_dict)
        test_loader = DataLoader(dataset_test, batch_size=20, num_workers=0, shuffle=False)
        test_result = run_one_epoch(eval_model, test_loader, optimizer, scheduler, loss, best_epoch, mode='test')
    
        row_df['acc'], row_df['uar'] = test_result['acc'], test_result['uar']
        save_result_df = pd.concat([save_result_df, row_df])
        del dataset_test, test_loader
        
    row_df = pd.DataFrame(index=['average'])
    row_df['acc'], row_df['uar'] = np.mean(save_result_df['acc']), np.mean(save_result_df['uar'])
    save_result_df = pd.concat([save_result_df, row_df])
    save_result_df.to_csv(str(attack_model_result_path.joinpath('private_' + str(args.dataset) + '_result.csv')))

