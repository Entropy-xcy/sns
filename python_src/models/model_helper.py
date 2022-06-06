from torch.optim.adam import Adam
import torch as th
import torch.nn as nn

from hyper_params import CHECKPOINTS_PATH
from timing_dataset import *
import time
import orjson 
import random
import numpy as np
import copy
import json

PLOT = True
batch_size = 1024

def rrse(pred, target):
    target_mean = th.mean(target)
    s_top = th.sum((pred - target) * (pred - target))
    s_bot = th.sum((target - target_mean) * (target - target_mean))
    return (s_top / s_bot)**0.5

def maep(pred, target):
    loss =  ((target - pred).abs() / target.abs()).mean() * 100.0
    return loss

def mep(pred, target):
    loss = ((pred - target) / target).mean() * 100.0
    return loss

def corre_coeff(pred, target):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    return np.corrcoef(pred, target)[0][1]

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def score_model(model_class, model_hparams, dataset, embedding_weights, iteration=10):
    scores = []
    for _ in range(iteration):
        model = model_class(model_hparams, dataset, embedding_weights)
        ff2ff_timing_train_t, ff2ff_timing_test_t = random_split(ff2ff_timing_dataset, [len(ff2ff_timing_dataset) - val_set_len, val_set_len])

        ff2ff_train_dataloader_t = DataLoader(ff2ff_timing_train_t, batch_size, shuffle=True, 
                                pin_memory=True)
        ff2ff_test_dataloader_t = DataLoader(ff2ff_timing_test_t, batch_size, shuffle=False, 
                                    pin_memory=True)

        X_train, X_train_wc, y_train = next(iter(ff2ff_train_dataloader_t))
        X_test, X_test_wc, y_test = next(iter(ff2ff_test_dataloader_t))
        # print(y_test[0])

        model = model.to('cuda:0')
        X_train = X_train.to('cuda:0')
        X_train_wc = X_train_wc.to('cuda:0')
        X_test = X_test.to('cuda:0')
        X_test_wc = X_test_wc.to('cuda:0')
        y_train = y_train.to('cuda:0')
        y_test = y_test.to('cuda:0')

        optimizer = Adam(model.parameters())

        num_epochs = model_hparams['epochs']
        start = time.time()
        for epoch in range(num_epochs):
            target = y_train

            out = model(X_train)
            out = th.ravel(out)

            optimizer.zero_grad()

            loss =  MAEPLoss(target, out)
            # print(loss.grad)

            loss.backward()
            optimizer.step()

            if (epoch+1) % 20 == 0:
                print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item():.6f}')
        end = time.time()
        print("Training Takes: ", end - start)

        model.eval()
        with th.no_grad():
            test_pred = model(X_test)
        
        model_loss = MAEPLoss(th.ravel(test_pred), y_test)

        print("Test Loss:", model_loss.cpu().item())
        scores.append(model_loss)
    
    return th.Tensor(scores)

def train_model(model_class, model_hparams, dataset, embedding_weights, iteration=1):
    model = model_class(model_hparams, dataset, embedding_weights)
    ff2ff_timing_train_t, ff2ff_timing_test_t = random_split(ff2ff_timing_dataset, [len(ff2ff_timing_dataset) - val_set_len, val_set_len])

    ff2ff_train_dataloader_t = DataLoader(ff2ff_timing_train_t, batch_size, shuffle=True, 
                                pin_memory=True)
    ff2ff_test_dataloader_t = DataLoader(ff2ff_timing_test_t, batch_size, shuffle=True, 
                                pin_memory=True)

    X_train, X_train_wc, y_train = next(iter(ff2ff_train_dataloader_t))
    X_test, X_test_wc, y_test = next(iter(ff2ff_test_dataloader_t))

    model = model.to('cuda:0')
    X_train = X_train.to('cuda:0')
    X_train_wc = X_train_wc.to('cuda:0')
    X_test = X_test.to('cuda:0')
    X_test_wc = X_test_wc.to('cuda:0')
    y_train = y_train.to('cuda:0')
    y_test = y_test.to('cuda:0')

    optimizer = Adam(model.parameters())

    num_epochs = model_hparams['epochs']
    start = time.time()
    for epoch in range(num_epochs):
        target = y_train

        print(X_train)
        out = model(X_train)
        out = th.ravel(out)

        optimizer.zero_grad()

        print(target.shape, out.shape)
        loss =  MAEPLoss(target, out)
        
        loss.backward()
        #print(loss.grad)
        optimizer.step()

        if (epoch+1) % 20 == 0:
            print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item():.6f}')
    end = time.time()
    print("Training Takes: ", end - start)

    model.eval()
    with th.no_grad():
        test_pred = model(X_test)
        
    model_loss = MAEPLoss(th.ravel(test_pred), y_test)

    print("Test Loss:", model_loss.cpu().item())
    
    return model

def predict_on_circuit(model, design, path_count=500):
    indices = random.choices(range(path_count), k=path_count)

    timings = predict_ff2ff_timing_batch(model, design['paths'], indices)
    
    max_indice = th.argmax(timings)
    # print(max_indice)
    max_path = design['paths'][max_indice]

    return th.max(timings).cpu().item(), max_path

def predict_ff2ff_timing(model, test_path):
    seq_enc = ff2ff_timing_dataset.encode_seq(test_path)
    seq_enc = seq_enc[None, :]
    seq_enc = seq_enc.to(device)
    # print(seq_enc)
    pred = model(seq_enc)

    return ff2ff_timing_dataset.denormalize_timing(pred), max_path

def predict_ff2ff_timing_batch(model, paths, indices):
    seq_batch = []
    for i in indices:
        seq_batch.append(paths[i])
    
    seq_batch_enc = th.zeros((len(indices), MAX_LEN), dtype=th.long)
    for i in range(len(indices)):
        seq_batch_enc[i] = ff2ff_timing_dataset.encode_seq(seq_batch[i])
    
    seq_batch_enc = seq_batch_enc.to(device)

    pred = model(seq_batch_enc)

    return ff2ff_timing_dataset.denormalize_timing(pred)

def save_test_result(model_base_name, hparams, 
                        real_timing, pred_timing, 
                        hparams_ignore_list=['batch_size', 'use_gpu']):
    # Determine Output filename
    json_output_filename = "eval/" + model_base_name
    for k in hparams.keys():
        if k not in hparams_ignore_list:
            json_output_filename += "_" + str(hparams[k])
    json_output_filename += ".json"

    # Determine output JSON
    json_output = {}
    json_output['model_name'] = model_base_name
    for k in hparams.keys():
        if k not in hparams_ignore_list:
            json_output[k] = hparams[k]
    
    json_output["target"] = real_timing.tolist()
    json_output["predict"] = pred_timing.tolist()
    
    with open(json_output_filename, 'w') as json_out_file:
        json_out_file.write(json.dumps(json_output))
        json_out_file.close()

