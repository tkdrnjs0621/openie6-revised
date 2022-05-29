import os
import numpy as np
import torch
import torch.nn as nn
import random  #
import argparse  #
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
from transformers import AutoTokenizer
from collections import OrderedDict

import data
import metric
from model import IGL_OIE 

def train(model, optimizer,max_epoch, print_interval,train_dl, valid_dl, load_path=None, save_path='./model.pt'):
    loaded_epoch = 0
    loaded_best_f1 = -1
    _metric = metric.Carb(5,True)
    
    
    if load_path!= None:
        state = torch.load(load_path)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        loaded_epoch = state["epoch"]
        loaded_best_f1 = state["best_f1"]
        
    best_f1 = 0 if loaded_best_f1 == -1 else loaded_best_f1

    for epoch in np.array(list(range(max_epoch - loaded_epoch))) + loaded_epoch:
        model.train()
        totalloss=0
        print("Starting Epoch",str(epoch+1))
        for step, sample in enumerate(train_dl):
            # print(sample)
            outputs = model(sample)
            
            optimizer.zero_grad()
            loss = outputs["loss"]
            totalloss = totalloss+loss.item()
            loss.backward()
            optimizer.step()
            
            if (step + 1) % print_interval == 0:
                print('epoch:', epoch + 1, 'step:', step + 1,'/',len(train_dl), 'avg loss:', totalloss/step)
        print("Epoch "+str(epoch+1)+" ended! Evaluating model...")
        with torch.no_grad():
            model.eval()
            outputs=[]
            for step, sample in enumerate(valid_dl):
                output_dict = model(sample)
                outputD = {"predictions": output_dict['predictions'], "scores": output_dict['scores'],
                   "ground_truth": sample["labels"], "meta_data": sample["meta_data"]}
                output = OrderedDict(outputD)
                outputs.append(output)
                
            for output in outputs:
                # if type(output['meta_data'][0]) != type(""):
                #     output['meta_data'] = [self._meta_data_vocab.itos[m] for m in output['meta_data']]
                
                _metric(output['predictions'], output['meta_data'], output['scores'])
            metrics = _metric.get_metric(reset=True, mode='dev')
            result = {"eval_f1": metrics['carb_f1'], "eval_auc": metrics['carb_auc'], "eval_lastf1": metrics['carb_lastf1']}

            print('\nEvaluation Results: '+str(result))
                
            if metrics['carb_f1'] > best_f1:
                print("New best valid f1, saving model")
                best_f1=metrics['carb_f1']
                # Save your states
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_f1": best_f1
                    # ...
                }
                torch.save(state, save_path)
            print('Valid epoch: %d, Valid f1: %.6f, Best valid accuracy: %.6f' % (epoch + 1, metrics['carb_f1'], best_f1))


def main():
    torch.multiprocessing.set_start_method('spawn')
    if(False):
        train_dataset = data.process_data("data/openie4_labels")
        val_dataset = data.process_data("data/dev.txt")
        test_dataset = data.process_data("data/test.txt")
        with open('train_dataset.pkl','wb') as f:
            pickle.dump(train_dataset, f)
        with open('val_dataset.pkl','wb') as f:
            pickle.dump(val_dataset, f)
        with open('test_dataset.pkl','wb') as f:
            pickle.dump(test_dataset, f)
    else:
        with open('train_dataset.pkl','rb') as f:
            train_dataset = pickle.load(f)
        with open('val_dataset.pkl','rb') as f:
            val_dataset = pickle.load(f)
        with open('test_dataset.pkl','rb') as f:
            test_dataset = pickle.load(f)
            
            
    train_dataloader = DataLoader(train_dataset[0][:len(train_dataset[0])], batch_size=12, collate_fn=data.pad_data, num_workers=4)
    val_dataloader = DataLoader(val_dataset[0], batch_size=12, collate_fn=data.pad_data, num_workers=4)
    test_dataloader = DataLoader(test_dataset[0], batch_size=12, collate_fn=data.pad_data, num_workers=1)
    # train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=1)
    # val_dataloader = DataLoader(val_dataset, batch_size=24, num_workers=1)
    # test_dataloader = DataLoader(test_dataset, batch_size=24, num_workers=1)
    
    Model = IGL_OIE()
    optimizer = optim.AdamW(Model.parameters(),lr=2e-5)
    
    train(Model, optimizer, 100, 20,train_dataloader, val_dataloader)
    


if __name__ == "__main__":
    main()