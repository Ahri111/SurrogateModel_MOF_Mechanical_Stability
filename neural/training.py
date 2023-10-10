import pandas as pd
import os
import random
from sklearn import preprocessing
from scipy.stats import boxcox
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from glob import glob
from sparse import load_npz

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset 
import torch.nn.functional as F

import numpy as np 
import matplotlib.pyplot as plt 
import time 

from utils import *
from model import NeuralNet

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)

def run_training(params,fold,X,y,input_size):
    
    kfold = KFold(n_splits = fold, shuffle = True, random_state = 42)
    model = NeuralNet(input_size,hidden_size=params["hidden_size"],
                     fc_count=params["num_layers"])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=params["learning_rate"])
    criterion = nn.L1Loss()
    early_stopping_iter = 10
    early_stopping_counter = 0
    all_score = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(y)):
        best_score = -np.inf
        X_train, X_test = X[train_idx], X[val_idx]
        y_train, y_test = y[train_idx], y[val_idx]
        
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test,y_test)
        train_loader = DataLoader(train_dataset,batch_size=params["batch_size"],shuffle=True)
        val_loader = DataLoader(test_dataset,batch_size=params["batch_size"])
        
        num_epochs = 10000
        
        scores_epoch = []
        
        for epoch in range(num_epochs):
            model.train()
            final_loss = 0
            for batch_idx, samples in enumerate(train_loader):
                optimizer.zero_grad()
                data_x,data_y = samples
                inputs = data_x.to(device)
                targets = data_y.to(device)
                outputs = model(inputs)
                loss = criterion(targets,outputs)
                loss.backward()
                optimizer.step()
        
            model.eval()
            final_loss = 0
            score = 0
            
            for batch_idx,samples in enumerate(val_loader):
                data_x,data_y = samples
                inputs = data_x.to(device)
                targets = data_y.to(device)
                outputs = model(inputs)
                loss = criterion(targets,outputs)
                score += r2_score(targets.data.cpu().detach().numpy(),outputs.data.cpu().detach().numpy())
                final_loss += loss.item()
            
            final_score = score/len(val_loader)
            valid_loss = final_loss/len(val_loader)
            
            print(f"fold: {fold}, epoch: {epoch}, \
                    loss : {valid_loss}, r2 : {final_score}")
            
            if final_score > best_score:
                best_score = final_score
                state = {
                    "state_dict" : model.state_dict(),
                    "optimizer" : model.state_dict()
                }
                torch.save(state,"ckpt/best.pt")
            else:
                early_stopping_counter +=1
                
            if early_stopping_counter > early_stopping_iter:
                break
            
        all_score.append(final_score)
        average_score = np.mean(all_score)
        
    return average_score

def run_evaluate(params,fold,X,y,input_size,best_mdoel):
    
    kfold = KFold(n_splits = fold, shuffle = True, random_state = 42)
    model = NeuralNet(input_size,hidden_size=params["hidden_size"],
                     fc_count=params["num_layers"])
    # 내일 SOAP 완성되면 이 부분까지 적어서 해보자 
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=params["learning_rate"])
    criterion = nn.L1Loss()
    all_score = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(y)):
        best_score = -np.inf
        X_train, X_test = X[train_idx], X[val_idx]
        y_train, y_test = y[train_idx], y[val_idx]
        
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test,y_test)
        train_loader = DataLoader(train_dataset,batch_size=params["batch_size"],shuffle=True)
        val_loader = DataLoader(test_dataset,batch_size=params["batch_size"])
        
        num_epochs = 10000
        
        scores_epoch = []
        
        for epoch in range(num_epochs):
            model.train()
            final_loss = 0
            for batch_idx, samples in enumerate(train_loader):
                optimizer.zero_grad()
                data_x,data_y = samples
                inputs = data_x.to(device)
                targets = data_y.to(device)
                outputs = model(inputs)
                loss = criterion(targets,outputs)
                loss.backward()
                optimizer.step()
        
            model.eval()
            final_loss = 0
            score = 0
            
            for batch_idx,samples in enumerate(val_loader):
                data_x,data_y = samples
                inputs = data_x.to(device)
                targets = data_y.to(device)
                outputs = model(inputs)
                loss = criterion(targets,outputs)
                score += r2_score(targets.data.cpu().detach().numpy(),outputs.data.cpu().detach().numpy())
                final_loss += loss.item()
            
            final_score = score/len(val_loader)
            valid_loss = final_loss/len(val_loader)
            
            print(f"fold: {fold}, epoch: {epoch}, \
                    loss : {valid_loss}, r2 : {final_score}")
            
            if final_score > best_score:
                best_score = final_score
                state = {
                    "state_dict" : model.state_dict(),
                    "optimizer" : model.state_dict()
                }
                torch.save(state,"ckpt/best_test.pt")
            else:
                early_stopping_counter +=1
                
            if early_stopping_counter > early_stopping_iter:
                break
            
        all_score.append(final_score)
        average_score = np.mean(all_score)
        
    return average_score 


