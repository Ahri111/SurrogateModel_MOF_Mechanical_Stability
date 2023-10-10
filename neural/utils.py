import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.stats import boxcox
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from glob import glob
from sparse import load_npz

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# RACs + geometric descriptors

def data_loader(base_path, file_name):
    directory = str(os.path.join(base_path, file_name))
    data = pd.read_csv(directory,index_col=0)
    data = data.dropna(axis=0)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    return data

def remove_constant_value_features(df):
    return [e for e in df.columns if df[e].nunique() == 1]

def get_descriptors(data,descriptors_columns):
    descriptors = data[descriptors_columns].copy()
    drop_col = remove_constant_value_features(descriptors)
    new_df_columns = [e for e in descriptors.columns if e not in drop_col]
    descriptors = descriptors[new_df_columns]
    
    return descriptors

def one_filter(data,one_filter_columns):
    one_property = data[one_filter_columns].copy()
    
    return one_property

def data_scaler(data):
    scaler = preprocessing.StandardScaler()
    
    return scaler.fit_transform(data)

def box_cox_transform(data):
    transformed_data, best_lambda = boxcox(data)

    return transformed_data, best_lambda

# For SOAP

class soap(Dataset):
    
    def __init__(self,file_directory,soap_directory,feature,target):
        self.file = pd.read_csv(file_directory)
        self.filename = self.file["filename"]
        self.soap_directory =  soap_directory
        self.columns = feature
        self.feature = self.get_descriptors(data = self.file[feature],
                                            descriptors_columns = self.columns)
        self.property = self.file[target]
        
    def remove_constant_value_features(self, df):
        return [e for e in df.columns if df[e].nunique() == 1]
    
    def get_descriptors(self,data,descriptors_columns):
        descriptors = data[descriptors_columns].copy()
        drop_col = remove_constant_value_features(descriptors)
        new_df_columns = [e for e in descriptors.columns if e not in drop_col]
        descriptors = descriptors[new_df_columns]
    
        return descriptors
        
    def __getitem__(self,idx):
        
        feature = np.array(self.feature)[idx]
        soap_filename = self.soap_directory + "/" + self.filename[idx][:-3] + "npz"
        soap_feature = load_npz(soap_filename).todense()
        target = np.array(self.property)[idx]
        batch_feature = torch.tensor(feature,dtype=torch.float)
        batch_soap = torch.tensor(soap_feature,dtype = torch.float)
        batch_x =  torch.cat([batch_feature,batch_soap])
        batch_target = torch.tensor(target,dtype = torch.float)

        return batch_x, batch_target
    
    def __len__(self):
        
        return  len(self.property)
    
class Topology(Dataset):
    
    def __init__(self,file_directory,soap_directory,feature,target):
        self.file = pd.read_csv(file_directory)
        self.filename = self.file["filename"]
        self.soap_directory =  soap_directory
        self.columns = feature
        self.feature = self.get_descriptors(data = self.file[feature],
                                            descriptors_columns = self.columns)
        self.property = self.file[target]
        
    def remove_constant_value_features(self, df):
        return [e for e in df.columns if df[e].nunique() == 1]
    
    def get_descriptors(self,data,descriptors_columns):
        descriptors = data[descriptors_columns].copy()
        drop_col = remove_constant_value_features(descriptors)
        new_df_columns = [e for e in descriptors.columns if e not in drop_col]
        descriptors = descriptors[new_df_columns]
    
        return descriptors
        
    def __getitem__(self,idx):
        
        feature = np.array(self.feature)[idx]
        soap_filename = self.soap_directory + "/" + self.filename[idx][:-3] + "npz"
        soap_feature = load_npz(soap_filename).todense()
        target = np.array(self.property)[idx]
        batch_feature = torch.tensor(feature,dtype=torch.float)
        batch_soap = torch.tensor(soap_feature,dtype = torch.float)
        batch_x =  torch.cat([batch_feature,batch_soap])
        batch_target = torch.tensor(target,dtype = torch.float)

        return batch_x, batch_target
    
    def __len__(self):
        
        return  len(self.property)
        
    
    
# 
    

    
    
        