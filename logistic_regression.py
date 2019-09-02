import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from random import shuffle
from sklearn import linear_model
from sklearn.model_selection import KFold,GridSearchCV
class logistic_regression:
    TrainX_vector=0
    TrainY_vector=0
    ValX_vector=0
    ValY_vector=0
    Weight=0
    Train_rmse=0
    Val_rmse=0
    split_index=0
    train_array_x=[]
    train_array_y=[]
    val_array_x=[]
    val_array_y=[]
    def __init__(self):
        self.TrainX_vector=0
        self.TrainY_vector=0
        self.ValX_vector=0
        self.ValY_vector=0
        self.Weight=0
        self.Train_rmse=0
        self.Val_rmse=0
        self.split_index=0
        self.train_array_x=[]
        self.train_array_y=[]
        self.val_array_x=[]
        self.val_array_y=[]
    
    def find_vector_no_folds(self,dataset):
        dataset=dataset.sample(frac=1)
        train_len=math.floor(7*len(dataset)/10)
        input_val=dataset[train_len:][dataset.columns[1:len(dataset.keys())-1]]
        output_val=dataset[train_len:][dataset.columns[len(dataset.keys())-1:]]
        
        input_train=dataset[0:train_len][dataset.columns[1:len(dataset.keys())-1]]
        output_train=dataset[0:train_len][dataset.columns[len(dataset.keys())-1:]]
        
        trainX=input_train.reset_index()
        trainY=output_train.reset_index()
        
        ValX=input_val.reset_index()
        ValY=output_val.reset_index()
        
        self.TrainX_vector=self.convert_to_numpy(trainX,1)
        self.TrainY_vector=self.convert_to_numpy(trainY)
        self.ValX_vector=self.convert_to_numpy(ValX,1)
        self.ValY_vector=self.convert_to_numpy(ValY)
        
        self.train_array_x.append(self.TrainX_vector)
        self.val_array_x.append(self.ValX_vector)
        self.train_array_y.append(self.TrainY_vector)
        self.val_array_y.append(self.ValY_vector)