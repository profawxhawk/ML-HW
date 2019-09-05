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
    TestX_vector=0
    TestY_vector=0
    train_array_x=[]
    train_array_y=[]
    val_array_x=[]
    val_array_y=[]
    def __init__(self):
        self.TrainX_vector=0
        self.TrainY_vector=0
        self.ValX_vector=0
        self.ValY_vector=0
        self.TestX_vector=0
        self.TestY_vector=0
        self.Weight=0
        self.train_array_x=[]
        self.train_array_y=[]
        self.val_array_x=[]
        self.val_array_y=[]
    
    def find_vector_no_folds(self,dataset,testset):
        dataset=dataset.sample(frac=1)
        train_len=math.floor(7*len(dataset)/10)
        input_val=dataset[train_len:][dataset.columns[0:len(dataset.keys())-1]]
        output_val=dataset[train_len:][dataset.columns[len(dataset.keys())-1:]]
        
        input_train=dataset[0:train_len][dataset.columns[0:len(dataset.keys())-1]]
        output_train=dataset[0:train_len][dataset.columns[len(dataset.keys())-1:]]
        
        input_test=testset[0:][testset.columns[0:len(testset.keys())-1]]
        output_test=testset[0:][testset.columns[len(testset.keys())-1:]]
        
        trainX=input_train.reset_index()
        trainY=output_train.reset_index()
        
        ValX=input_val.reset_index()
        ValY=output_val.reset_index()
        
        TestX=input_test.reset_index()
        TestY=output_test.reset_index()
        
        self.TrainX_vector=self.convert_to_numpy(trainX,1)
        self.TrainY_vector=self.convert_to_numpy(trainY)
        self.ValX_vector=self.convert_to_numpy(ValX,1)
        self.ValY_vector=self.convert_to_numpy(ValY)
        self.TestX_vector=self.convert_to_numpy(TestX,1)
        self.TestY_vector=self.convert_to_numpy(TestY)
        
        self.normalize_vector(self.TrainX_vector)
        self.normalize_vector(self.ValX_vector)
        self.normalize_vector(self.TestX_vector)
        
        
        self.train_array_x.append(self.TrainX_vector)
        self.val_array_x.append(self.ValX_vector)
        self.train_array_y.append(self.TrainY_vector)
        self.val_array_y.append(self.ValY_vector)
        
    def label_change(self,i):
        self.TrainY_vector=np.where(self.TrainY_vector == i, 1, 0)
        self.TestY_vector=np.where(self.TestY_vector == i, 1, 0)
    def normalize_vector(self,vector):
        for i in range((vector.shape[1])-1):
            # vector[:,i]=(vector[:,i]-vector[:,i].mean())/np.std(vector[:,i])
            vector[:,i]=(vector[:,i]-vector[:,i].min())/(vector[:,i].max()-vector[:,i].min())
    def convert_to_numpy(self,data,flag=0):
        data_vector=[]
        for i in range(len(data)):
            temp=[]
            for j in (data.keys()):
                if j!='index':
                    temp.append(data[j][i])
            if flag==1:
                temp.append(1)
            data_vector.append(temp)
        data_vector=np.asarray(data_vector)
        return data_vector

    def sigmoid(self,x):
        sig=1 / (1 + np.exp(-x))
        sig[sig == 1.0] = 0.9999
        sig[sig == 0.0] = 0.0001
        return sig
    
    def loss_differential(self,test,fold_count):
        if test=="True":
            temp=np.matmul(self.val_array_x[fold_count-1],self.Weight)
            temp=self.sigmoid(temp)-self.val_array_y[fold_count-1]
        else:
            temp=np.matmul(self.train_array_x[fold_count-1],self.Weight)
            temp=self.sigmoid(temp)-self.train_array_y[fold_count-1]
        return np.transpose((temp))
    
    def cost_func(self,y,h,model="none",reg_param=0):
        if model=="l1":
            return ((-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()+reg_param*(sum(np.absolute(self.Weight))))
        elif model=="l2":
            return ((-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()+reg_param*(sum(np.square(self.Weight))))
        else:
            return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def prob_convert(self,vector):
        return np.where(vector >= 0.5, 1, 0)
    
    def accuracy(self,type):
        count=0
        length=1
        if type=="train":
            length=len(self.TrainY_vector)
            predict=self.sigmoid(self.TrainX_vector @ self.Weight)
            predict=self.prob_convert(predict)
            for i in range(len(predict)):
                if predict[i]==self.TrainY_vector[i]:
                    count+=1

        if type=="val":
            length=len(self.ValY_vector)
            predict=self.sigmoid(self.ValX_vector @ self.Weight)
            predict=self.prob_convert(predict)
            for i in range(len(predict)):
                if predict[i]==self.ValY_vector[i]:
                    count+=1
            
        if type=="test":
            length=len(self.TestY_vector)
            predict=self.sigmoid(self.TestX_vector @ self.Weight)
            predict=self.prob_convert(predict)
            for i in range(len(predict)):
                if predict[i]==self.TestY_vector[i]:
                    count+=1       
        return (count/length)
            
    def gradient_descent_normal(self,alpha,test="False"):
        temp_cost=0 
        if test=="False":
            h = self.sigmoid(self.TrainX_vector @ self.Weight)
            gradient = (self.TrainX_vector.T @ (h - self.TrainY_vector)) / self.TrainY_vector.size
            self.Weight -= alpha * gradient
            temp_cost=self.cost_func(self.TrainY_vector,h)
        else:
            h = self.sigmoid(self.ValX_vector @ self.Weight)
            gradient = (self.ValX_vector.T @ (h - self.ValY_vector)) / self.ValY_vector.size
            self.Weight -= alpha * gradient
            temp_cost=self.cost_func(self.ValY_vector,h)
            
        return temp_cost
    
    def gradient_descent_ridge(self,alpha,reg_param,test="False"):
        temp_cost=0 
        if test=="False":
            h = self.sigmoid(self.TrainX_vector @ self.Weight)
            gradient = (((self.TrainX_vector.T @ (h - self.TrainY_vector)) / self.TrainY_vector.size)+2*reg_param*self.Weight)
            self.Weight -= alpha * gradient
            temp_cost=self.cost_func(self.TrainY_vector,h,"l2",reg_param)
        else:
            h = self.sigmoid(self.ValX_vector @ self.Weight)
            gradient = (((self.ValX_vector.T @ (h - self.ValY_vector)) / self.ValY_vector.size)+2*reg_param*self.Weight)
            self.Weight -= alpha * gradient
            temp_cost=self.cost_func(self.ValY_vector,h,"l2",reg_param)
            
        return temp_cost,self.accuracy("train")
    
    def gradient_descent_lasso(self,alpha,reg_param,test="False"):
        temp_cost=0 
        if test=="False":
            h = self.sigmoid(self.TrainX_vector @ self.Weight)
            temp=((self.TrainX_vector.T @ (h - self.TrainY_vector)) / self.TrainY_vector.size)
            for i in range(len(self.Weight)):
                if(self.Weight[i]<0):
                    self.Weight[i]=self.Weight[i]-alpha*((temp[i])-reg_param)
                else:
                    self.Weight[i]=self.Weight[i]-alpha*((temp[i])+reg_param)
            temp_cost=self.cost_func(self.TrainY_vector,h,"l1",reg_param)
        else:
            h = self.sigmoid(self.ValX_vector @ self.Weight)
            temp=((self.ValX_vector.T @ (h - self.ValY_vector)) / self.ValY_vector.size)
            for i in range(len(self.Weight)):
                if(self.Weight[i]<0):
                    self.Weight[i]=self.Weight[i]-alpha*((temp[i])-reg_param)
                else:
                    self.Weight[i]=self.Weight[i]-alpha*((temp[i])+reg_param)
            temp_cost=self.cost_func(self.TrainY_vector,h,"l1",reg_param)
            
        return temp_cost,self.accuracy("train")