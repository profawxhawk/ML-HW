import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from random import shuffle
from sklearn import linear_model
from sklearn.model_selection import KFold,GridSearchCV
class linear_regression:
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
        
    def regularization_hyper_param_tune(self,model,least_fold):
        
        if model=='l1':
            lin_reg_regularization = linear_model.Lasso()
        else:
            lin_reg_regularization = linear_model.Ridge()
            
        param={'alpha':[1,0.1,0.01,0.001,0.0001,10,20,15,5,25,30]}
        fold_cv=KFold(n_splits=5,shuffle=True)
        clf=GridSearchCV(lin_reg_regularization, param, cv=fold_cv)
        clf.fit(self.train_array_x[least_fold],self.train_array_y[least_fold])
        return (clf.best_params_)
        
    def split_equal(self,list,n,val_size):
        return [list[i*val_size:(i+1)*val_size] for i in range(n)]
    
    def random_split(self,dataset,fold_size):
        val_size=math.floor(len(dataset)/fold_size)
        temp_list=list(range(val_size*fold_size))
        shuffle(temp_list)
        self.split_index=self.split_equal(temp_list,fold_size,val_size)
        
    def find_vectors_k_fold(self,dataset,fold_size):
        self.random_split(dataset,fold_size)
        input_val=pd.DataFrame([],columns=['Age1','Age2','Age3','Factor1','Factor2','Factor3','Factor4','Factor5','Factor6','Factor7'])
        output_val=pd.DataFrame([],columns=['Output'])
        input_train=pd.DataFrame([],columns=['Age1','Age2','Age3','Factor1','Factor2','Factor3','Factor4','Factor5','Factor6','Factor7'])
        output_train=pd.DataFrame([],columns=['Output'])
        
        for fold_count in range(1,fold_size+1):
            print(fold_count)
            for i in range(len(self.split_index[fold_count-1])):
                input_val=input_val.append(dataset[self.split_index[fold_count-1][i]:self.split_index[fold_count-1][i]+1][dataset.columns[1:len(dataset.keys())-1]])
                output_val=output_val.append(dataset[self.split_index[fold_count-1][i]:self.split_index[fold_count-1][i]+1][dataset.columns[len(dataset.keys())-1:]])
            for i in range(len(dataset)):
                if i not in self.split_index[fold_count-1]:
                    input_train=input_train.append(dataset[i:i+1][dataset.columns[1:len(dataset.keys())-1]])
                    output_train=output_train.append(dataset[i:i+1][dataset.columns[len(dataset.keys())-1:]])
            
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
            
            
        
        
    def find_vector(self,dataset,fold_size):
        val_size=math.floor(len(dataset)/fold_size)
        for fold_count in range(1,fold_size+1):
            input_val=dataset[(fold_count-1)*val_size:fold_count*val_size][dataset.columns[1:len(dataset.keys())-1]]
            output_val=dataset[(fold_count-1)*val_size:fold_count*val_size][dataset.columns[len(dataset.keys())-1:]]
            
            input_train=dataset[0:(fold_count-1)*val_size][dataset.columns[1:len(dataset.keys())-1]]
            output_train=dataset[0:(fold_count-1)*val_size][dataset.columns[len(dataset.keys())-1:]]
            
            input_train=input_train.append(dataset[(fold_count)*val_size:][dataset.columns[1:len(dataset.keys())-1]])
            output_train=output_train.append(dataset[(fold_count)*val_size:][dataset.columns[len(dataset.keys())-1:]])
            
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
        
    def convert_data_to_csv(self,file):
        f=open(file,'r')
        k=[]
        for line in f:
            temp=line.split()[0]
            op=[]
            if temp=='M':
                op.append(1.0)
                op.append(0.0)
                op.append(0.0)
            if temp=='F':
                op.append(0.0)
                op.append(1.0)
                op.append(0.0)
            if temp=='I':
                op.append(0.0)
                op.append(0.0)
                op.append(1.0)
            for i in line.split()[1:]:
                op.append(float(i))
            k.append(op)
        print(k[0]) 
        new_data=pd.DataFrame(k,columns=['Age1','Age2','Age3','Factor1','Factor2','Factor3','Factor4','Factor5','Factor6','Factor7','Output'])
        name=input('Input a file name in the format filename.csv')
        new_data.to_csv(name)
        return ("./"+name)

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

    def optimise_weight_normal(self,index):
        self.Weight=np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(self.train_array_x[index]),self.train_array_x[index])),np.transpose(self.train_array_x[index])),self.train_array_y[index])

    def cost_func_train(self,fold_count):
        temp=np.matmul(self.train_array_x[fold_count-1],self.Weight)
        temp=np.sqrt(((temp-self.train_array_y[fold_count-1])**2).mean())
        self.Train_rmse=temp
        return temp
        
    def cost_func_val(self,fold_count):
        temp1=np.matmul(self.val_array_x[fold_count-1],self.Weight)
        temp1=np.sqrt(((temp1-self.val_array_y[fold_count-1])**2).mean())
        self.Val_rmse=temp1
        return temp1
            
    def loss_differential(self,test,fold_count):
        if test=="True":
            temp=np.matmul(self.val_array_x[fold_count-1],self.Weight)
            temp=temp-self.val_array_y[fold_count-1]
        else:
            temp=np.matmul(self.train_array_x[fold_count-1],self.Weight)
            temp=temp-self.train_array_y[fold_count-1]
        return np.transpose(temp)

    def gradient_descent(self,alpha,fold_count,test="False"):
        # cost=[0.000]
        # print("Gradient descent started")
        # for i in range(iterations):
            # for i in range(len(self.TrainX_vector)):
            #     temp=np.matmul(self.TrainX_vector[i],self.Weight)
            #     temp=temp-self.TrainY_vector[i]
            #     temp1=np.zeros((self.TrainX_vector.shape[1],1))
            #     temp1=self.TrainX_vector[i].reshape((self.TrainX_vector.shape[1],1)) 
            #     rty=(alpha*(np.matmul(temp1,temp))).reshape((self.TrainX_vector.shape[1],1)) 
            #     self.Weight=self.Weight-rty
            #     temp_cost=self.cost_func_train()
            # print(temp_cost)
        temp=self.loss_differential(test,fold_count)
        temp_cost=0
        if test=="False":
            self.Weight=self.Weight-(np.transpose((alpha*(np.matmul((temp),(self.train_array_x[fold_count-1])))))/len(self.train_array_x[fold_count-1]))
            temp_cost=self.cost_func_train(fold_count)
        else:
            self.Weight=self.Weight-(np.transpose((alpha*(np.matmul((temp),(self.val_array_x[fold_count-1])))))/len(self.val_array_x[fold_count-1]))
            temp_cost=self.cost_func_val(fold_count)
        # if i>=0:
        #     # if abs(temp_cost-cost[-1])<=0.00001:
        #     #     break
        #     if i%10==0:
        #         #print(temp_cost)
        #         cost.append(temp_cost)
        # if test=="False":
        #     self.Train_rmse=sum(cost)/len(cost)
        # else:
        #     self.Val_rmse=sum(cost)/len(cost) 
        # print("Gradient descent ended")
        return temp_cost
    
    def gradient_descent_ridge(self,alpha,fold_count,reg_param,test="False"):
        temp=self.loss_differential(test,fold_count+1)
        temp_cost=0 
        if test=="False":
            self.Weight=self.Weight-(np.transpose((alpha*((np.matmul((temp),(self.train_array_x[fold_count]))/len(self.train_array_x[fold_count]))+2*reg_param*(np.transpose(self.Weight))))))
            temp_cost=self.cost_func_train(fold_count)
        else:
            self.Weight=self.Weight-(np.transpose((alpha*((np.matmul((temp),(self.val_array_x[fold_count]))/len(self.val_array_x[fold_count]))+2*reg_param*(np.transpose(self.Weight))))))
            temp_cost=self.cost_func_val(fold_count)
        return temp_cost
    
    
    def gradient_descent_lasso(self,alpha,fold_count,reg_param,test="False"):
        temp=self.loss_differential(test,fold_count+1)
        temp_cost=0 
        temp1=0
        if test=="False":
            temp1=np.transpose(alpha*(np.matmul((temp),(self.train_array_x[fold_count]))/len(self.train_array_x[fold_count])))
        else:
            temp1=np.transpose(alpha*(np.matmul((temp),(self.val_array_x[fold_count]))/len(self.val_array_x[fold_count])))
        for i in range(len(self.Weight)):
            if(self.Weight[i]<0):
                self.Weight[i]=self.Weight[i]-((temp1[i])-alpha*reg_param)
            else:
                self.Weight[i]=self.Weight[i]-((temp1[i])+alpha*reg_param)
                
        if test=="False":
            temp_cost=self.cost_func_train(fold_count)
        else:
            temp_cost=self.cost_func_val(fold_count)
                
        return temp_cost
    
    def find_vector_no_folds(self,dataset):
        dataset=dataset.sample(frac=1)
        input_train=dataset[0:][dataset.columns[0:len(dataset.keys())-1]]
        output_train=dataset[0:][dataset.columns[len(dataset.keys())-1:]]
        
        trainX=input_train.reset_index()
        trainY=output_train.reset_index()
        
        self.TrainX_vector=self.convert_to_numpy(trainX,1)
        self.TrainY_vector=self.convert_to_numpy(trainY)
        
        self.train_array_x.append(self.TrainX_vector)
        self.train_array_y.append(self.TrainY_vector)
