import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pickle
import sklearn.multiclass as sm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def convert(data):
    im = data
    im_r = im[0:1024].reshape(32, 32).astype('float64')
    im_g = im[1024:2048].reshape(32, 32).astype('float64')
    im_b = im[2048:].reshape(32, 32).astype('float64')
    img = np.dstack((im_r, im_g, im_b))
    return normalize(data).flatten()

def extract_data():
    train_data=[]
    for i in range(1,6):
        file='./cifar-10-batches-py/data_batch_'+str(i)
        temp=unpickle(file)
        temp_data=[]
        mat_data=[]
        for j in range(len(temp[b'data'])):
            op=convert(temp[b'data'][j])
            mat_data.append(op)
        temp_data.append(np.asarray(mat_data).astype('float64'))
        temp_data.append(np.asarray(temp[b'labels']).reshape(temp[b'data'].shape[0],1))
        train_data.append(temp_data)
    temp=unpickle('./cifar-10-batches-py/test_batch')
    temp_data=[]
    mat_data=[]
    for j in range(len(temp[b'data'])):
        op=convert(temp[b'data'][j])
        mat_data.append(op)
    temp_data.append(np.asarray(mat_data).astype('float64'))
    temp_data.append(np.asarray(temp[b'labels']).reshape(temp[b'data'].shape[0],1))
    test_data=temp_data
    temp=unpickle('./cifar-10-batches-py/batches.meta')
    labels=temp[b'label_names']
    for i in range(len(labels)):
        labels[i]=(labels[i].decode("utf-8"))
    return train_data,test_data,labels

def normalize123(data,test):
    data=data.astype('float64')
    test=test.astype('float64')
    mean = np.mean(data, axis = 0)
    data -= mean
    test -= mean
    return data,test

def svm_linear(X,Y,x,y,model):
    X_train=X
    X_test=x
    Y_train=Y
    Y_test=y
    svm = SVC(kernel='linear')
    if model=='ovo':
        mod = sm.OneVsOneClassifier(svm)
    else:
        mod = sm.OneVsRestClassifier(svm)
    mod.fit(X_train,Y_train.ravel())
    y_pred = mod.predict(X_test)
    accuracy = accuracy_score(Y_test.ravel(), y_pred)
    print("Model accuracy is: ", accuracy)
    results = confusion_matrix(Y_test.ravel(),y_pred) 
    print("Confusion Matrix: ",results) 

def svm_rbf(X,Y,x,y,model):
    X_train=X
    X_test=x
    Y_train=Y
    Y_test=y
    svm = SVC(kernel='rbf')
    if model=='ovo':
        mod = sm.OneVsOneClassifier(svm)
    else:
        mod = sm.OneVsRestClassifier(svm)
    mod.fit(X_train,Y_train.ravel())
    y_pred = mod.predict(X_test)
    accuracy = accuracy_score(Y_test.ravel(), y_pred)
    print("Model accuracy is: ", accuracy)
    results = confusion_matrix(Y_test.ravel(),y_pred) 
    print("Confusion Matrix: ",results) 
    
def svm_poly(X,Y,x,y,model):
    X_train=X
    X_test=x
    Y_train=Y
    Y_test=y
    svm = SVC(kernel='poly')
    if model=='ovo':
        mod = sm.OneVsOneClassifier(svm)
    else:
        mod = sm.OneVsRestClassifier(svm)
    mod.fit(X_train,Y_train.ravel())
    y_pred = mod.predict(X_test)
    accuracy = accuracy_score(Y_test.ravel(), y_pred)
    print("Model accuracy is: ", accuracy)
    results = confusion_matrix(Y_test.ravel(),y_pred) 
    print("Confusion Matrix: ",results)

def split_train(data):
    split=[]
    split.append([data[0][:2000],data[1][:2000]])
    split.append([data[0][2000:4000],data[1][2000:4000]])
    split.append([data[0][4000:6000],data[1][4000:6000]])
    split.append([data[0][6000:8000],data[1][6000:8000]])
    split.append([data[0][8000:],data[1][8000:]])
    return split

def call_svm(X,Y,x,y):
    print("linear svm ovo start")
    svm_linear(X,Y,x,y,'ovo')
    print("linear svm ovr start")
    svm_linear(X,Y,x,y,'ovr')
#     svm_rbf(X,Y,x,y,'ovo')
#     svm_rbf(X,Y,x,y,'ovr')
#     svm_poly(X,Y,x,y,'ovo')
#     svm_poly(X,Y,x,y,'ovr')

if (__name__ == "__main__"):
    train_data,test_data,labels=extract_data()
    for i in train_data:
        temp_test=test_data
        i[0],temp_test[0]=normalize123(i[0],temp_test[0])
    k_cross_split=split_train(train_data[0])
    
    trainx=[]
    trainy=[]
    
    trainx.append(np.vstack((k_cross_split[1][0],k_cross_split[2][0],k_cross_split[3][0],k_cross_split[4][0])))
    trainy.append(np.vstack((k_cross_split[1][1],k_cross_split[2][1],k_cross_split[3][1],k_cross_split[4][1])))

    trainx.append(np.vstack((k_cross_split[0][0],k_cross_split[2][0],k_cross_split[3][0],k_cross_split[4][0])))
    trainy.append(np.vstack((k_cross_split[0][1],k_cross_split[2][1],k_cross_split[3][1],k_cross_split[4][1])))

    trainx.append(np.vstack((k_cross_split[1][0],k_cross_split[0][0],k_cross_split[3][0],k_cross_split[4][0])))
    trainy.append(np.vstack((k_cross_split[1][1],k_cross_split[0][1],k_cross_split[3][1],k_cross_split[4][1])))

    trainx.append(np.vstack((k_cross_split[1][0],k_cross_split[2][0],k_cross_split[0][0],k_cross_split[4][0])))
    trainy.append(np.vstack((k_cross_split[1][1],k_cross_split[2][1],k_cross_split[0][1],k_cross_split[4][1])))

    trainx.append(np.vstack((k_cross_split[1][0],k_cross_split[2][0],k_cross_split[3][0],k_cross_split[0][0])))
    trainy.append(np.vstack((k_cross_split[1][1],k_cross_split[2][1],k_cross_split[3][1],k_cross_split[0][1])))

    testx=[k_cross_split[0][0],k_cross_split[1][0],k_cross_split[2][0],k_cross_split[3][0],k_cross_split[4][0]]
    testy=[k_cross_split[0][1],k_cross_split[1][1],k_cross_split[2][1],k_cross_split[3][1],k_cross_split[4][1]]

    for i in range(5):
        print("svm for "+str(i+1)+" started")
        call_svm(trainx[i],trainy[i],testx[i],testy[i])
        break
    