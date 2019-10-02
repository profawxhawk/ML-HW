import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def extract_data():
    train_data=[]
    for i in range(1,6):
        file='./cifar-10-batches-py/data_batch_'+str(i)
        temp=unpickle(file)
        temp_data=[]
        temp_data.append(temp[b'data'])
        temp_data.append(np.asarray(temp[b'labels']).reshape(temp[b'data'].shape[0],1))
        train_data.append(temp_data)
    temp=unpickle('./cifar-10-batches-py/test_batch')
    temp_data=[]
    temp_data.append(temp[b'data'])
    temp_data.append(np.asarray(temp[b'labels']).reshape(temp[b'data'].shape[0],1))
    test_data=temp_data
    temp=unpickle('./cifar-10-batches-py/batches.meta')
    labels=temp[b'label_names']
    for i in range(len(labels)):
        labels[i]=(labels[i].decode("utf-8"))
    return train_data,test_data,labels

if (__name__ == "__main__"):
    train_data,test_data,labels=extract_data()
    