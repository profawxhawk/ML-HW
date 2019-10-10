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
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings("ignore")
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
    for i in range(1,2):
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
    temp=unpickle('./cifar-10-batches-py/batches.meta')
    labels=temp[b'label_names']
    for i in range(len(labels)):
        labels[i]=(labels[i].decode("utf-8"))
    return train_data,labels

def normalize123(data):
    data=data.astype('float64')
    mean = np.mean(data, axis = 0)
    data -= mean
    return data

def roc(ovo,X_test,Y_test):
    y = label_binarize(Y_test, classes=[0,1,2,3,4,5,6,7,8,9])
    prob = ovo.decision_function(X_test)
    fpr=[0]*10
    tpr=[0]*10
    for i in range(10):
        fpr[i],tpr[i],temp=roc_curve(y[:, i],prob[:, i])
    plt.title('Receiver Operating Characteristic')
    for i in range(10):
        plt.plot(fpr[i],tpr[i],label=('Class for '+str(i)))
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.legend(loc='bottom right')
    plt.show()
    
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
    roc(mod,X_test,Y_test)
    print(mod.estimators_[0].coef_)

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
    roc(mod,X_test,Y_test)
    print(mod.estimators_[0].dual_coef_)
    
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
    roc(mod,X_test,Y_test)
    print(mod.estimators_[0].dual_coef_)
    
def pca(train):
    pca = PCA(n_components=100, random_state=0, svd_solver='randomized')
    pca.fit(train)
    X_train = pca.transform(train)
    return X_train
def pca_proc(i):
    return pca(i)
    
def split_train(data):
    split=[]
    split.append([data[0][:2000],data[1][:2000]])
    split.append([data[0][2000:4000],data[1][2000:4000]])
    split.append([data[0][4000:6000],data[1][4000:6000]])
    split.append([data[0][6000:8000],data[1][6000:8000]])
    split.append([data[0][8000:],data[1][8000:]])
    return split
def change_label(index,input):
    temp=np.where(input == i, 1, 0)
    return temp
def one_vs_all(ker,TrainX,TrainY,TestX,TestY):
    fpr_li=[]
    tpr_li=[]
    values=[]
    for i in range(10):
        mod=SVC(kernel=ker, probability=True)
        temp_trainy=change_label(i,TrainY)
        temp_testy=change_label(i,TestY)
        mod.fit(TrainX,temp_trainy.ravel())
        probs = mod.decision_function(TestX)
        fpr, tpr, thresholds = roc_curve(temp_testy, probs)
        fpr_li.append(fpr)
        tpr_li.append(tpr)
        y_pred = mod.predict(TestX)
        values.append(mod.predict_proba(TestX)[:,1])
    predict=[]
    for j in range(2000):
        max_val=values[0][j]
        index=0
        for i in range(1,10):
            if(values[i][j]>max_val):
                max_val=values[i][j]
                index=i
        predict.append(index)
    predict=np.asarray(predict)  
    accuracy = accuracy_score(TestY,predict)
    print("Model accuracy is: ", accuracy)
    results = confusion_matrix(TestY,predict) 
    print("Confusion Matrix: ",results) 
    for i in range(len(fpr_li)):
        plt.plot(fpr_li[i],tpr_li[i],label=('Class for '+str(i)))
    plt.legend(loc='lower right')
    plt.show()
def find_pair(TrainX,TrainY,i,j):
    X=[]
    Y=[]
    for k in range(len(TrainY)):
        if TrainY[k]==[i] or TrainY[k]==[j]:
            X.append(TrainX[k])
            Y.append(TrainY[k])
    return np.asarray(X),np.asarray(Y)
            
def one_vs_one(ker,TrainX,TrainY,TestX,TestY):
    values=[]
    for i in range(10):
        for j in range(i+1,10):
            X,Y=find_pair(TrainX,TrainY,i,j)
            mod=SVC(kernel=ker, probability=True)
            mod.fit(X,Y.ravel())
            values.append(mod.predict(TestX))
    predict=[]
    for j in range(2000):
        max_count=[0]*10
        for i in range(len(values)):
            max_count[values[i][j]]=max_count[values[i][j]]+1
        temp=-10
        index=0
        for k in range(10):
            if max_count[k]>temp:
                temp=max_count[k]
                index=k
        predict.append(index)
    predict=np.asarray(predict)
    accuracy = accuracy_score(TestY,predict)
    print("Model accuracy is: ", accuracy)
    results = confusion_matrix(TestY,predict) 
    print("Confusion Matrix: ",results) 
    
def call_svm(X,Y,x,y):
    print("linear svm ovo start")
    svm_linear(X,Y,x,y,'ovo')
    # one_vs_one('linear',X,Y,x,y)
    print("linear svm ovr start")
    svm_linear(X,Y,x,y,'ovr')
    # one_vs_all('linear',X,Y,x,y)
    print("rbf svm ovo start")
    svm_rbf(X,Y,x,y,'ovo')
    # one_vs_one('rbf',X,Y,x,y)
    print("rbf svm ovr start")
    svm_rbf(X,Y,x,y,'ovr')
    # one_vs_all('rbf',X,Y,x,y)
    print("poly svm ovo start")
    svm_poly(X,Y,x,y,'ovo')
    # one_vs_one('poly',X,Y,x,y)
    print("poly svm ovr start")
    svm_poly(X,Y,x,y,'ovr')
    # one_vs_all('poly',X,Y,x,y)

if (__name__ == "__main__"):
    train_data,labels=extract_data()
    for i in train_data:
        i[0]=normalize123(i[0])
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
    