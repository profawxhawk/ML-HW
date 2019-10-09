import h5py
import numpy as np
import pandas as pd
from scipy import stats
from svm import SVM
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import warnings
from sklearn.neighbors import LocalOutlierFactor
warnings.filterwarnings("ignore")
def ker_1(X,Y):
    return np.square(np.dot(X, Y.T))
def ker_2(X,Y):
    gram_matrix = np.zeros((X.shape[0],Y.shape[0]))
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            gram_matrix[i,j] = np.exp(-1*np.linalg.norm(x-y)**2)
    return gram_matrix
def ker_3(X, Y):
    M = np.array([[2,0],[0,1]])
    return np.dot(np.dot(X, M), Y.T)
def fit_and_contour(X,Y,myker):
    cols=[]
    for i in Y:
        if i==0:
            cols.append('black')
        else:
            cols.append('blue')
    plt.scatter(X[:,0], X[:,1],c=cols)
    svc= SVC(kernel=myker)
    svc.fit(X,Y.ravel())
    # Get limit of axis to form meshgrid
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Creating the meshgrid
    x1 = np.linspace(xlim[0], xlim[1], 200)
    y1 = np.linspace(ylim[0], ylim[1], 200)
    ym1, xm1 = np.meshgrid(y1, x1)
    # stacking everything to get all possible combinations
    xy = np.vstack([xm1.ravel(), ym1.ravel()]).T
    if myker==ker_3:
        Z = svc.predict(np.c_[xm1.ravel(), ym1.ravel()])
        Z = Z.reshape(xm1.shape)
        plt.contourf(xm1, ym1, Z > 0,alpha=0.4)
        ax.contour(xm1, ym1, Z, colors='r', levels=[-1, 0, 1],linestyles=['--', '-', '--'])
        plt.show()
    else:
        Z = svc.decision_function(xy).reshape(xm1.shape)
        # Plotting the boundary only for =-1,=0 and =1.
        plt.contourf(xm1, ym1, Z > 0,alpha=0.4)
        ax.contour(xm1, ym1, Z, colors='r', levels=[-1, 0, 1],linestyles=['--', '-', '--'])
        plt.show()
    
def ploty(x,y):
    cols=[]
    for i in y:
        if i==0:
            cols.append('black')
        else:
            cols.append('blue')
    plt.scatter(x[:,0],x[:,1],c=cols)
    plt.show()

def extract_h5(file_name):
    f = h5py.File(file_name,'r')
    data=[]
    for key in f.keys():
        group = f[key]
        if key=='y':
            temp=np.asarray(group[()])
            temp=np.reshape(temp,(temp.shape[0],1))
            data.append(temp)
        else:
            data.append(np.asarray(group[()]))
    return data

def iqr(da):
    o=pd.DataFrame(da)
    Q1 = o.quantile(0.25)
    Q3 = o.quantile(0.75)
    IQR = Q3 - Q1
    out = o[~((o < (Q1 - 0.1 * IQR)) |(o > (Q3 + 0.1 * IQR))).any(axis=1)]
    return out

def z_score(temp,kernel):
    z = np.abs(stats.zscore(temp[1]))
    k = temp[1][(z < 1.5).all(axis=1)]
    z1 = np.abs(stats.zscore(temp[-1]))
    k1 = temp[-1][(z1 < 1.5).all(axis=1)]
    input=np.append(k,k1,axis=0)
    print(input.shape)
    output=[1]*len(k)
    output0=[0]*len(k1)
    output=output+(output0)
    output=np.asarray(output).reshape(input.shape[0],1)
    print(output.shape)
    data=[input,output]
    fit_and_contour(data[0],data[1],kernel)
        
def outlier_remove_lof(x,y):
    clf = LocalOutlierFactor()
    y_pred = clf.fit_predict(x)
    scores = clf.negative_outlier_factor_
    temp=-scores
    x1=[]
    y1=[]
    for i in range(len(temp)):
        if temp[i]<1.2:
            x1.append(x[i])
            y1.append(y)
    return x1,y1
            
def seg(X,Y):
    ones=[]
    zeros=[]
    for i,v in enumerate(Y):
        if v==0:
            zeros.append(X[i])
        else:
            ones.append(X[i])
            
    return {-1:np.array(zeros), 1:np.array(ones)}

def split_data(data):
    split=(int)(4*(len(data[1]))/5)
    x1=np.array(data[0])
    y1=np.array(data[1])
    temp=np.column_stack((x1,y1))
    return (temp[0:split],temp[split:])

def outlier_remove(index,data):
    if index==2:
        kernel=ker_3
    else:
        kernel='rbf'
    svc= SVC(kernel='rbf')
    X=data[0]
    Y=data[1]
    svc.fit(X,Y.ravel())
    data_woo=[]
    output_woo=[]
    for (i,j) in zip(X,Y):
        ans=svc.predict([i])
        if ans==-1:
            ans=0
        if ans==j:
            data_woo.append(i)
            output_woo.append(j)
    
    seg_data=seg(data_woo,output_woo)   
    x,y=outlier_remove_lof(seg_data[1],1)
    x1,y1=outlier_remove_lof(seg_data[-1],-1)
    x2=np.asarray(x+x1)
    y2=np.asarray(y+y1)
    z_score(seg_data,kernel)
    
    
if (__name__ == "__main__"):
    data=[]
    for i in range(1,6):
        data.append(extract_h5('./q1_datasets/data_'+str(i)+'.h5'))
    # print("Data points scatter plot")
    # for i in data:
    #     ploty(i[0],i[1])
    # print("Decision boundaries for datasets")
    # fit_and_contour(data[0][0],data[0][1],ker_1)
    # fit_and_contour(data[1][0],data[1][1],ker_2)
    # fit_and_contour(data[2][0],data[2][1],ker_3)
    # fit_and_contour(data[3][0],data[3][1],'rbf')
    # fit_and_contour(data[4][0],data[4][1],'rbf')
    
    # print("Outlier removed data")
    # for i,j in enumerate(data):
    #     outlier_remove(i,j)

    # print("Linear svm")
    # print()
    # for i in range(3,5):
    #     print("for dataset "+str(i+1))
    #     train,test=split_data(data[i])
    #     svm_model=SVM(train,test)
    #     svm_model.linear_svm()
    
    # print("RBF svm")
    # print()
    # for i in range(3,5):
    #     print("for dataset "+str(i+1))
    #     train,test=split_data(data[i])
    #     svm_model=SVM(train,test)
    #     svm_model.rbf_svm()
    train,test=split_data(data[4])
    svm_model=SVM(train,test)
    svm_model.rbf_svm()
    
                
