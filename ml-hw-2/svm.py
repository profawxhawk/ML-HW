import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 
from sklearn.svm import SVC
class SVM:
    train=0
    test=0
    def __init__(self,train,test):
        self.train=train
        self.test=test
    
    def predict_rbf(self,svc,x):
        s_v=svc.support_vectors_
        coef=svc.dual_coef_[0]
        base=svc.intercept_
        pred=[]
        for i in range(len(x)):
            temp_in=x[i]
            ans=0
            for i in range(len(coef)):
                temp=coef[i]
                ans+=(temp*np.exp(-0.6*np.linalg.norm(s_v[i]-temp_in)**2))
            ans+=base
            if(ans>=0):
                    pred.append(1)
            else:
                    pred.append(0)
    
        return np.asarray(pred)
    def rbf_svm(self):
        svc= SVC(kernel='rbf')
        svc.fit(self.train[:,0:2],self.train[:,2].ravel())
        
        y_pred_train=svc.predict(self.train[:,0:2])
        y_pred1_train=self.predict_rbf(svc,self.train[:,0:2])
        print("Train set")
        print('Accuracy score from SVM predict')
        self.accuracy(y_pred_train)
        print('Accuracy score from self defined predict')
        self.accuracy(y_pred1_train)
        
        y_pred_test=svc.predict(self.test[:,0:2])
        y_pred1_test=self.predict_rbf(svc,self.test[:,0:2])
        print("Test set")
        print('Accuracy score from SVM predict')
        self.accuracy(y_pred_test,True)
        print('Accuracy score from self defined predict')
        self.accuracy(y_pred1_test,True)
        
    def linear_svm(self):
        svc= SVC(kernel='linear')
        svc.fit(self.train[:,0:2],self.train[:,2].ravel())
        
        y_pred_train=svc.predict(self.train[:,0:2])
        y_pred1_train=self.predict_linear(self.train[:,0:2],svc.coef_,svc.intercept_)
        y_pred1_train=np.where(y_pred1_train==-1.,0,1)
        print("Train set")
        print('Accuracy score from SVM predict')
        self.accuracy(y_pred_train)
        print('Accuracy score from self defined predict')
        self.accuracy(y_pred1_train)
        
        y_pred_test=svc.predict(self.test[:,0:2])
        y_pred1_test=self.predict_linear(self.test[:,0:2],svc.coef_,svc.intercept_)
        y_pred1_test=np.where(y_pred1_test==-1.,0,1)
        print("Test set")
        print('Accuracy score from SVM predict')
        self.accuracy(y_pred_test,True)
        print('Accuracy score from self defined predict')
        self.accuracy(y_pred1_test,True)
        
    def accuracy(self,pred,test=False):
        if test==True:
            print ('Accuracy Score :',accuracy_score(self.test[:,2].ravel(),pred))
        else:
            print ('Accuracy Score :',accuracy_score(self.train[:,2].ravel(),pred))
    
    def fit(self, data):
        pass
    
    def predict_linear(self,x,w,b):
        return np.sign(np.dot(x,w.T)+b)