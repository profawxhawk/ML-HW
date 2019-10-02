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
        
    def linear_svm(self):
        svc= SVC(kernel='linear')
        svc.fit(self.train[:,0:2],self.train[:,2].ravel())
        y_pred=svc.predict(self.test[:,0:2])
        y_pred1=self.predict_linear(self.test[:,0:2],svc.coef_,svc.intercept_)
        y_pred1=np.where(y_pred1==-1.,0,1)
        print('Accuracy score from SVM predict')
        self.accuracy(y_pred)
        print('Accuracy score from self defined predict')
        self.accuracy(y_pred1)
    def accuracy(self,pred):
        print ('Accuracy Score :',accuracy_score(self.test[:,2].ravel(),pred))
    def fit(self, data):
        pass
    
    def predict_linear(self,x,w,b):
        return np.sign(np.dot(x,w.T)+b)