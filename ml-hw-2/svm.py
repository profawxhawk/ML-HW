import numpy as np
import matplotlib.pyplot as plt
class SVM:
    weight=0
    bias=0
    features=0
    data_split=0
    def __init__(self,dimensions,features):
        self.weight=np.zeros(dimensions)
        self.bias=0
        self.features=features
    def set(self,d):
        self.data_split=d
    def fit(self, data):
        pass

    def predict(self):
        return np.sign(np.dot(self.features,self.weight)+self.bias)