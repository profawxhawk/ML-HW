import codecs,numpy as np
from sklearn.linear_model import LogisticRegression
import logistic_regression as lr
import warnings
warnings.filterwarnings("ignore")
def byte_to_int(b): 
    return int(codecs.encode(b, 'hex'), 16)
def sigmoid(x):
        sig=1 / (1 + np.exp(-x))
        sig[sig == 1.0] = 0.9999
        sig[sig == 0.0] = 0.0001
        return sig
def ubyte_to_data(train_images,train_labels):
    data_images=train_images.read()
    length=byte_to_int(data_images[4:8])
    rows=byte_to_int(data_images[8:12])
    cols=byte_to_int(data_images[12:16])
    temp = np.frombuffer(data_images,dtype = np.uint8, offset = 16) 
    images = temp.reshape(length,rows*cols) 
    
    data_labels=train_labels.read()
    length=byte_to_int(data_labels[4:8])
    temp1 = np.frombuffer(data_labels,dtype = np.uint8, offset = 8) 
    labels = temp1.reshape(length,1) 
    return images,labels
def onevsall(train_images,train_labels,test_images,test_labels):
    weight_array=[]
    for i in range(1,11):
        lr_model=lr.logistic_regression()
        lr_model.TrainX_vector=train_images
        lr_model.TrainY_vector=train_labels
        lr_model.TestX_vector=test_images
        lr_model.TestY_vector=test_labels
        lr_model.label_change(i)
        lr_model.Weight=lr.np.zeros((lr_model.TrainX_vector.shape[1],1))
        print("Gradient descent started for "+str(i))
        for j in range(100):
            (lr_model.gradient_descent_normal(0.0001))
        print("Gradient descent ended for "+str(i))
        weight_array.append(lr_model.Weight)
    return weight_array
def accuracy(x,y):
    length=len(x)
    count=0;
    for i in range(len(x)):
        if y[i]==x[i]:
            count+=1
    return (count/length)
if (__name__ == "__main__"):
    
    train_images=open('./MNSIT/train-images.idx3-ubyte','rb')
    test_images=open('./MNSIT/t10k-images.idx3-ubyte','rb')
    train_labels=open('./MNSIT/train-labels.idx1-ubyte','rb')
    test_labels=open('./MNSIT/t10k-labels.idx1-ubyte','rb')
    train_images,train_labels=ubyte_to_data(train_images,train_labels)
    test_images,test_labels=ubyte_to_data(test_images,test_labels)
    # weight_array=onevsall(train_images,train_labels,test_images,test_labels)
    final_array=[]
    for i in weight_array:
        final_array.append(sigmoid(test_images @ i))
    np_weight = np.array(final_array)
    final_label=[]
    for i in range((np_weight.shape[1])):
        temp=np.where(np_weight[:,i]==np.max(np_weight[:,i]))[0][0]+1
        final_label.append([temp])
    np_final_label = np.array(final_label)
    print(len(np_final_label))
    print(accuracy(test_labels,np_final_label))
    
    
    

    
    