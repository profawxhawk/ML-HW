import codecs,numpy as np
from sklearn.linear_model import LogisticRegression
import logistic_regression as lr
import warnings
from sklearn.metrics import roc_curve
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
def confusion_matrix_count(x,y,z,threshold):
        h=sigmoid(x @ z)
        h=np.where(h >= threshold, 1, 0)
        tp=0
        fp=0
        fn=0
        tn=0
        for i in range(len(y)):
            if y[i]==1 and h[i]==1:
                tp+=1
            if y[i]==1 and h[i]==0:
                fn+=1
            if y[i]==0 and h[i]==1:
                fp+=1
            if y[i]==0 and h[i]==0:
                tn+=1
        return tp,fp,fn,tn
def one_vs_all(train_images,train_labels,test_images,test_labels,model,iter):
    weight_array=[]
    roc_outer=[]
    fpr_li=[]
    tpr_li=[]
    for i in range(10):
        lr_model=lr.logistic_regression()
        lr_model.TrainX_vector=train_images
        lr_model.TrainY_vector=train_labels
        lr_model.TestX_vector=test_images
        lr_model.TestY_vector=test_labels
        lr_model.label_change(i)
        print(model+" Gradient descent started for label "+str(i))
        clf=LogisticRegression(penalty=model,max_iter=iter,C=0.001,solver='liblinear')
        clf.fit(lr_model.TrainX_vector,lr_model.TrainY_vector.ravel())
        score_train=clf.score(lr_model.TrainX_vector,lr_model.TrainY_vector.ravel())
        score_test=clf.score(lr_model.TestX_vector,lr_model.TestY_vector.ravel())
        if model=="l2":
            # roc_inner=[]
            # for j in range(1,11):
            #     tp,fp,fn,tn=confusion_matrix_count(lr_model.TestX_vector,lr_model.TestY_vector,clf.coef_[0].T.reshape(len(clf.coef_[0]),1),j/10)
            #     roc_inner.append([tp/(tp+fn),fp/(fp+tn)])
            # roc_outer.append(roc_inner)
            probs = clf.predict_proba(lr_model.TestX_vector)
            probs = probs[:, 1]
            fpr, tpr, thresholds = roc_curve(lr_model.TestY_vector, probs)
            fpr_li.append(fpr)
            tpr_li.append(tpr)
        print("Train Accuracy: "+str(score_train))
        print("Test Accuracy: "+str(score_test))
        weight_array.append(clf.coef_[0].T.reshape(len(clf.coef_[0]),1))
        print(model+" Gradient descent ended for label "+str(i))
    return weight_array,roc_outer,tpr_li,fpr_li
def accuracy(x,y):
    length=len(x)
    count=0;
    for i in range(len(x)):
        if y[i]==x[i]:
            count+=1
    return (count/length)

def multi_class_accuracy(weight_array,x,y,model):
    final_array=[]
    for i in weight_array:
        final_array.append(sigmoid(x @ i))
    np_weight = np.array(final_array)
    final_label=[]
    for i in range((np_weight.shape[1])):
        temp=np.where(np_weight[:,i]==np.max(np_weight[:,i]))[0][0]
        final_label.append([temp])
    np_final_label = np.array(final_label)
    print(str(accuracy(y,np_final_label)))
    
if (__name__ == "__main__"):
    
    train_images=open('./MNSIT/train-images.idx3-ubyte','rb')
    test_images=open('./MNSIT/t10k-images.idx3-ubyte','rb')
    train_labels=open('./MNSIT/train-labels.idx1-ubyte','rb')
    test_labels=open('./MNSIT/t10k-labels.idx1-ubyte','rb')
    train_images,train_labels=ubyte_to_data(train_images,train_labels)
    test_images,test_labels=ubyte_to_data(test_images,test_labels)
    lr_model=lr.logistic_regression()
    lr_model.TrainX_vector=train_images
    lr_model.TrainY_vector=train_labels
    lr_model.TestX_vector=test_images
    lr_model.TestY_vector=test_labels
    weight_array,a,b,c=one_vs_all(train_images,train_labels,test_images,test_labels,"l1",15)
    print("Final Train accuracy for l1 regularization ")
    multi_class_accuracy(weight_array,train_images,train_labels,"l1")
    print("Final Test accuracy for l1 regularization ")
    multi_class_accuracy(weight_array,test_images,test_labels,"l1")
    weight_array1,roc_outer,tpr,fpr=one_vs_all(train_images,train_labels,test_images,test_labels,"l2",10)
    print("Final Train accuracy for l2 regularization ")
    multi_class_accuracy(weight_array1,train_images,train_labels,"l2")
    print("Final Test  accuracy for l2 regularization ")
    multi_class_accuracy(weight_array1,test_images,test_labels,"l2")
    lr.plt.title("ROC curve for L2 regularization")
    lr.plt.xlabel('FPR', fontsize=18)
    lr.plt.ylabel('TPR', fontsize=18)
    
    for i in range(len(fpr)):
        lr.plt.plot(fpr[i],tpr[i],label=('Class for '+str(i)))
    lr.plt.legend(loc='lower right')
    lr.plt.show()
    
    # roc=lr.np.array(roc)
    # for i in range(len(roc)):
    #     lr.plt.plot(roc[i][:,1],roc[i][:,0])
    # lr.plt.show()
    
    
    
    

    
    