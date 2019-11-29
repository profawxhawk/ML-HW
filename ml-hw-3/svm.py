from sklearn.svm import SVC
import numpy as np;
import warnings
from sklearn.metrics import hinge_loss
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
train_images=np.load('./q2/train_images.npy')
test_images=np.load('./q2/test_images.npy')
train_labels=np.load('./q2/train_labels.npy')
test_labels=np.load('./q2/test_labels.npy')
labels = np.array([0, 1, 2, 3,4,5,6,7,8,9])
error1=[]
for i in range(10):
    svc= SVC(kernel='rbf')
    svc.fit(train_images,train_labels.ravel())
    y_pred1 = svc.decision_function(train_images)
    error=hinge_loss(train_labels, y_pred1, labels)  
    error1.append(error)
    if i==9:
        y_pred = svc.predict(test_images)
        accuracy = accuracy_score(test_labels.ravel(), y_pred)
        print("svcel accuracy on test set is: ", accuracy)
        results = confusion_matrix(test_labels.ravel(),y_pred) 
        print("Confusion Matrix on test set is: ",results) 

        y_pred1 = svc.predict(train_images)
        accuracy1 = accuracy_score(train_labels.ravel(), y_pred1)
        print("svcel accuracy on train set is: ", accuracy1)
        results1 = confusion_matrix(train_labels.ravel(),y_pred1) 
        print("Confusion Matrix on train set is: ",results1) 

print(error1)
plt.plot(error1)
plt.ylabel('iterations')
plt.xlabel('Training loss')
plt.show()
