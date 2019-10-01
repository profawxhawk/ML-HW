import h5py
import numpy as np
import svm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
def ker_d1(X,Y):
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
        plt.pcolormesh(xm1, ym1, Z, cmap=plt.cm.Paired)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1],c=cols, edgecolors='k')
        plt.title('3-Class classification using Support Vector Machine with custom'
                ' kernel')
        plt.axis('tight')
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

if (__name__ == "__main__"):
    data=[]
    for i in range(1,6):
        data.append(extract_h5('./q1_datasets/data_'+str(i)+'.h5'))
    # for i in data:
        # ploty(i[0],i[1])
    fit_and_contour(data[2][0],data[2][1],ker_3)
        
    #     svm=svm((2,1),i[0])
    #     input={-1:[],1:[]}
    #     for j in range(len(i[1])):
    #         if i[1][j]==0:
    #             input[-1].append(i[0][j])
    #         else:
    #             input[1].append(i[0][j])
    #     break
    #     svm.set(input)
