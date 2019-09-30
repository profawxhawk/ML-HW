import h5py
import numpy as np
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
    print(data)
