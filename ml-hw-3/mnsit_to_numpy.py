import codecs,numpy as np
import keras
def byte_to_int(b): 
    return int(codecs.encode(b, 'hex'), 16)
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
        
train_images=open('./fashion-mnsit/train-images-idx3-ubyte','rb')
test_images=open('./fashion-mnsit/t10k-images-idx3-ubyte','rb')
train_labels=open('./fashion-mnsit/train-labels-idx1-ubyte','rb')
test_labels=open('./fashion-mnsit/t10k-labels-idx1-ubyte','rb')
train_images,train_labels=ubyte_to_data(train_images,train_labels)
test_images,test_labels=ubyte_to_data(test_images,test_labels)

# num_classes = 10
# train_labels= keras.utils.to_categorical(train_labels, num_classes)
# test_labels = keras.utils.to_categorical(test_labels, num_classes)

print(train_labels.shape)

np.save('./fashion-mnsit/train_images_np',train_images)
np.save('./fashion-mnsit/train_labels_np1',train_labels)
np.save('./fashion-mnsit/test_images_np',test_images)
np.save('./fashion-mnsit/test_labels_np1',test_labels)
