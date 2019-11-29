#Resource pytorch tutorials
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
# from skimage import transform
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np;
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score 
import torch.nn.functional as F
import random;
import math;
import matplotlib.pyplot as plt
seed = 50
np.random.seed(seed)
torch.manual_seed(seed)
class CNN_fmnist(nn.Module):
    def __init__(self):
        super(CNN_fmnist, self).__init__()
        self.conv1 =  nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d((2,2), 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d((2,2), 2)
        self.fc1 = nn.Linear(3136, 1000)
        self.fc2 = nn.Linear(1000, 10)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = (self.fc2(x))
        return  F.log_softmax(x)

num_epochs = 5
batch_size = 100
learning_rate = 0.01

train_images=np.load('./fashion-mnsit/train_images_np.npy')/255
test_images=np.load('./fashion-mnsit/test_images_np.npy')/255
train_labels=np.load('./fashion-mnsit/train_labels_np.npy')
test_labels=np.load('./fashion-mnsit/test_labels_np.npy')

train_set=np.column_stack((train_images,train_labels))
test_set=np.column_stack((test_images,test_labels))

train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)

cnn = CNN_fmnist()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
num_epochs=10
loss_list = []
acc_list = []
loss1=0
acc1=0
loss_list = []
acc_list = []
loss_temp=0
acc_temp=0
for epoch in range(num_epochs):
    for i, (inputs) in enumerate(train_loader):
        images=inputs[:, :784].reshape(batch_size,1,28,28)
        labels=inputs[:, 784:]
        cnn.double()
        outputs = cnn(images.double())
        loss = criterion(outputs,  torch.max(labels, 1)[1])
        loss_temp=(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total = labels.size(0)
        predicted = torch.max(outputs.data, 1)[1]
        correct = (predicted ==  torch.max(labels, 1)[1]).sum().item()
        acc_temp=(correct / total)
    loss_list.append(loss_temp)
    acc_list.append(acc_temp)

plt.plot(loss_list)
plt.ylabel('iterations')
plt.xlabel('Training loss')
plt.show()
pred1=[]
correct1=[]
final_acc=[]
input_x=torch.rand((100,10))
output_x=torch.rand((100,10))
for i, (inputs) in enumerate(train_loader):
        images=inputs[:, :784].reshape(batch_size,1,28,28)
        labels=inputs[:, 784:]
        cnn.double()
        outputs = cnn(images.double())
        total = labels.size(0)
        input_x = torch.cat((input_x.float(), outputs.float()), dim=0)
        output_x = torch.cat((output_x.float(), labels.float()), dim=0)
        predicted1 = torch.max(outputs.data, 1)[1]
        pred1.append(predicted1)
        correct1.append( torch.max(labels, 1)[1])
        acc_temp=((predicted ==  torch.max(labels, 1)[1]).sum().item()/ total)
        final_acc.append(acc_temp)
print(sum(final_acc)/len(final_acc))
print(confusion_matrix(torch.Tensor(correct1).reshape(-1),torch.Tensor(pred1).reshape(-1)))
pred1=[]
correct1=[]
final_acc=[]
input_x1=torch.rand((100,10))
output_x1=torch.rand((100,10))
for i, (inputs) in enumerate(test_loader):
        images=inputs[:, :784].reshape(batch_size,1,28,28)
        labels=inputs[:, 784:]
        cnn.double()
        outputs = cnn(images.double())
        total = labels.size(0)
        input_x1 = torch.cat((input_x1.float(), outputs.float()), dim=0)
        output_x1 = torch.cat((output_x1.float(), labels.float()), dim=0)
        predicted1 = torch.max(outputs.data, 1)[1]
        pred1.append(predicted1)
        correct1.append( torch.max(labels, 1)[1])
        acc_temp=((predicted ==  torch.max(labels, 1)[1]).sum().item()/ total)
        final_acc.append(acc_temp)
print(sum(final_acc)/len(final_acc))
print(confusion_matrix(torch.Tensor(correct1).reshape(-1),torch.Tensor(pred1).reshape(-1)))

