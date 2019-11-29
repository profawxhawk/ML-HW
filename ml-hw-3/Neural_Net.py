import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
# import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 
class Neural_Net:
    layers=0
    number_of_nodes=[]
    activation=""
    learning_rate=0.0
    neural_net_value_matrix=[]
    weights=[]
    biases=[]
    network_labels=[]
    def relu_activation(self,input):
        output=np.maximum(0,input)
        return output
    def sigmoid_activation(self,input):
        input = np.clip( input, -500, 500 )
        output=1.0/(1.0 + np.exp(-input))
        return output
    def linear_activation(self,input):
        output=input
        return output
    def tanh_activation(self,input):
        output=np.tanh(input)
        return output
    def softmax_activation(self,input):
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)

        # input=np.clip(input,-500,500)
        # temp = np.exp(input)
        # output= temp / np.sum(temp,axis=0)
        # output[output==0.0]=0.01
        # output[output==1.0]=0.99
        # return output

    def sigmoiddiff(self, s):
        return s * (1 - s)
    
    def tanhdiff(self, s):
        return (1-np.square(s))
    
    def lineardiff(self, s):
        return 1
    
    def Reludiff(self, s):
        return 1.0*(s>0)
    

    def __init__(self,layers,nodes,act,rate):
        self.layers=layers
        self.number_of_nodes=nodes
        self.activation=act
        self.learning_rate=rate

    def compute_cost(x,y):
        m=y.shape[1]              
        cost = - (1 / m) * np.sum(
            np.multiply(y, np.log(x)) + np.multiply(1 - y, np.log(1 - x)))
        return cost

    def forward_prop(self):
        for i in range(len(self.neural_net_value_matrix)-1):
            output=(self.neural_net_value_matrix[i]@(self.weights[i]))+self.biases[i]
            if i!=(len(self.neural_net_value_matrix)-2):
                if self.activation=="Relu":
                    self.neural_net_value_matrix[i+1] = self.relu_activation(output)
                elif self.activation=="linear":
                    self.neural_net_value_matrix[i+1] = self.linear_activation(output)
                elif self.activation=="tanh":
                    self.neural_net_value_matrix[i+1] = self.tanh_activation(output)
                elif self.activation=="sigmoid":
                    self.neural_net_value_matrix[i+1] = self.sigmoid_activation(output)
            else:
                self.neural_net_value_matrix[i+1] = self.softmax_activation(output)

    def cross_entropy(self,x,y):
        final_loss=0
        for i,j in zip(x,y):
            self.neural_net_value_matrix[0]=i.reshape(1,784)
            self.forward_prop()
            temp_loss=(j.reshape(1,10)*np.log(self.neural_net_value_matrix[-1]))
            temp_loss=np.sum(temp_loss)
            temp_loss=-temp_loss
            final_loss+=temp_loss
        return (final_loss/y.shape[0])

    def back_prop(self):
        error=(self.neural_net_value_matrix[-1]-self.network_labels)
        output_delta=error
        updates=[]
        bias_update=[]
        updates.append((self.neural_net_value_matrix[-2].T@output_delta))
        bias_update.append(output_delta)
        i=-2
        while i>=(-1*len(self.weights)):
            error=output_delta@self.weights[i+1].T
            diff=[]
            output=self.neural_net_value_matrix[i]
            if self.activation=="Relu":
                diff = self.Reludiff(output)
            elif self.activation=="linear":
                diff = self.lineardiff(output)
            elif self.activation=="tanh":
                diff = self.tanhdiff(output)
            elif self.activation=="sigmoid":
                diff = self.sigmoiddiff(output)
            output_delta=diff*error
            updates.append((self.neural_net_value_matrix[i-1].T@output_delta) )
            bias_update.append(output_delta)
            i=i-1
        updates.reverse()
        bias_update.reverse()
        return updates,bias_update



    def update_weights(self,updates,bias_update):
        for i in range(len(self.weights)):
            self.weights[i]-=self.learning_rate*updates[i]
            self.biases[i]-=self.learning_rate*np.sum(bias_update[i], axis=0, keepdims=True)

    def create_network(self):
        input=np.zeros((1,self.number_of_nodes[0]))
        output=np.zeros((1,self.number_of_nodes[-1]))
        self.neural_net_value_matrix.append(input)
        for i in range(1,len(self.number_of_nodes)-1):
            hidden=np.zeros((1,self.number_of_nodes[i]))
            self.biases.append(np.zeros((1,self.number_of_nodes[i])))
            self.neural_net_value_matrix.append(hidden)
        self.biases.append(np.zeros((1,self.number_of_nodes[-1])))
        self.neural_net_value_matrix.append(output)
        for i in range(len(self.neural_net_value_matrix)-1):
            self.weights.append(np.random.randn(self.neural_net_value_matrix[i].shape[1],self.neural_net_value_matrix[i+1].shape[1]))
            self.weights[-1]=(self.weights[-1])/100



        
    def fit(self,x,y,size,epochs):
        self.create_network()
        loss=[]
        for k in range(epochs):
            print("epoch: "+str(k))
            count=1
            for i,j in zip(x,y):
                count+=1
                print(count)
                self.neural_net_value_matrix[0]=i.T.reshape(1,784)
                self.network_labels=j.T.reshape(1,10)
                self.forward_prop()
                updates,bias_update=self.back_prop()
                if(np.isnan(self.neural_net_value_matrix[-1]).any()):
                    print(self.weights)
                    print(count)
                    break
                self.update_weights(updates,bias_update)
            loss.append(self.cross_entropy(x,y))
        # np.save('./MNSIT/relu_weights.npy',np.asarray(self.weights))
        # np.save('./MNSIT/relu_bias.npy',np.asarray(self.bias))
        # plt.plot(history.history['loss'])
        # plt.ylabel('iterations')
        # plt.xlabel('Training loss')
        # plt.show()

    
    def predict(self,x):
        pred=[]
        for i in x:
            self.neural_net_value_matrix[0]=i.T.reshape(1,784)
            self.forward_prop()
            pred.append(self.neural_net_value_matrix[-1])
        return np.asarray(pred)

    def score(self,y_pred,y_labels):
        count=0
        for i,j in zip(y_pred,y_labels):
            result = np.where(i == np.amax(i))
            if j[result]==1:
                count+=1
        return (count/10000)


# if __name__ == "__main__ ":

train_images=np.load('./MNSIT/train_images_np.npy')/255
test_images=np.load('./MNSIT/test_images_np.npy')/255
train_labels=np.load('./MNSIT/train_labels_np.npy')
test_labels=np.load('./MNSIT/test_labels_np.npy')
train_labels1=np.load('./MNSIT/train_labels_1_np.npy')
test_labels1=np.load('./MNSIT/test_labels_1_np.npy')
# net=Neural_Net(5,[784,256,128,64,10],"Relu",0.1)
# net.fit(train_images,train_labels,1,10)
# pred=net.predict(test_images)
# print(net.score(pred,test_labels))
# net=Neural_Net(5,[784,256,128,64,10],"tanh",0.1)
# net.fit(train_images,train_labels,1,10)
# pred=net.predict(test_images)
# print(net.score(pred,test_labels))
# net=Neural_Net(5,[784,256,128,64,10],"linear",0.1)
# net.fit(train_images,train_labels,1,10)
# pred=net.predict(test_images)
# print(net.score(pred,test_labels))
net=Neural_Net(7,[784,64,32,32,32,32,10],"sigmoid",0.1)
net.fit(train_images,train_labels,1,10)
pred=net.predict(test_images)
print(net.score(pred,test_labels))
# mlp=MLPClassifier(hidden_layer_sizes=(256,128,64),activation='relu',max_iter=100,learning_rate='constant',learning_rate_init=0.1)
# mlp.fit(train_images,train_labels)
# predictions = mlp.predict(test_images)
# print(confusion_matrix(test_labels,predictions))

# Relu sklearn
# mlp=MLPClassifier(hidden_layer_sizes=(256,128,64),activation='relu',max_iter=100,learning_rate='constant',learning_rate_init=0.1,solver='sgd', verbose=10)
# mlp.fit(train_images,train_labels1)
# predictions = mlp.predict(test_images)
# print(confusion_matrix(test_labels1,predictions))
# print("Training set score: %f" % mlp.score(train_images,train_labels1))
# print("Test set score: %f" % mlp.score(test_images,test_labels1))


# Neural network




