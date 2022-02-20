from posixpath import abspath
import sys,os
from pathlib import Path
from turtle import width

from numpy import block
sys.path.insert(0,os.path.join('Code/Preprocess'))
from DataPreparation import dataPrepare

# Package for our NN model
from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, input, hidden, output):
        super(Model, self).__init__()
        self.l1 = nn.Linear(input, hidden)
        self.l2 = nn.Linear(hidden , hidden)
        self.l3 = nn.Linear(hidden, 2) # 2 --> represents number of class
    
    def forward(self,x):
        out = F.relu(self.l1(x)) # Layer 1
        out = F.relu(self.l2(out)) # Layer 2
        out = F.sigmoid(self.l3(out)) # output layer
        return out

class NeuralNet:
    """Code inspired from https://www.kaggle.com/shivammehta007/spam-not-spam-classifier-with-pytorch and did some modifications.

    """
    def __init__(self,path,input_dim,hidden_layer,output_dim,epochs) -> None:
        """A constructor where we define the path to our dataset,hidden layer, output dimensions and epochs.

        Args:
            path (String): [description]
            input_dim ([type]): [description]
            hidden_layer ([type]): [description]
            output_dim ([type]): [description]
            epochs ([type]): [description]
        """
        self.path = path
        self.input_dim = input_dim
        self.hidden_layer = hidden_layer
        self.output_dim = output_dim
        self.epochs = epochs
    
    def dataProduce(self):
        import numpy as np
        from sklearn.model_selection import train_test_split
        fetchData = dataPrepare()
        self.X,self.y = fetchData.deepLearningInput()
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.3,shuffle=True, random_state=34)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def plotGraph(self,aList,label_title):
        immedDir = Path(__file__).parent.parent
        path_to_save = os.path.join(immedDir,'Images/NeuralNetwork')

        # Check if the output folder path present if not create it
        if not os.path.exists(os.path.join(immedDir, "Images")):
            os.mkdir(os.path.join(immedDir, "Images"))
        
        if not os.path.exists(os.path.join(immedDir, "Images/NeuralNetwork")):
            os.mkdir(os.path.join(immedDir, "Images/NeuralNetwork"))
        
        plt.plot(aList, label=label_title)
        plt.legend()
        plt.title(f'model {label_title}')
        plt.ylabel(f'{label_title}')
        plt.xlabel('epoch')
        plt.savefig( os.path.join( path_to_save, f"{label_title}Plotfor"+self.path.split("/")[-1].split(".")[0]+".png" ))
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    def trainCalculation(self,epochs):
        self.X_train,self.X_test,self.y_train,self.y_test = self.dataProduce()
        x_train = Variable(torch.from_numpy(self.X_train)).float()
        y_train = Variable(torch.from_numpy(self.y_train.values)).long()
        train_loss,train_acc = [],[] 

        self.model = Model(self.input_dim, self.hidden_layer, self.output_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Calculating the model loss and accuracy for train dataset
        self.model.train()
        for epoch in range(epochs):
            for batch in range(10):
                optimizer.zero_grad()
                y_pred = self.model(x_train)
                loss = criterion(y_pred, y_train)
                print ("epoch #",epoch)
                print ("loss: ", loss.item())
                train_loss.append(loss.item())

                pred = torch.max(y_pred, 1)[1].eq(y_train).sum()
                print ("acc:(%) ", (100*pred/len(x_train)).item())
            
            train_acc.append((100*pred/len(x_train)).item())
            loss.backward()
            optimizer.step()

        self.plotGraph(train_loss,"Loss")
        self.plotGraph(train_acc,"Accuracy")
        return max(train_acc)

    def testCaluculation(self):
        # torch.manual_seed(0)
        # self.X_train,self.X_test,self.y_train,self.y_test = self.dataProduce()
        test_loss,test_acc = 0,0

        ##model = Model(self.input_dim, self.hidden_layer, self.output_dim)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        self.model.eval()
        x_test = Variable(torch.from_numpy(self.X_test)).float()
        y_test = Variable(torch.from_numpy(self.y_test.to_numpy())).long()

        with torch.no_grad():
            y_pred = self.model(x_test)
            loss = criterion(y_pred, y_test)
            pred = torch.max(y_pred, 1)[1].eq(y_test).sum()
            print ("Test loss: ", loss.item())
            print ("Test acc (%): ", (100*pred/len(x_test)).item())
        test_loss += loss.item()
        test_acc += (100*pred/len(x_test)).item()
        return test_acc

    def run(self):
        train = self.trainCalculation(self.epochs)
        print()
        test = self.testCaluculation()
        
        vals = ['Train', 'Test']
        plt.bar(vals,[train,test])
        plt.title("Test Train accuracy")
        # plt.xlabel()
        plt.savefig('Code/Images/NeuralNetwork/Test Train accuracy.png')
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        plt.show()
        return test


if __name__ == '__main__':
    NeuralNet('YoutubeComplete.csv',1000,5,2,10).run()
