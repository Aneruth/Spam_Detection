import sys
import tensorflow as tf
sys.path.append("../")
# sys.path.insert(0, '/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Preprocess')

import torch
import torch.nn as nn

import tensorflow as tf # Adding padding to our dataset (train and test models)
import numpy as np
from Preprocess.DataPreparation import dataPrepare

import warnings
warnings.filterwarnings("ignore")

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


class Data:

    def __init__(self,path) -> None:
        self.path = path

    def getData(self):
        """Considers the path of the dataset which holds the sparse matrix. With the help of tensorflow we are
        creating the padding for our dataset (X_train and X_test).

        Returns:
            Tensor: Tensor data with padding and dataframe
        """
        fetchData = dataPrepare()
    
        X_train, X_test, y_train, y_test =  fetchData.Vectorizer(self.path)

        # Adding the padding to our sparse matrix
        X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train.todense())
        X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test.todense())

        # make training and test sets in torch
        x_train = torch.from_numpy(X_train).type(torch.Tensor)
        x_test = torch.from_numpy(X_test).type(torch.Tensor)
        y_train = torch.from_numpy(y_train.to_numpy()).type(torch.Tensor)
        y_test = torch.from_numpy(y_test.to_numpy()).type(torch.Tensor)
        
        print(f'Size of our y-train dataset is {y_train.size()} and X-train dataset is {x_train.size()}')

        return x_train,x_test,y_train,y_test

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        """LSTM model which takes the following input:
                - Input Dimension
                - Hidden Layers (Number)
                - Total Number of Layers
                - Ouput Diemension

        Args:
            input_dim (int): Length of the dataset
            hidden_dim (int): Total hidden layer
            num_layers (int): Total layers to be present
            output_dim (int): Number of output diemension layer
        """
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out
    
class LSTMImplement:

    def __init__(self,path, hidden_dim, num_layers, output_dim) -> None:
        """A cosntructor which takes the path of the dataset, number of hidden dimensions, number of layers and output
        diemensions as our parameters. This will plot our accuracy plot and loss plot.

        Args:
            path (String): Path of our dataset
            hidden_dim (int): Total hidden layer
            num_layers (int): Total layers to be present
            output_dim (int): Number of output diemension layer
        """
        self.path = path
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

    def run(self):
        fetchData = dataPrepare()
        X_train, X_test, y_train, y_test =  fetchData.Vectorizer(self.path)
        
        x_train,x_test,y_train,y_test = Data(self.path).getData()
        input_dimension = X_train.shape[1]
        model = LSTM(input_dim=input_dimension, hidden_dim=self.hidden_dim, output_dim=self.output_dim, num_layers=self.num_layers)

        loss_fn = torch.nn.MSELoss()

        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        print(model)
        print(len(list(model.parameters())))
        for i in range(len(list(model.parameters()))):
            print(list(model.parameters())[i].size())

        num_epochs = 1000
        loss_val  = np.zeros(num_epochs)
        acc_val  = np.zeros(num_epochs)

        for t in range(num_epochs):
            # Forward pass
            y_train_pred = model(x_train)

            loss = loss_fn(y_train_pred, y_train)
            if t % 10 == 0 and t !=0:
                print(f"Epoch {t} MSE is {loss.item()}")
            loss_val[t] = loss.item()
            
            pred = torch.max(y_train_pred, 1)[1].eq(y_train).sum()
            # pred = model_accuracy(y_train_pred,y)
            if t % 10 == 0 and t !=0:
                print(f"Epoch {t} accuracy(%) is {(100*pred/len(y_train)).item()}")
            acc_val[t] = (100*pred/len(y_train)).item()
            # Zero out gradient, else they will accumulate between epochs
            optimiser.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimiser.step()
        
        import matplotlib.pyplot as plt
        plt.plot(loss_val, label="Training loss")
        plt.legend()
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig(f'/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Images/LSTM/LossPlotfor{self.path.split("/")[-1].split(".")[0]}.png')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        plt.plot(acc_val, label="Accuracy")
        plt.legend()
        plt.show()

class LSTMPca():
    def __init__(self, path, hidden_dim, num_layers, output_dim) -> None:
        self.path = path
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

    def lstmPCA(self):
            from PCA import PrincipleComponentAnalysis

            pcaInput = PrincipleComponentAnalysis(self.path)
            lstmPca = LSTMImplement(self.path)
            X_train, X_test, y_train, y_test = lstmPca.neuralNetModelInput()
            pca_std,X_pca_test, X_pca_train,y_test,y_train = pcaInput.produceData(X_train, X_test, y_train, y_test)

            input_dimension = X_train.shape[1]
            model = LSTM(input_dim=input_dimension, hidden_dim=self.hidden_dim, output_dim=self.output_dim, num_layers=self.num_layers)

            loss_fn = torch.nn.MSELoss()

            optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
            print(model)
            print(len(list(model.parameters())))
            for i in range(len(list(model.parameters()))):
                print(list(model.parameters())[i].size())

            num_epochs = 1000
            loss_val  = np.zeros(num_epochs)
            acc_val  = np.zeros(num_epochs)

            for t in range(num_epochs):
                # Forward pass
                y_train_pred = model(X_pca_train)

                loss = loss_fn(y_train_pred, y_train)
                if t % 10 == 0 and t !=0:
                    print(f"Epoch {t} MSE is {loss.item()}")
                loss_val[t] = loss.item()
                
                pred = torch.max(y_train_pred, 1)[1].eq(y_train).sum()
                # pred = model_accuracy(y_train_pred,y)
                if t % 10 == 0 and t !=0:
                    print(f"Epoch {t} accuracy(%) is {(100*pred/len(y_train)).item()}")
                acc_val[t] = (100*pred/len(y_train)).item()
                # Zero out gradient, else they will accumulate between epochs
                optimiser.zero_grad()

                # Backward pass
                loss.backward()

                # Update parameters
                optimiser.step()
            
            import matplotlib.pyplot as plt
            plt.plot(loss_val, label="Training loss")
            plt.legend()
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.savefig(f'/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Images/LSTM/LossPlotforLSTMPCA{self.path.split("/")[-1].split(".")[0]}.png')
            plt.show(block=False)
            plt.pause(3)
            plt.close()

            plt.plot(acc_val, label="Accuracy")
            plt.legend()
            plt.show()

# Testing
if __name__ == "__main__":
    lstm = LSTMImplement(path='/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Data/Youtube01-Psy.csv',hidden_dim=28,num_layers=4,output_dim=1)
    lstm.run()