from posixpath import abspath
import sys,os
from pathlib import Path

# Lstm import
from keras.layers import LSTM, Embedding, Dense, Flatten
from keras.models import Sequential

sys.path.insert(0,os.path.join('Code/Preprocess'))

import tensorflow as tf # Adding padding to our dataset (train and test models)
import numpy as np
from DataPreparation import dataPrepare # package to fetch all the data

import warnings
warnings.filterwarnings("ignore")

path = 'YouTubeComplete.csv'
class LSTMImplement:

    def __init__(self) -> None:
        """A cosntructor which takes the path of the dataset, number of hidden dimensions, number of layers and output
        diemensions as our parameters. This will plot our accuracy plot and loss plot.

        Args:
            path (String): Path of our dataset
            hidden_dim (int): Total hidden layer
            num_layers (int): Total layers to be present
            output_dim (int): Number of output diemension layer
        """
        # path = path
        pass
    
    def getData(self):
        """Considers the path of the dataset which holds the sparse matrix. With the help of tensorflow we are
        creating the padding for our dataset (X_train and X_test).

        Returns:
            Tensor: Tensor data with padding and dataframe
        """
        from sklearn.model_selection import train_test_split
        typeModel = 'lstm'
        X,y =  dataPrepare().deepLearningInput(typeModel)

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,shuffle=True, random_state=34)

        # Adding the padding to our sparse matrix
        X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train,maxlen=120)
        X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test,maxlen=120)
        return X_train,X_test,y_train,y_test

    def plotGraph(self,aList,label_title):
        import matplotlib.pyplot as plt

        immedDir = Path(__file__).parent.parent
        path_to_save = os.path.join(immedDir,'Images/LSTM')

        # Check if the output folder path present if not create it
        if not os.path.exists(os.path.join(immedDir, "Images")):
            os.mkdir(os.path.join(immedDir, "Images"))
        
        if not os.path.exists(os.path.join(immedDir, "Images/LSTM")):
            os.mkdir(os.path.join(immedDir, "Images/LSTM"))
        
        plt.plot(aList, label=label_title)
        plt.legend()
        plt.title(f'model {label_title}')
        plt.ylabel(f'{label_title}')
        plt.xlabel('epoch')
        plt.savefig( os.path.join( path_to_save, f"{label_title}Plotfor"+path.split("/")[-1].split(".")[0]+".png" ))
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    def run(self):

        X_train,X_test,y_train,y_test = self.getData()

        model = Sequential()
        model.add(Embedding(input_dim=6000, output_dim=32, input_length=120))
        model.add(LSTM(units=20,return_sequences=True))
        model.add(LSTM(units=16,return_sequences=False))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        model.summary()
        
        hist = model.fit(X_train, y_train, epochs=10, batch_size=64,validation_data=(X_test,y_test))


        immedDir = Path(__file__).parent.parent
        # parentDir = os.path.dirname(abspath(immedDir))

        path_to_save = os.path.join(immedDir,'Images/LSTM')

        # Check if the output folder path present if not create it
        if not os.path.exists(os.path.join(immedDir, "Images")):
            os.mkdir(os.path.join(immedDir, "Images"))
        
        if not os.path.exists(os.path.join(immedDir, "Images/LSTM")):
            os.mkdir(os.path.join(immedDir, "Images/LSTM"))
        
        self.plotGraph(hist.history['acc'],"Accuracy")
        self.plotGraph(hist.history['loss'],"Loss")


if __name__ == '__main__':
    LSTMImplement().run()