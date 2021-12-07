import sys

import tensorflow as tf
sys.path.insert(0, '/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Preprocess')
from DataPreparation import dataPrepare

from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.models import Sequential

import warnings
warnings.filterwarnings("ignore")

class Lstm:

    def __init__(self,path) -> None:
        self.path = path
    
    def lstm(self):
        fetchData = dataPrepare()
        X_train, X_test, y_train, y_test = fetchData.Vectorizer(self.path)
        X_train = tf.expand_dims(X_train.todense(), axis=-1)
        # The LSTM architecture
        regressor = Sequential()
        # First LSTM layer with Dropout regularisation
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50, return_sequences=True)) # Second LSTM layer
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50, return_sequences=True)) # Third LSTM layer
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units=1)) # The output layer

        # Compiling the RNN
        regressor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Fitting to the training set
        regressor.fit(X_train,y_train,epochs=1,batch_size=1)

if __name__ == '__main__':
    nerural = Lstm('/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Data/Youtube01-Psy.csv')
    nerural.lstm()