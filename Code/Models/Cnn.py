import sys
sys.path.insert(0, '/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Preprocess')
from DataPreparation import dataPrepare

# Package for our CNN model
from keras.models import Sequential
from keras.layers import Dense

class ConvNeuralNet:

    def __init__(self,path) -> None:
        self.path = path 
    
    def cnn(self):
        fetchData = dataPrepare()
        X,y = fetchData.deepLearningInput(self.path)
        
        neural = Sequential()
        neural.add(Dense(12, input_dim=0, activation='relu')) # our input feature count is 1 so input_dim is 1
        neural.add(Dense(8, activation='relu'))
        neural.add(Dense(1, activation='sigmoid'))

        neural.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Keras fit for our dataset
        hist = neural.fit(X, y, epochs=150, batch_size=10,verbose=0)

        # evaluate the keras model
        _, neural_accuracy = neural.evaluate(X,y)
        acc = (neural_accuracy*100)
        
        print(f'Accuracy of our model is: {acc}')

        # matplotlib package
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15,8))
        # summarize history for accuracy
        plt.plot(hist.history['accuracy'])
        plt.plot(hist.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        # summarize history for loss
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    nerural = ConvNeuralNet('/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Data/Youtube01-Psy.csv')
    nerural.cnn()
