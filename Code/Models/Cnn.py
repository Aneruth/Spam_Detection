import sys
sys.path.insert(0, '/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Preprocess')
from DataPreparation import dataPrepare

# Package for our CNN model
from keras.models import Sequential
from keras.layers import Dense
import warnings
warnings.filterwarnings("ignore")
class ConvNeuralNet:

    def __init__(self,path) -> None:
        self.path = path 
    
    def cnn(self):
        
        fetchData = dataPrepare()
        X_train, X_test, y_train, y_test = fetchData.Vectorizer(self.path)
        
        neural = Sequential()
        neural.add(Dense(12, input_dim = X_train.shape[1], activation='relu')) # our input feature count is 1010 then we need to reduce the diemensions
        neural.add(Dense(8, activation='relu'))
        neural.add(Dense(1, activation='sigmoid'))

        neural.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(neural.summary())
        # Keras fit for our dataset
        hist = neural.fit(X_train.todense(), y_train, epochs=20, validation_data=(X_test.todense(), y_test), shuffle=True)

        # evaluate the keras model
        _, neural_accuracy = neural.evaluate(X_train.todense(),y_train)
        acc = (neural_accuracy*100)
        
        print(f'Accuracy of our model is: {acc}')

        # matplotlib package
        import matplotlib.pyplot as plt

        # summarize history for accuracy
        plt.plot(hist.history['accuracy'])
        plt.plot(hist.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        
        plt.savefig(f'/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Images/NeuralNetwork/AccuracyPlotfor{self.path.split("/")[-1].split(".")[0]}.png')
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        
        # summarize history for loss
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Images/NeuralNetwork/LossPlotfor{self.path.split("/")[-1].split(".")[0]}.png')
        plt.show(block=False)
        plt.pause(3)
        plt.close()


'''if __name__ == '__main__':
    nerural = ConvNeuralNet('/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Data/Youtube01-Psy.csv')
    nerural.cnn()'''
        