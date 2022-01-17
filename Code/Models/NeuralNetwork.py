from posixpath import abspath
import sys,os
from pathlib import Path
sys.path.insert(0,os.path.join('Code/Preprocess'))
from DataPreparation import dataPrepare

# Package for our CNN model
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout

class NeuralNet:

    def __init__(self,path) -> None:
        self.path = path 
    
    def neuralNetModelInput(self):
        fetchData = dataPrepare()
        X_train, X_test, y_train, y_test = fetchData.Vectorizer(self.path)
        return X_train, X_test, y_train, y_test
        
    def nnConfiguration(self):
        """A Neural Network Model build where we configure the NN.

        Returns:
            - int: Accuracy of our model
            - hist: History of our model which is useful for plotting
        """
        X_train, X_test, y_train, y_test = self.neuralNetModelInput()
        validation_dataset = (X_test.todense(),y_test)
        input_dimension = X_train.shape[1]
        neural = Sequential()
        neural.add(Dense(24, input_dim = input_dimension, activation='relu')) # our input feature count is 1010 then we need to reduce the diemensions
        neural.add(Dropout(0.2, input_shape=(input_dimension,)))
        neural.add(Dense(12,activation='relu'))
        neural.add(Dense(8, activation='relu'))
        neural.add(Dense(1, activation='sigmoid'))
        neural.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(neural.summary())
        # Keras fit for our dataset
        hist = neural.fit(X_train.todense(), y_train, epochs=20, validation_data = validation_dataset, shuffle=True)

        # evaluate the keras model
        _, neural_accuracy = neural.evaluate(X_train.todense(),y_train)
        acc = (neural_accuracy*100)
        
        print(f'Accuracy of our model is: {acc}')
        return acc,hist

    def metrices(self):
        acc,hist = self.nnConfiguration()
        # matplotlib package
        import matplotlib.pyplot as plt

        # summarize history for accuracy
        plt.plot(hist.history['accuracy'])
        plt.plot(hist.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        immedDir = Path(__file__).parent.parent
        # parentDir = os.path.dirname(abspath(immedDir))

        path_to_save = os.path.join(immedDir,'Images/NeuralNetwork')

        # Check if the output folder path present if not create it
        if not os.path.exists(os.path.join(immedDir, "Images")):
            os.mkdir(os.path.join(immedDir, "Images"))
        
        if not os.path.exists(os.path.join(immedDir, "Images/NeuralNetwork")):
            os.mkdir(os.path.join(immedDir, "Images/NeuralNetwork"))

        plt.savefig( os.path.join( path_to_save, "NeuralNetworkAccPlotfor"+self.path.split("/")[-1].split(".")[0]+".png" ))
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
        plt.savefig( os.path.join( path_to_save, "NeuralNetworkLossPlotfor"+self.path.split("/")[-1].split(".")[0]+".png" ))
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        return acc

class NeuralNetPca():
    def __init__(self, path) -> None:
        self.path = path
    
    def nnPCA(self):
        from PCA import PrincipleComponentAnalysis

        pcaInput = PrincipleComponentAnalysis(self.path)
        nnPca = NeuralNet(self.path)
        X_train, X_test, y_train, y_test = nnPca.neuralNetModelInput()
        pca_std,X_pca_test, X_pca_train,y_test,y_train = pcaInput.produceData(X_train, X_test, y_train, y_test)
        
        from keras.layers import GaussianNoise
        model = Sequential()

        model.add(Dense(12, input_dim=100, activation='relu'))
        model.add(GaussianNoise(pca_std))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # https://stackoverflow.com/questions/61742556/valueerror-shapes-none-1-and-none-2-are-incompatible
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        his = model.fit(X_pca_train, y_train, epochs=100, batch_size=256, validation_split=0.15, verbose=2,validation_data=(X_pca_test, y_test), shuffle=True)

        import matplotlib.pyplot as plt
        # summarize history for accuracy
        plt.plot(his.history['accuracy'])
        plt.plot(his.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        
        immedDir = Path(__file__).parent.parent
        # parentDir = os.path.dirname(abspath(immedDir))

        path_to_save = os.path.join(immedDir,'Images/NeuralNetwork/PCA')

        # Check if the output folder path present if not create it
        if not os.path.exists(os.path.join(immedDir, "Images")):
            os.mkdir(os.path.join(immedDir, "Images"))
        
        if not os.path.exists(os.path.join(immedDir, "Images/NeuralNetwork/PCA")):
            os.mkdir(os.path.join(immedDir, "Images/NeuralNetwork/PCA"))
        
        plt.savefig( os.path.join( path_to_save, "PCA_AccuracyPlotfor"+self.path.split("/")[-1].split(".")[0]+".png" ))
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        plt.plot(his.history['loss'])
        plt.plot(his.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig( os.path.join( path_to_save, "PCA_LossPlotfor"+self.path.split("/")[-1].split(".")[0]+".png" ))
        plt.show(block=False)
        plt.pause(3)
        plt.close()


if __name__ == '__main__':
    NeuralNet('YoutubeComplete.csv').metrices()