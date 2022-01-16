import sys,os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
sys.path.insert(0,os.path.join('Code/Models'))

import SupportVectorMachine,NaiveBayes,RandomForest,NeuralNetwork,lstm

data = 'YoutubeComplete.csv'

svm = SupportVectorMachine.SupportVector(data)
svm.supportVectorMachine()

rf = RandomForest.randomForest(data).metrics()

nb = NaiveBayes.NaiveBayes(data).naiveBayes()

nerural = NeuralNetwork.NeuralNet(data).metrices()

rnn = lstm.LSTMImplement(data,28,4,1).run()

nerural = NeuralNetwork.NeuralNetPca(data).nnPCA()