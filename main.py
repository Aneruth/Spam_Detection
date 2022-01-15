import sys,os
sys.path.insert(0,os.path.join('Code/Models'))

import SupportVectorMachine,NaiveBayes,RandomForest,NeuralNetwork

data = 'YoutubeComplete.csv'

svm = SupportVectorMachine.SupportVector(data)
svm.supportVectorMachine()

rf = RandomForest.randomForest(data).metrics()

nb = NaiveBayes.NaiveBayes(data).naiveBayes()

nerural = NeuralNetwork.NeuralNet(data).metrices()