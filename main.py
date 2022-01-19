import sys,os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.insert(0,os.path.join('Code/Models'))

import SupportVectorMachine,NaiveBayes,RandomForest,nnPytorch,lstm

data = 'YoutubeComplete.csv'

svm = SupportVectorMachine.SupportVector(data)
svm.supportVectorMachine()

rf = RandomForest.randomForest(data).metrics()

nb = NaiveBayes.NaiveBayes(data).naiveBayes()

nerural = nnPytorch.NeuralNet(data,1000,100,2,30).run()

rnn = lstm.LSTMImplement().run()