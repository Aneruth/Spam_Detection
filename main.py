import sys,os
sys.path.insert(0, '/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Models')

import SupportVectorMachine,NaiveBayes,RandomForest

# SVM for all dataset
for data in sorted(os.listdir('/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Data/')):
    ml = SupportVectorMachine.SupportVector('/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Data/'+data)
    ml.metrics()


# Naive bayes Algorithm for all dataset
for data in sorted(os.listdir('/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Data/')):
    ml = NaiveBayes.Naive_Bayes('/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Data/'+data)
    ml.metrics()

# Random forest Algorithm for all dataset
for data in sorted(os.listdir('/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Data/')):
    ml = RandomForest.randomForest('/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Data/'+data)
    ml.metrics()