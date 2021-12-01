import sys,os
sys.path.insert(0, '/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Models')

import SupportVectorMachine,NaiveBayes

# SVM for all dataset
for data in sorted(os.listdir('/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Data/')):
    ml = SupportVectorMachine.SupportVector('/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Data/'+data)
    ml.metrics()


# For Naive bayes Algorithm
for data in sorted(os.listdir('/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Data/')):
    ml = NaiveBayes.Naive_Bayes('/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Data/'+data)
    ml.metrics()