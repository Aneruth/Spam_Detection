import sys
sys.path.insert(0, '/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Models')

import SupportVectorMachine

ml = SupportVectorMachine.SupportVector('/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Data/Youtube01-Psy.csv')
ml.metrics()