from posixpath import abspath
import sys,os
from pathlib import Path
sys.path.insert(0,os.path.join('spam_classifier/Preprocess'))
from DataPreparation import dataPrepare

import warnings
warnings.filterwarnings("ignore")

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

class NaiveBayes:

    def __init__(self,path) -> None:
        self.path = path
    
    def nb(self):
        """The Bayes' Theorem is used to preprocess for classification techniques known as 
        Naive Bayes classifiers. It is a family of algorithms that share a similar idea, 
        namely that each pair of features being classified is independent of the others.

        Returns:
            [list]: returns label test data and prediction values
        """
        fetchData = dataPrepare()
        self.X_train, self.X_test, self.y_train, self.y_test = fetchData.Vectorizer(self.path)

        #Create a Gaussian Classifier
        nb = GaussianNB()

        # Train the model using the training sets
        # https://stackoverflow.com/questions/30502284/a-sparse-matrix-was-passed-but-dense-data-is-required-use-x-toarray-to-conve/37248794
        nb.fit(self.X_train.todense(),self.y_train)

        nb_prediction = nb.predict(self.X_test.todense())
        
        # nb_accuracy_score = round(accuracy_score(y_test, nb_prediction)*100,2)
        # print(f"Accuracy score for Naive bayes is: {nb_accuracy_score}%")
        return self.y_test,nb_prediction
    
    def naiveBayes(self):
        """A function which calculates all the metrices and plots the graph.

        Returns:
            float: Returns the predcited score value
        """
        import matplotlib.pyplot as plt

        self.y_test,nb_prediction = self.nb()

        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        print(f'The confusion matrix for Support Vector Machine before hyper parameter tunning is \n{confusion_matrix(self.y_test,nb_prediction)}')

        print(f'\n Classification report for Support Vector Machine before hyper parameter tunning is:\n{classification_report(self.y_test,nb_prediction)}')

        acc_score_before_hyper = round(accuracy_score(self.y_test,nb_prediction)*100,2)
        print(f'Accuracy Score before hyper parameter tunning is: {acc_score_before_hyper}%')

        '''
        Hpyperparameter tuning the model with the help of grid search

        Performing grid search which is a type of cross validation for Support Vector Machine
        - C is a hypermeter which is set before the training model and used to control error.
        - Gamma is also a hypermeter which is set before the training model and used to give curvature weight of the decision boundary.
        - More gamma more the curvature and less gamma less curvature.
        '''
        # Initialise the gird search variable 
        import numpy as np
        param_grid_nb = {'var_smoothing': np.logspace(0,-9, num=100)}

        '''
        Important parameter for grid search 
            - verbose --> get the quality of the data more the verbose more the quality
            - scoring --> a parameter where we define what we need to infer from the dataset
        '''
        from sklearn.model_selection import GridSearchCV
        # feed the search variable to the grid search 
        grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)

        # Grid Search fixable
        grid.fit(self.X_train.todense(),self.y_train)

        # to get the best parameter fit from the grid
        print('The best parameter is:',grid.best_params_)
        
        # to get the predictions from the grid
        grid_predict = grid.predict(self.X_test.todense())

        # To print the classification report and confusion matrix for the SVM
        print('Report for Suport Vector Machine')
        # To print the confusion matrix
        print('The confusion matrix after grid search is:' + '\n \n',confusion_matrix(self.y_test,grid_predict))

        # To print the classification report
        print('\nThe classification report after grid search is:' + f'\n{classification_report(self.y_test,grid_predict)}')
       
        immedDir = Path(__file__).parent.parent
        # parentDir = os.path.dirname(abspath(immedDir))
       
        # to get the best score from the grid
        print('The best score is:',((grid.best_score_)*100).round(2),'%')
        print()
        s = [acc_score_before_hyper,((grid.best_score_)*100).round(2)]
        n = ['Before Hyperparameter','After Hyperparameter']
        plt.figure(figsize=(12,6))
        plt.title('Graph to compare accuracy score before and after hyperparameter tunning')
        plt.xlabel('Naive Bayes Algorithm')
        plt.ylabel('Accuracy Score')
        plt.bar(['Before Hyperparameter','After Hyperparameter'],[acc_score_before_hyper,((grid.best_score_)*100).round(2)])
        for i in range(len(s)):
            plt.annotate(str(s[i]), xy=(n[i],s[i]), ha='center', va='bottom')
        
        path_to_save = os.path.join(immedDir, "Images/NaiveBayes")
        
        # Check if the output folder path present if not create it
        if not os.path.exists(os.path.join(immedDir, "Images")):
            os.mkdir(os.path.join(immedDir, "Images"))
        
        if not os.path.exists(os.path.join(immedDir, "Images/NaiveBayes")):
            os.mkdir(os.path.join(immedDir, "Images/NaiveBayes"))
        
        plt.savefig( os.path.join( path_to_save, "NaiveBayesAccPlotfor"+self.path.split("/")[-1].split(".")[0]+".png" ))
        
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        return ((grid.best_score_)*100).round(2)