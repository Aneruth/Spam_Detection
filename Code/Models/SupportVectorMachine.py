from posixpath import abspath
import sys,os
from pathlib import Path
sys.path.insert(0,os.path.join('Code/Preprocess'))
from DataPreparation import dataPrepare
# Importing the package for SVM algorithm
from sklearn.svm import SVC

class SupportVector:
    """Support vector machine is highly preferred by many as it produces significant accuracy 
    with less computation power. Support Vector Machine, abbreviated as SVM can be used for 
    both regression and classification tasks. But, it is widely used in classification objectives.
    """
    
    def __init__(self,path) -> None:
        self.path = path

    def svm(self):
        """The objective of the support vector machine algorithm is to find a hyperplane in an 
        N-dimensional space(N — the number of features) that distinctly classifies the data points

        Returns:
            list: returns label test data and prediction values
        """
        fetchData = dataPrepare()
        self.X_train, self.X_test, self.y_train, self.y_test = fetchData.Vectorizer(self.path)
        
        # Creating a support vector model
        self.sv = SVC()

        # Fitting our data
        self.sv.fit(self.X_train,self.y_train)

        # Predicting our SVM Model
        self.SVMprediction = self.sv.predict(self.X_test)

        return self.y_test,self.SVMprediction
    
    def supportVectorMachine(self):
        """A function which calculates all the metrices and plots the graph.
        
        Hpyperparameter tuning the model with the help of grid search

        Performing grid search which is a type of cross validation for Support Vector Machine
        - C is a hypermeter which is set before the training model and used to control error.
        - Gamma is also a hypermeter which is set before the training model and used to give curvature weight of the decision boundary.
        - More gamma more the curvature and less gamma less curvature.

        Returns:
            float: Returns the predcited score value
        """
        import matplotlib.pyplot as plt

        self.y_test,self.SVMprediction = self.svm()

        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        print(f'The confusion matrix for Support Vector Machine before hyper parameter tunning is \n{confusion_matrix(self.y_test,self.SVMprediction)}')

        print(f'\nClassification report for Support Vector Machine before hyper parameter tunning is:\n{classification_report(self.y_test,self.SVMprediction)}')

        acc_score_before_hyper = round(accuracy_score(self.y_test,self.SVMprediction)*100,2)
        print(f'Accuracy Score before hyper parameter tunning is: {acc_score_before_hyper}%')

        # Initialise the gird search variable 
        pg = {"C":[0.1,1,10,100,1000],"gamma":[1,.1,.01,.001,.0001]}

        '''
        Important parameter for grid search 
        - verbose --> get the quality of the data more the verbose more the quality
        - scoring --> a parameter where we define what we need to infer from the dataset
        '''
        from sklearn.model_selection import GridSearchCV
        # feed the search variable to the grid search 
        grid = GridSearchCV(SVC(),pg,verbose=3,scoring='accuracy')

        # Grid Search fixable
        grid.fit(self.X_train,self.y_train)

        # to get the best parameter fit from the grid
        print('The best parameter is:',grid.best_params_)
        
        # to get the predictions from the grid
        grid_predict = grid.predict(self.X_test)

        # To print the classification report and confusion matrix for the SVM
        print('Report for Suport Vector Machine')
        # To print the confusion matrix
        print('The confusion matrix after grid search is:' + '\n \n',confusion_matrix(self.y_test,grid_predict))

        print('\n')

        # To print the classification report
        print('The classification report after grid search is:' + f'\n{classification_report(self.y_test,grid_predict)}')

        # to get the best score from the grid
        print('The best score is:',((grid.best_score_)*100).round(2),'%')
        print()
        s = [acc_score_before_hyper,((grid.best_score_)*100).round(2)]
        n = ['Before Hyperparameter','After Hyperparameter']
        
        immedDir = Path(__file__).parent.parent
        # parentDir = os.path.dirname(abspath(immedDir))
        
        plt.figure(figsize=(12,6))
        plt.title('Graph to compare accuracy score before and after hyperparameter tunning')
        plt.xlabel('SVM Algorithm')
        plt.ylabel('Accuracy Score')
        plt.bar(['Before Hyperparameter','After Hyperparameter'],[acc_score_before_hyper,((grid.best_score_)*100).round(2)])
        for i in range(len(s)):
            plt.annotate(str(s[i]), xy=(n[i],s[i]), ha='center', va='bottom')
        
        path_to_save = os.path.join(immedDir,'Images/SVM')
        
        # Check if the output folder path present if not create it
        if not os.path.exists(path_to_save):
            os.mkdir(os.path.join(immedDir, "Images"))
            os.mkdir(os.path.join(immedDir, "Images/SVM"))
        
        plt.savefig( os.path.join( path_to_save, "svmAccPlotfor"+self.path.split("/")[-1].split(".")[0]+".png" ))
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        return ((grid.best_score_)*100).round(2)