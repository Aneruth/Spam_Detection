import sys
sys.path.insert(0, '/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Preprocess')

from DataPreparation import dataPrepare
# Importing the package for SVM algorithm
from sklearn.svm import SVC

class SupportVector:
    
    def __init__(self,path) -> None:
        self.path = path

    def svm(self):
        
        fetchData = dataPrepare()
        self.X_train, self.X_test, self.y_train, self.y_test = fetchData.Vectorizer(self.path)
        
        # Creating a support vector model
        self.sv = SVC()

        # Fitting our data
        self.sv.fit(self.X_train,self.y_train)

        # Predicting our SVM Model
        self.SVMprediction = self.sv.predict(self.X_test)

        return self.y_test,self.SVMprediction
    
    def metrics(self):
        self.y_test,self.SVMprediction = self.svm()

        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        print(f'The confusion matrix for Support Vector Machine before hyper parameter tunning is \n{confusion_matrix(self.y_test,self.SVMprediction)}')

        print(f'\n Classification report for Support Vector Machine before hyper parameter tunning is:\n{classification_report(self.y_test,self.SVMprediction)}')

        acc_score_before_hyper = round(accuracy_score(self.y_test,self.SVMprediction)*100,2)
        print(f'Accuracy Score before hyper parameter tunning is: {acc_score_before_hyper}%')

        '''
        Hpyperparameter tuning the model with the help of grid search

        Performing grid search which is a type of cross validation for Support Vector Machine
        - C is a hypermeter which is set before the training model and used to control error.
        - Gamma is also a hypermeter which is set before the training model and used to give curvature weight of the decision boundary.
        - More gamma more the curvature and less gamma less curvature.
        '''
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
        print('Report for Supoort Vector Machine')
        # To print the confusion matrix
        print('The confusion matrix after grid search is:' + '\n \n',confusion_matrix(self.y_test,grid_predict))

        print('\n')

        # To print the classification report
        print('The classification report after grid search is:' + '\n \n',classification_report(self.y_test,grid_predict))

        # to get the best score from the grid
        print('The best score is:',((grid.best_score_)*100).round(2),'%')
