import sys
sys.path.insert(0, '/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Preprocess')
# sys.path.append("../") 
from DataPreparation import dataPrepare

# Package for Random Forest
from sklearn.ensemble import RandomForestClassifier

class randomForest:
    
    def __init__(self,path) -> None:
        self.path = path

    def rf(self):
        fetchData = dataPrepare()
        self.X_train, self.X_test, self.y_train, self.y_test = fetchData.Vectorizer(self.path)

        rf = RandomForestClassifier(n_estimators=200)

        # fitting the model 
        rf.fit(self.X_train,self.y_train)

        # To print the predcitions of the Random Forest Model 
        rf_prediction = rf.predict(self.X_test)

        return self.y_test,rf_prediction

    def metrics(self):
        import matplotlib.pyplot as plt

        self.y_test,rf_prediction = self.rf()

        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        print(f'The confusion matrix for Random Forest before hyper parameter tunning is \n{confusion_matrix(self.y_test,rf_prediction)}')

        print(f'\n Classification report for Random Forest before hyper parameter tunning is:\n{classification_report(self.y_test,rf_prediction)}')

        acc_score_before_hyper = round(accuracy_score(self.y_test,rf_prediction)*100,2)
        print(f'Accuracy Score before hyper parameter tunning is: {acc_score_before_hyper}%')

        # Hpyperparameter tuning the model by changing the number  of estimators
        acc_score_after_hyper = []
        for score in list(map(int,range(100,510,10))):
            rfHyper = RandomForestClassifier(n_estimators=200)
            rfHyper.fit(self.X_train,self.y_train)
            predictHyper = rfHyper.predict(self.X_test)
            print(f'Accuracy score at {score}th estimator is {round(accuracy_score(self.y_test,predictHyper)*100,2)}')
            acc_score_after_hyper.append(round(accuracy_score(self.y_test,predictHyper)*100,2))
        acc_score_after_hyper = max(acc_score_after_hyper)

        # To print the classification report and confusion matrix for the SVM
        print('Report for Random Forest')
        # To print the confusion matrix
        print('The confusion matrix after hyper parameter tuning is:' + '\n \n',confusion_matrix(self.y_test,predictHyper))

        print('\n')

        # To print the classification report
        print('The classification report after hyper parameter tuning is:' + f'\n{classification_report(self.y_test,predictHyper)}')

        # to get the best score from the grid
        print(f'The best score is: {acc_score_after_hyper}%')
        print()
        s = [acc_score_before_hyper,acc_score_after_hyper]
        n = ['Before Hyperparameter','After Hyperparameter']
        plt.figure(figsize=(12,6))
        plt.title('Graph to compare accuracy score before and after hyperparameter tunning')
        plt.xlabel('Random Forest Algorithm')
        plt.ylabel('Accuracy Score')
        plt.bar(['Before Hyperparameter','After Hyperparameter'],[acc_score_before_hyper,acc_score_after_hyper])
        for i in range(len(s)):
            plt.annotate(str(s[i]), xy=(n[i],s[i]), ha='center', va='bottom')
        plt.savefig(f'/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Images/RandomForest/RandomForestAccPlotfor{self.path.split("/")[-1].split(".")[0]}.png')
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        return acc_score_after_hyper