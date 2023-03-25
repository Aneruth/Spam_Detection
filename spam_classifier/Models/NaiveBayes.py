import os
import warnings
import pickle
from sklearn.naive_bayes import GaussianNB
from spam_classifier.pipelines import train_pipeline
from sklearn.metrics import *

warnings.filterwarnings("ignore")


class NaiveBayes:

    def __init__(self):
        self.y_hat = None
        self.y_test = None
        self.model = None

    def run_model(self):
        """The Bayes' Theorem is used to preprocess for classification techniques known as 
        Naive Bayes classifiers. It is a family of algorithms that share a similar idea, 
        namely that each pair of features being classified is independent of the others.
        """
        X_train, X_test, y_train, self.y_test = train_pipeline.vectorize()
        self.model = GaussianNB()
        self.model.fit(X_train.toarray(), y_train)
        self.y_hat = self.model.predict(X_test.toarray())
        self.save_model()

    def get_score(self) -> dict:
        """Get the accuracy score of the model

        Returns:
            dict: returns dictionary of accuracy, precision, recall and f1 score
        """
        accuracy = accuracy_score(self.y_test, self.y_hat)
        precision = precision_score(self.y_test, self.y_hat)
        recall = recall_score(self.y_test, self.y_hat)
        f1 = f1_score(self.y_test, self.y_hat)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def save_model(self):
        """Save the model
        """
        path = 'spam_classifier/trained_models/'
        full_path = os.path.join(path, 'model.pkl')
        with open(full_path, 'wb') as f:
            pickle.dump(self, f)

    def get_individual_score(self, input_string: str) -> dict:
        """Get the individual score of the model

        Args:
            input_string (str): The text to be predicted

        Returns:
            dict: returns dictionary of accuracy, precision, recall and f1 score
        """
        y_tst = train_pipeline.transform_text(input_string)
        y_hat = self.model.predict(y_tst)
        accuracy = accuracy_score(y_tst, y_hat)
        precision = precision_score(y_tst, y_hat)
        recall = recall_score(y_tst, y_hat)
        f1 = f1_score(y_tst, y_hat)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def predict_text(self, text: str) -> str:
        """Predict the text

        Args:
            text (str): The text to be predicted

        Returns:
            str: The predicted text
        """
        mapping = {0: "ham", 1: "spam"}
        result = self.model.predict(train_pipeline.transform_text(text))
        return mapping[result[0]]
