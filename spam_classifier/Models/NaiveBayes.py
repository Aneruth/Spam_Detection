import warnings
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

    def predict_text(self, text: str) -> str:
        """Predict the text

        Args:
            text (str): The text to be predicted

        Returns:
            str: The predicted text
        """
        mapping = {0: "ham", 1: "spam"}
        return mapping[train_pipeline.predict_text(self.model, text)[0]]


if __name__ == "__main__":
    nb = NaiveBayes()
    nb.run_model()
    print(nb.get_score())

    print(nb.predict_text("Hi there, how are you doing?"))