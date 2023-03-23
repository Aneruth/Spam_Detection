import numpy as np
from config.core import config
from pipeline import CreatePipeline
from Preprocess.preprocess import Parser
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def run_training() -> tuple:
    """ Train the model
    @return: A tuple of X_train, X_test, y_train, y_test
    """

    # read training data
    parser = Parser()
    data = parser.load_dataset()
    features = data[config.model_config.features]
    target = data[config.model_config.target]
    pipeline = CreatePipeline(features).create_pipeline()

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        features,  # predictors after tfidf
        target,  # target
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    # persist trained model
    parser.save_pipeline(pipeline_to_persist=pipeline)

    return X_train, X_test, y_train, y_test


def vectorize():
    X_tr, X_ts, y_tr, y_ts = run_training()

    # Since we are using the text as input, we need to convert it to a sparse matrix
    # We are using the tfidf method to convert our text to a sparse matrix
    tfidf = TfidfVectorizer(encoding="latin-1", strip_accents="unicode", stop_words="english")

    # Fit and transform the training data
    # For strange reason I need to convert the list to a string
    feat = tfidf.fit_transform(X_tr['CONTENT'].tolist())

    print(X_tr.shape, feat.shape)

    # TODO: Use this inside the model and containerize it using docker


if __name__ == "__main__":

    vectorize()
