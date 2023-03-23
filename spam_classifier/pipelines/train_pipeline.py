from spam_classifier.config.core import config
from spam_classifier.pipelines.pipeline import CreatePipeline
from spam_classifier.Preprocess.preprocess import Parser
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# Since we are using the text as input, we need to convert it to a sparse matrix
# We are using the tfidf method to convert our text to a sparse matrix
tfidf = TfidfVectorizer(encoding="latin-1", strip_accents="unicode", stop_words="english")


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


def vectorize() -> tuple:
    """ Vectorize the text data using tfidf method
    @return: A tuple of transformed X_train, X_test and y_train, y_test
    """
    X_train, X_test, y_train, y_test = run_training()

    # Fit and transform the training data
    # For strange reason I need to convert the list to a string or a numpy array
    # NOTE: TFIDF is not working with pandas dataframe
    X_tr_transform = tfidf.fit_transform(X_train['CONTENT'])

    X_ts_transform = tfidf.transform(X_test['CONTENT'])

    return X_tr_transform, X_ts_transform, y_train, y_test


def predict_text(model: object, text: str) -> str:
    """Predict the text

    Args:
        model (object): The model to be used
        text (str): The text to be predicted

    Returns:
        str: The predicted text
    """
    text = tfidf.transform([text])
    return model.predict(text.toarray())
